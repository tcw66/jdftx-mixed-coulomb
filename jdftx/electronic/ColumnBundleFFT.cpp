/* Batched scalar-wavefunction FFTs. GPL v3 or later; part of JDFTx. */
#include <electronic/ColumnBundleFFT.h>
#include <electronic/ElecInfo.h>
#include <core/Operators.h>
#include <map>
#include <tuple>

#ifdef GPU_ENABLED
namespace {
struct FFTWorkspace {
	vector3<int> shape;
	std::map<int,cufftHandle> plans;
	ManagedArray<complex> values;
	ManagedArray<char> work;
	size_t workBytes=0;
	// Initialize allocator singletons before registering this static object's destructor.
	FFTWorkspace() { values.init(1,true); work.init(1,true); }
	~FFTWorkspace() { for(const auto& item:plans) cufftDestroy(item.second); }
	cufftHandle prepare(const GridInfo& grid,int batch) {
		if(!(shape==grid.S)) {
			for(const auto& item:plans) cufftDestroy(item.second);
			plans.clear(); shape=grid.S; workBytes=0;
		}
		// Retain the largest buffer instead of reallocating for partial batches.
		if(values.nData()<size_t(grid.nr)*batch) values.init(size_t(grid.nr)*batch,true);
		auto found=plans.find(batch);
		if(found==plans.end()) {
			cufftHandle plan; size_t bytes=0; int sizes[3]={shape[0],shape[1],shape[2]};
			if(cufftCreate(&plan)!=CUFFT_SUCCESS || cufftSetAutoAllocation(plan,0)!=CUFFT_SUCCESS
				|| cufftMakePlanMany(plan,3,sizes,nullptr,1,grid.nr,nullptr,1,grid.nr,CUFFT_Z2Z,batch,&bytes)!=CUFFT_SUCCESS)
				die("Cannot create batched wavefunction FFT plan.\n");
			plans.emplace(batch,plan);
			if(bytes>workBytes) {
				work.init(bytes,true); workBytes=bytes;
				for(const auto& item:plans) if(cufftSetWorkArea(item.second,work.dataGpu())!=CUFFT_SUCCESS)
					die("Cannot assign batched FFT workspace.\n");
			} else if(cufftSetWorkArea(plan,work.dataGpu())!=CUFFT_SUCCESS) die("Cannot assign batched FFT workspace.\n");
			found=plans.find(batch);
		}
		return found->second;
	}
};
FFTWorkspace& fftWorkspace() { static FFTWorkspace workspace; return workspace; }
int batchLimit() {
	static int limit=[]() {
		const char* setting=getenv("JDFTX_FFT_BATCH_SIZE");
		if(!setting) return 32;
		char* end=nullptr; long count=strtol(setting,&end,10);
		if(*end || count<0 || count>256) die("JDFTX_FFT_BATCH_SIZE must be between 0 and 256.\n");
		return int(count);
	}();
	return limit;
}
void inverseColumns(const ColumnBundle& C,int start,int count,FFTWorkspace& workspace,cufftHandle plan) {
	if(cudaMemset(workspace.values.dataGpu(),0,size_t(count)*C.basis->gInfo->nr*sizeof(complex))!=cudaSuccess)
		die("Cannot clear the active FFT workspace.\n");
	columnFFTScatter_gpu(size_t(count)*C.colLength(),C.basis->nbasis,C.basis->gInfo->nr,
		C.basis->index.dataGpu(),C.dataGpu()+size_t(start)*C.colLength(),workspace.values.dataGpu());
	if(cufftExecZ2Z(plan,(cufftDoubleComplex*)workspace.values.dataGpu(),
		(cufftDoubleComplex*)workspace.values.dataGpu(),CUFFT_INVERSE)!=CUFFT_SUCCESS) die("Batched inverse FFT failed.\n");
}
}
bool useBatchedColumnFFT() { return batchLimit()>0; }
namespace {
template<typename Descriptor> void uploadDescriptors(const std::vector<Descriptor>& host,ManagedArray<Descriptor>& device)
{
	if(device.nData()<host.size()) device.init(host.size(),true);
	if(host.size() && cudaMemcpy(device.dataGpu(),host.data(),host.size()*sizeof(Descriptor),cudaMemcpyHostToDevice)!=cudaSuccess)
		die("Cannot copy batched wavefunction descriptors.\n");
}
}
double batchedColumnDot(const std::vector<ColumnBundle>& x,const std::vector<ColumnBundle>& y,
	int start,int stop,int nStates,bool commensurate)
{
	// Reuse storage; vector addresses may change between Lanczos iterations.
	static ManagedArray<ColumnDotDescriptor> pairs;
	static ManagedArray<double> partial,result;
	std::vector<ColumnDotDescriptor> descriptors;
	descriptors.reserve(size_t(stop-start)*(commensurate ? 1 : 2));
	for(int k=start;k<stop;k++) for(int sign=0;sign<(commensurate ? 1 : 2);sign++) {
		int index=k+sign*nStates;
		if(x[index] && y[index]) {
			assert(x[index].nData()==y[index].nData());
			descriptors.push_back({x[index].dataGpu(),y[index].dataGpu(),x[index].nData()});
		}
	}
	if(descriptors.empty()) return 0.;
	if(pairs.nData()<descriptors.size()) {
		pairs.init(descriptors.size(),true); partial.init(descriptors.size(),true);
	}
	if(!result.nData()) result.init(1,true);
	if(cudaMemcpy(pairs.dataGpu(),descriptors.data(),descriptors.size()*sizeof(ColumnDotDescriptor),cudaMemcpyHostToDevice)!=cudaSuccess)
		die("Cannot copy response dot-product descriptors.\n");
	columnDot_gpu(descriptors.size(),pairs.dataGpu(),partial.dataGpu(),result.dataGpu());
	double value;
	if(cudaMemcpy(&value,result.dataGpu(),sizeof(value),cudaMemcpyDeviceToHost)!=cudaSuccess)
		die("Cannot retrieve response dot product.\n");
	return value;
}
ColumnBundle batchedLocalPotential(const ColumnBundle& C,const double* realPotential,const complex* complexPotential,
	const ColumnBundle& Cout,const ManagedArray<complex>* cached)
{
	assert(!C.isSpinor() && !Cout.isSpinor());
	assert(C.nCols()==Cout.nCols() && C.basis->gInfo==Cout.basis->gInfo);
	ColumnBundle result=Cout.similar();
	const GridInfo& grid=*C.basis->gInfo; auto& workspace=fftWorkspace();
	int limit=std::max(1,batchLimit());
	for(int start=0;start<C.nCols();start+=limit) {
		int count=std::min(limit,C.nCols()-start); auto plan=workspace.prepare(grid,count);
		if(cached) {
			if(cudaMemcpy(workspace.values.dataGpu(),cached->dataGpu()+size_t(start)*grid.nr,size_t(count)*grid.nr*sizeof(complex),cudaMemcpyDeviceToDevice)!=cudaSuccess)
				die("Cannot copy cached wavefunctions to the FFT workspace.\n");
		} else inverseColumns(C,start,count,workspace,plan);
		columnFFTMultiply_gpu(size_t(count)*grid.nr,grid.nr,realPotential,complexPotential,workspace.values.dataGpu());
		if(cufftExecZ2Z(plan,(cufftDoubleComplex*)workspace.values.dataGpu(),
			(cufftDoubleComplex*)workspace.values.dataGpu(),CUFFT_FORWARD)!=CUFFT_SUCCESS) die("Batched forward FFT failed.\n");
		columnFFTGather_gpu(size_t(count)*Cout.colLength(),Cout.basis->nbasis,grid.nr,Cout.basis->index.dataGpu(),
			workspace.values.dataGpu(),result.dataGpu()+size_t(start)*Cout.colLength());
	}
	return result;
}
void batchedLocalPotentials(const std::vector<const ColumnBundle*>& input,
	const std::vector<ColumnBundle*>& output,const double* realPotential,const complex* complexPotential)
{
	assert(input.size()==output.size());
	if(input.empty()) return;
	const GridInfo& grid=*input.front()->basis->gInfo;
	std::vector<ColumnFFTDescriptor> all;
	int maximumInput=0,maximumOutput=0;
	for(size_t b=0;b<input.size();b++) {
		const ColumnBundle& C=*input[b]; ColumnBundle& result=*output[b];
		assert(!C.isSpinor() && !result.isSpinor() && C.nCols()==result.nCols());
		assert(C.basis->gInfo==&grid && result.basis->gInfo==&grid);
		maximumInput=std::max(maximumInput,int(C.basis->nbasis));
		maximumOutput=std::max(maximumOutput,int(result.basis->nbasis));
		for(int col=0;col<C.nCols();col++) all.push_back({C.dataGpu()+size_t(col)*C.colLength(),
			result.dataGpu()+size_t(col)*result.colLength(),C.basis->index.dataGpu(),result.basis->index.dataGpu(),
			C.basis->nbasis,result.basis->nbasis});
	}
	auto& workspace=fftWorkspace(); int limit=std::max(1,batchLimit());
	static ManagedArray<ColumnFFTDescriptor> descriptors;
	std::vector<ColumnFFTDescriptor> batch; batch.reserve(limit);
	for(size_t start=0;start<all.size();start+=limit) {
		int count=std::min(size_t(limit),all.size()-start); auto plan=workspace.prepare(grid,count);
		batch.assign(all.begin()+start,all.begin()+start+count); uploadDescriptors(batch,descriptors);
		if(cudaMemset(workspace.values.dataGpu(),0,size_t(count)*grid.nr*sizeof(complex))!=cudaSuccess)
			die("Cannot clear the active FFT workspace.\n");
		columnFFTScatterBatch_gpu(count,maximumInput,grid.nr,descriptors.dataGpu(),workspace.values.dataGpu());
		if(cufftExecZ2Z(plan,(cufftDoubleComplex*)workspace.values.dataGpu(),
			(cufftDoubleComplex*)workspace.values.dataGpu(),CUFFT_INVERSE)!=CUFFT_SUCCESS) die("Batched inverse FFT failed.\n");
		columnFFTMultiply_gpu(size_t(count)*grid.nr,grid.nr,realPotential,complexPotential,workspace.values.dataGpu());
		if(cufftExecZ2Z(plan,(cufftDoubleComplex*)workspace.values.dataGpu(),
			(cufftDoubleComplex*)workspace.values.dataGpu(),CUFFT_FORWARD)!=CUFFT_SUCCESS) die("Batched forward FFT failed.\n");
		columnFFTGatherBatch_gpu(count,maximumOutput,grid.nr,descriptors.dataGpu(),workspace.values.dataGpu());
	}
}
void updateColumnBundleGPU(ColumnBundle& output,const ColumnBundle* input,const diagMatrix& scales,
	double outputScale,double inputScale,bool scaleOutput)
{
	assert(scales.size()==size_t(output.nCols()));
	static ManagedArray<ColumnUpdateDescriptor> descriptors;
	std::vector<ColumnUpdateDescriptor> host; host.reserve(output.nCols());
	for(int col=0;col<output.nCols();col++) host.push_back({output.dataGpu()+size_t(col)*output.colLength(),
		input ? input->dataGpu()+size_t(col)*input->colLength() : nullptr,output.colLength(),
		outputScale*(scaleOutput ? scales[col] : 1.),inputScale*(scaleOutput ? 1. : scales[col])});
	uploadDescriptors(host,descriptors);
	columnUpdate_gpu(host.size(),output.colLength(),descriptors.dataGpu());
}
ManagedArray<complex> batchedRealSpaceColumns(const ColumnBundle& C)
{
	assert(!C.isSpinor());
	const GridInfo& grid=*C.basis->gInfo; auto& workspace=fftWorkspace();
	ManagedArray<complex> result; result.init(size_t(C.nCols())*grid.nr,true);
	int limit=std::max(1,batchLimit());
	for(int start=0;start<C.nCols();start+=limit) {
		int count=std::min(limit,C.nCols()-start); auto plan=workspace.prepare(grid,count);
		inverseColumns(C,start,count,workspace,plan);
		if(cudaMemcpy(result.dataGpu()+size_t(start)*grid.nr,workspace.values.dataGpu(),size_t(count)*grid.nr*sizeof(complex),cudaMemcpyDeviceToDevice)!=cudaSuccess)
			die("Cannot cache real-space wavefunctions.\n");
	}
	gpuErrorCheck(); return result;
}
#endif

std::vector<diagMatrix> batchedDiagDot(const std::vector<const ColumnBundle*>& x,
	const std::vector<const ColumnBundle*>& y)
{
	assert(x.size()==y.size());
	std::vector<diagMatrix> result; result.reserve(x.size());
	#ifdef GPU_ENABLED
	static ManagedArray<ColumnDotDescriptor> pairs;
	static ManagedArray<double> values;
	std::vector<ColumnDotDescriptor> host;
	for(size_t b=0;b<x.size();b++) {
		assert(x[b]->basis==y[b]->basis && x[b]->nCols()==y[b]->nCols());
		result.emplace_back(x[b]->nCols());
		for(int col=0;col<x[b]->nCols();col++) host.push_back({x[b]->dataGpu()+size_t(col)*x[b]->colLength(),
			y[b]->dataGpu()+size_t(col)*y[b]->colLength(),x[b]->colLength()});
	}
	if(host.empty()) return result;
	uploadDescriptors(host,pairs);
	if(values.nData()<host.size()) values.init(host.size(),true);
	columnDiagDot_gpu(host.size(),pairs.dataGpu(),values.dataGpu());
	std::vector<double> scalars(host.size());
	if(cudaMemcpy(scalars.data(),values.dataGpu(),scalars.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)
		die("Cannot retrieve batched column dot products.\n");
	size_t offset=0;
	for(auto& block:result) for(double& value:block) value=scalars[offset++];
	#else
	for(size_t b=0;b<x.size();b++) result.push_back(diagDot(*x[b],*y[b]));
	#endif
	return result;
}
void batchedCGStep(const std::vector<ColumnBundle*>& solution,const std::vector<ColumnBundle*>& residual,
	const std::vector<const ColumnBundle*>& direction,const std::vector<const ColumnBundle*>& applied,
	const std::vector<diagMatrix>& alpha)
{
	assert(solution.size()==residual.size() && solution.size()==direction.size()
		&& solution.size()==applied.size() && solution.size()==alpha.size());
	#ifdef GPU_ENABLED
	static ManagedArray<ColumnCGStepDescriptor> descriptors;
	std::vector<ColumnCGStepDescriptor> host; size_t maximum=0;
	for(size_t b=0;b<solution.size();b++) {
		size_t length=solution[b]->colLength(); maximum=std::max(maximum,length);
		assert(length==residual[b]->colLength() && length==direction[b]->colLength() && length==applied[b]->colLength());
		assert(alpha[b].size()==size_t(solution[b]->nCols()));
		for(int col=0;col<solution[b]->nCols();col++) host.push_back({solution[b]->dataGpu()+size_t(col)*length,
			residual[b]->dataGpu()+size_t(col)*length,direction[b]->dataGpu()+size_t(col)*length,
			applied[b]->dataGpu()+size_t(col)*length,length,alpha[b][col]});
	}
	uploadDescriptors(host,descriptors); columnCGStep_gpu(host.size(),maximum,descriptors.dataGpu());
	#else
	for(size_t b=0;b<solution.size();b++) {
		solution[b]->axpyColumns(1.,*direction[b],alpha[b]); residual[b]->axpyColumns(-1.,*applied[b],alpha[b]);
	}
	#endif
}
void batchedCGDirection(const std::vector<ColumnBundle*>& direction,
	const std::vector<const ColumnBundle*>& preconditioned,const std::vector<diagMatrix>& beta)
{
	assert(direction.size()==preconditioned.size() && direction.size()==beta.size());
	#ifdef GPU_ENABLED
	static ManagedArray<ColumnUpdateDescriptor> descriptors;
	std::vector<ColumnUpdateDescriptor> host; size_t maximum=0;
	for(size_t b=0;b<direction.size();b++) {
		size_t length=direction[b]->colLength(); maximum=std::max(maximum,length);
		assert(length==preconditioned[b]->colLength() && beta[b].size()==size_t(direction[b]->nCols()));
		for(int col=0;col<direction[b]->nCols();col++) host.push_back({direction[b]->dataGpu()+size_t(col)*length,
			preconditioned[b]->dataGpu()+size_t(col)*length,length,beta[b][col],1.});
	}
	uploadDescriptors(host,descriptors); columnUpdate_gpu(host.size(),maximum,descriptors.dataGpu());
	#else
	for(size_t b=0;b<direction.size();b++) {
		direction[b]->scaleColumns(beta[b]); *direction[b]+=*preconditioned[b];
	}
	#endif
}

void projectOutBatched(const std::vector<const ColumnBundle*>& projectors,
	const std::vector<ColumnBundle*>& values)
{
	assert(projectors.size()==values.size());
	#ifdef GPU_ENABLED
	static bool enabled=[]() {
		const char* setting=getenv("JDFTX_KI_BATCHED_PROJECTORS");
		if(setting && string(setting)!="0" && string(setting)!="1")
			die("JDFTX_KI_BATCHED_PROJECTORS must be 0 or 1.\n");
		return !setting || string(setting)!="0";
	}();
	if(enabled && !values.empty()) {
		using Shape=std::tuple<int,int,int,double>;
		std::map<Shape,std::vector<size_t>> groups;
		size_t count=0;
		for(size_t b=0;b<values.size();b++) {
			assert(!projectors[b]->isSpinor() && !values[b]->isSpinor());
			assert(projectors[b]->colLength()==values[b]->colLength());
			if(!*values[b] || !*projectors[b]) continue;
			double metric=values[b]->basis ? values[b]->basis->gInfo->detR : 1.;
			groups[{int(values[b]->colLength()),projectors[b]->nCols(),values[b]->nCols(),metric}].push_back(b);
			count+=size_t(projectors[b]->nCols())*values[b]->nCols();
		}
		static ManagedArray<complex> coefficients;
		static ManagedArray<complex*> bra,ket,small;
		if(coefficients.nData()<count) coefficients.init(count,true);
		std::vector<complex*> hostBra,hostKet,hostSmall;
		size_t offset=0;
		for(const auto& group:groups) for(size_t b:group.second) {
			hostBra.push_back(const_cast<complex*>(projectors[b]->dataGpu()));
			hostKet.push_back(values[b]->dataGpu());
			hostSmall.push_back(coefficients.dataGpu()+offset);
			offset+=size_t(projectors[b]->nCols())*values[b]->nCols();
		}
		uploadDescriptors(hostBra,bra); uploadDescriptors(hostKet,ket); uploadDescriptors(hostSmall,small);
		const cuDoubleComplex one={1.,0.},zero={0.,0.},minusOne={-1.,0.};
		size_t start=0;
		for(const auto& group:groups) {
			int length,occupied,bands; double metric;
			std::tie(length,occupied,bands,metric)=group.first;
			const cuDoubleComplex overlapScale={metric,0.};
			auto C=reinterpret_cast<const cuDoubleComplex* const*>(bra.dataGpu()+start);
			auto X=reinterpret_cast<const cuDoubleComplex* const*>(ket.dataGpu()+start);
			auto S=reinterpret_cast<cuDoubleComplex**>(small.dataGpu()+start);
			if(cublasZgemmBatched(cublasHandle,CUBLAS_OP_C,CUBLAS_OP_N,occupied,bands,length,
				&overlapScale,C,length,X,length,&zero,S,occupied,group.second.size())!=CUBLAS_STATUS_SUCCESS)
				die("Batched occupied overlap failed.\n");
			auto overlap=reinterpret_cast<const cuDoubleComplex* const*>(small.dataGpu()+start);
			auto output=reinterpret_cast<cuDoubleComplex**>(ket.dataGpu()+start);
			if(cublasZgemmBatched(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N,length,bands,occupied,
				&minusOne,C,length,overlap,occupied,&one,output,length,group.second.size())!=CUBLAS_STATUS_SUCCESS)
				die("Batched occupied projection failed.\n");
			start+=group.second.size();
		}
		return;
	}
	#endif
	for(size_t b=0;b<values.size();b++) if(*values[b] && *projectors[b]) {
		double metric=values[b]->basis ? values[b]->basis->gInfo->detR : 1.;
		*values[b]-=*projectors[b]*(metric*(*projectors[b]^*values[b]));
	}
}

complexScalarFieldArray diagouterI_cached(const ColumnBundle& response,
	const ManagedArray<complex>& ground,const ManagedArray<double>& fillings,bool groundFirst,
	int nDensities,const GridInfo* gInfoOut)
{
	assert(!response.isSpinor() && (nDensities==1 || nDensities==2));
	const GridInfo& grid=*response.basis->gInfo;
	assert(ground.nData()==size_t(response.nCols())*grid.nr && fillings.nData()==size_t(response.nCols()));
	complexScalarField density=complexScalarFieldData::alloc(grid); initZero(density);
	#ifdef GPU_ENABLED
	auto& workspace=fftWorkspace(); int limit=std::max(1,batchLimit());
	for(int start=0;start<response.nCols();start+=limit) {
		int count=std::min(limit,response.nCols()-start); auto plan=workspace.prepare(grid,count);
		inverseColumns(response,start,count,workspace,plan);
		columnPairDensity_gpu(grid.nr,count,ground.dataGpu()+size_t(start)*grid.nr,
			workspace.values.dataGpu(),fillings.dataGpu()+start,density->dataGpu(),groundFirst);
	}
	#else
	for(int col=0;col<response.nCols();col++) {
		auto values=I(response.getColumn(col,0));
		const complex* fixed=ground.data()+size_t(col)*grid.nr;
		eblas_accumProdComplex(grid.nr,fillings.data()[col],groundFirst ? fixed : values->data(),
			groundFirst ? values->data() : fixed,density->data());
	}
	#endif
	if(gInfoOut && &grid!=gInfoOut) density=changeGrid(density,*gInfoOut);
	complexScalarFieldArray result(nDensities);
	result[nDensities==2 ? response.qnum->index() : 0]=density;
	return result;
}

ManagedArray<complex> realSpaceColumns(const ColumnBundle& C)
{
	assert(!C.isSpinor());
	#ifdef GPU_ENABLED
	return batchedRealSpaceColumns(C);
	#else
	const GridInfo& grid=*C.basis->gInfo;
	ManagedArray<complex> result; result.init(size_t(C.nCols())*grid.nr);
	for(int col=0;col<C.nCols();col++) {
		auto values=I(C.getColumn(col,0));
		callPref(eblas_copy)(result.dataPref()+size_t(col)*grid.nr,values->dataPref(),grid.nr);
	}
	return result;
	#endif
}

ColumnBundle Idag_DiagV_I_cached(const ColumnBundle& C,const complexScalarFieldArray& V,
	const ManagedArray<complex>& cached,const ColumnBundle& Cout)
{
	assert(!C.isSpinor() && V.size()==1 && cached.nData()==size_t(C.nCols())*C.basis->gInfo->nr);
	const GridInfo& grid=*C.basis->gInfo;
	complexScalarField potential=V[0];
	if(&potential->gInfo!=&grid) potential=Jdag(changeGrid(Idag(potential),grid),true);
	#ifdef GPU_ENABLED
	return batchedLocalPotential(C,nullptr,potential->dataGpu(),Cout,&cached);
	#else
	ColumnBundle result=Cout.similar(); result.zero();
	auto values=complexScalarFieldData::alloc(grid);
	for(int col=0;col<C.nCols();col++) {
		callPref(eblas_copy)(values->dataPref(),cached.dataPref()+size_t(col)*grid.nr,grid.nr);
		result.accumColumn(col,0,Idag(potential*values));
	}
	return result;
	#endif
}
