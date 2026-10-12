/* Exact uniform-k-mesh correlations. GPL v3 or later; part of JDFTx. */
#include <electronic/KMeshCorrelation.h>
#include <core/Operators.h>
#include <algorithm>
#include <cstdlib>

namespace
{
int meshSlot(const vector3<>& k,const vector3<int>& folding)
{
	vector3<int> index;
	for(int d=0;d<3;d++) {
		double value=k[d]*folding[d];
		if(fabs(value-round(value))>1e-8) return -1;
		index[d]=int(round(value))%folding[d];
		if(index[d]<0) index[d]+=folding[d];
	}
	return (index[0]*folding[1]+index[1])*folding[2]+index[2];
}
complex phase(const GridInfo& grid,int r,const vector3<>& k)
{
	int z=r%grid.S[2],xy=r/grid.S[2],y=xy%grid.S[1],x=xy/grid.S[1];
	return cis((2*M_PI)*(k[0]*x/grid.S[0]+k[1]*y/grid.S[1]+k[2]*z/grid.S[2]));
}
}

struct KMeshCorrelation::Impl
{
	const GridInfo& grid;
	int nk,centers,tile;
	ManagedArray<int> slotToK,kToSlot,slotToQ,qToSlot;
	ManagedArray<vector3<>> mesh,qmesh;
	ManagedArray<const complex*> orbital;
	std::vector<ManagedArray<complex>> sources;
	ManagedArray<complex> a,b;
	#ifdef GPU_ENABLED
	cufftHandle plan=0;
	ManagedArray<char> work;
	#else
	fftw_plan forward=nullptr,backward=nullptr;
	#endif

	Impl(const GridInfo& grid,int nk,int centers,int tile):grid(grid),nk(nk),centers(centers),tile(tile) {}
	~Impl() {
		#ifdef GPU_ENABLED
		if(plan) cufftDestroy(plan);
		#else
		if(forward) fftw_destroy_plan(forward);
		if(backward) fftw_destroy_plan(backward);
		#endif
	}
	void fft(ManagedArray<complex>& values,bool inverse) {
		#ifdef GPU_ENABLED
		if(cufftExecZ2Z(plan,(cufftDoubleComplex*)values.dataGpu(),(cufftDoubleComplex*)values.dataGpu(),
			inverse ? CUFFT_INVERSE : CUFFT_FORWARD)!=CUFFT_SUCCESS) die("KI k-mesh FFT failed.\n");
		#else
		fftw_execute_dft(inverse ? backward : forward,(fftw_complex*)values.data(),(fftw_complex*)values.data());
		#endif
	}
	void packOrbital(int n,int start,int count) {
		#ifdef GPU_ENABLED
		kMeshPackOrbital_gpu(nk,centers,n,start,count,tile,grid.S,slotToK.dataGpu(),mesh.dataGpu(),orbital.dataGpu(),a.dataGpu());
		#else
		complex* values=a.data(); const int* indices=slotToK.data();
		const auto* points=mesh.data(); const auto* fields=orbital.data();
		for(int r=0;r<tile;r++) for(int slot=0;slot<nk;slot++) {
			int ik=indices[slot];
			values[size_t(r)*nk+slot]=r<count ? fields[ik*centers+n][start+r]*phase(grid,start+r,points[ik]) : complex();
		}
		#endif
	}
	void buildSource(int n) {
		sources[n].init(size_t(nk)*grid.nr,isGpuEnabled());
		for(int start=0;start<grid.nr;start+=tile) {
			int count=std::min(tile,grid.nr-start);
			packOrbital(n,start,count); fft(a,false);
			#ifdef GPU_ENABLED
			kMeshPower_gpu(a.nData(),a.dataGpu());
			#else
			for(size_t i=0;i<a.nData();i++) a.data()[i]=complex(a.data()[i].norm());
			#endif
			fft(a,true);
			#ifdef GPU_ENABLED
			kMeshStoreSource_gpu(nk,grid.nr,start,count,grid.S,qToSlot.dataGpu(),qmesh.dataGpu(),a.dataGpu(),sources[n].dataGpu());
			#else
			for(int iq=0;iq<nk;iq++) for(int r=0;r<count;r++)
				sources[n].data()[size_t(iq)*grid.nr+start+r]=a.data()[size_t(r)*nk+qToSlot.data()[iq]]
					*phase(grid,start+r,-qmesh.data()[iq])/(double(nk)*nk);
			#endif
		}
	}
};

KMeshCorrelation::KMeshCorrelation(const GridInfo& grid,const vector3<int>& folding,
	const std::vector<vector3<>>& mesh,const std::vector<vector3<>>& qmesh,
	const std::vector<std::vector<complexScalarField>>& orbital,double cacheMiB)
{
	const char* enabled=getenv("JDFTX_KI_FFT_CORRELATION");
	if(enabled && string(enabled)=="0") return;
	if(const char* setting=getenv("JDFTX_KI_CORRELATION_CACHE_MIB")) {
		char* end=nullptr; cacheMiB=strtod(setting,&end);
		if(*end || !std::isfinite(cacheMiB) || cacheMiB<0.) die("JDFTX_KI_CORRELATION_CACHE_MIB must be finite and nonnegative.\n");
	}
	const int nk=mesh.size(),centers=orbital.empty() ? 0 : orbital[0].size();
	if(!nk || !centers || qmesh.size()!=mesh.size() || orbital.size()!=mesh.size()
		|| folding[0]<=0 || folding[1]<=0 || folding[2]<=0 || nk!=folding[0]*folding[1]*folding[2]) return;
	std::vector<int> slotToK(nk,-1),kToSlot(nk),slotToQ(nk,-1),qToSlot(nk);
	for(int ik=0;ik<nk;ik++) {
		int slot=meshSlot(mesh[ik]-mesh[0],folding);
		if(slot<0 || slotToK[slot]>=0 || int(orbital[ik].size())!=centers) return;
		slotToK[slot]=ik; kToSlot[ik]=slot;
		slot=meshSlot(qmesh[ik],folding);
		if(slot<0 || slotToQ[slot]>=0) return;
		slotToQ[slot]=ik; qToSlot[ik]=slot;
	}
	// Source cache plus one orbital's owned-q potentials. The two FFT tile
	// buffers and explicit cuFFT workspace share the same overall budget.
	const double budget=cacheMiB*1024.*1024.;
	const double fields=double(centers+1)*nk*grid.nr*sizeof(complex);
	int tile=int(std::max(0.,std::min(1024.,(budget-fields)/(3.*nk*sizeof(complex)))));
	if(tile<1) {
		logPrintf("KI k-mesh correlations: direct path (%.1f MiB budget is insufficient).\n",cacheMiB); return;
	}
	auto candidate=std::unique_ptr<Impl>(new Impl(grid,nk,centers,tile));
	candidate->a.init(size_t(tile)*nk,isGpuEnabled()); candidate->b.init(size_t(tile)*nk,isGpuEnabled());
	int sizes[3]={folding[0],folding[1],folding[2]}; size_t workBytes=0;
	#ifdef GPU_ENABLED
	if(cufftCreate(&candidate->plan)!=CUFFT_SUCCESS || cufftSetAutoAllocation(candidate->plan,0)!=CUFFT_SUCCESS
		|| cufftMakePlanMany(candidate->plan,3,sizes,nullptr,1,nk,nullptr,1,nk,CUFFT_Z2Z,tile,&workBytes)!=CUFFT_SUCCESS)
		die("Cannot create KI k-mesh correlation FFT plan.\n");
	if(fields+2.*tile*nk*sizeof(complex)+workBytes>budget) {
		logPrintf("KI k-mesh correlations: direct path (FFT workspace exceeds %.1f MiB budget).\n",cacheMiB); return;
	}
	if(workBytes) candidate->work.init(workBytes,true);
	if(cufftSetWorkArea(candidate->plan,workBytes ? candidate->work.dataGpu() : nullptr)!=CUFFT_SUCCESS)
		die("Cannot assign KI k-mesh FFT workspace.\n");
	#else
	candidate->forward=fftw_plan_many_dft(3,sizes,tile,(fftw_complex*)candidate->a.data(),nullptr,1,nk,
		(fftw_complex*)candidate->a.data(),nullptr,1,nk,FFTW_FORWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
	candidate->backward=fftw_plan_many_dft(3,sizes,tile,(fftw_complex*)candidate->a.data(),nullptr,1,nk,
		(fftw_complex*)candidate->a.data(),nullptr,1,nk,FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
	if(!candidate->forward || !candidate->backward) die("Cannot create KI k-mesh correlation FFT plan.\n");
	#endif
	candidate->slotToK=ManagedArray<int>(slotToK); candidate->kToSlot=ManagedArray<int>(kToSlot);
	candidate->slotToQ=ManagedArray<int>(slotToQ); candidate->qToSlot=ManagedArray<int>(qToSlot);
	candidate->mesh=ManagedArray<vector3<>>(mesh); candidate->qmesh=ManagedArray<vector3<>>(qmesh);
	std::vector<const complex*> fieldsPointers;
	for(const auto& point:orbital) for(const auto& field:point) fieldsPointers.push_back(field->dataPref());
	candidate->orbital=ManagedArray<const complex*>(fieldsPointers);
	candidate->sources.resize(centers);
	for(int n=0;n<centers;n++) candidate->buildSource(n);
	logPrintf("KI k-mesh correlations: FFT sources and conduction convolution, %d grid points/tile, %.1f MiB peak (budget %.1f MiB/rank).\n",
		tile,(fields+2.*tile*nk*sizeof(complex)+workBytes)/(1024.*1024.),cacheMiB);
	impl=std::move(candidate);
}

KMeshCorrelation::~KMeshCorrelation() = default;

complexScalarField KMeshCorrelation::source(int n,int iq) const
{
	assert(impl && n>=0 && n<impl->centers && iq>=0 && iq<impl->nk);
	complexScalarField rho=complexScalarFieldData::alloc(impl->grid);
	callPref(eblas_copy)(rho->dataPref(),impl->sources[n].dataPref()+size_t(iq)*impl->grid.nr,impl->grid.nr);
	return rho;
}

void KMeshCorrelation::accumulateConduction(int n,const std::vector<complexScalarField>& potential,
	const std::vector<int>& emptyCenters,std::vector<matrix>& correction)
{
	assert(impl && int(potential.size())==impl->nk && int(correction.size())==impl->nk);
	if(emptyCenters.empty()) return;
	auto& p=*impl;
	std::vector<const complex*> potentialPointers;
	for(const auto& field:potential) potentialPointers.push_back(field ? field->dataPref() : nullptr);
	ManagedArray<const complex*> potentials(potentialPointers);
	#ifdef GPU_ENABLED
	std::vector<complex*> correctionPointers;
	for(auto& item:correction) correctionPointers.push_back(item.dataGpu());
	ManagedArray<complex*> corrections(correctionPointers); ManagedArray<int> rows(emptyCenters);
	#else
	const complex* const* v=potentials.data();
	#endif
	for(int start=0;start<p.grid.nr;start+=p.tile) {
		int count=std::min(p.tile,p.grid.nr-start);
		p.packOrbital(n,start,count); p.fft(p.a,false);
		#ifdef GPU_ENABLED
		kMeshPackPotential_gpu(p.nk,start,count,p.tile,p.grid.S,p.slotToQ.dataGpu(),p.qmesh.dataGpu(),potentials.dataGpu(),p.b.dataGpu());
		#else
		for(int r=0;r<p.tile;r++) for(int slot=0;slot<p.nk;slot++) {
			int iq=p.slotToQ.data()[slot];
			p.b.data()[size_t(r)*p.nk+slot]=(r<count && v[iq]) ? v[iq][start+r].conj()*phase(p.grid,start+r,-p.qmesh.data()[iq]) : complex();
		}
		#endif
		// W(k) = sum_q conj(V(q))*exp(-iq.r)*psi(k+q)/Nk:
		// backward( backward(A) * forward(psi) ) / Nk^2.
		p.fft(p.b,true);
		#ifdef GPU_ENABLED
		kMeshMultiply_gpu(p.a.nData(),p.a.dataGpu(),p.b.dataGpu());
		#else
		for(size_t i=0;i<p.a.nData();i++) p.b.data()[i]*=p.a.data()[i];
		#endif
		p.fft(p.b,true);
		const double scale=p.grid.dV/(double(p.nk)*p.nk);
		#ifdef GPU_ENABLED
		kMeshConduction_gpu(p.nk,p.centers,n,start,count,p.grid.S,p.kToSlot.dataGpu(),p.mesh.dataGpu(),p.orbital.dataGpu(),
			emptyCenters.size(),rows.dataGpu(),p.b.dataGpu(),corrections.dataGpu(),scale);
		#else
		for(int ik=0;ik<p.nk;ik++) for(int m:emptyCenters) {
			complex value;
			for(int r=0;r<count;r++) value+=p.orbital.data()[ik*p.centers+m][start+r].conj()
				*phase(p.grid,start+r,-p.mesh.data()[ik])*p.b.data()[size_t(r)*p.nk+p.kToSlot.data()[ik]];
			correction[ik].set(m,n,correction[ik](m,n)+scale*value);
		}
		#endif
	}
}
