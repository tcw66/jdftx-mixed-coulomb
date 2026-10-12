/* Batched scalar-wavefunction FFT kernels. GPL v3 or later; part of JDFTx. */
#include <electronic/ColumnBundleFFT.h>
#include <core/GpuKernelUtils.h>

__global__ void columnDotPartial_kernel(const ColumnDotDescriptor* pairs,double* partial)
{
	const auto pair=pairs[blockIdx.x];
	double value=0.;
	for(size_t i=threadIdx.x;i<pair.count;i+=blockDim.x)
		value+=pair.x[i].real()*pair.y[i].real()+pair.x[i].imag()*pair.y[i].imag();
	__shared__ double sums[256]; sums[threadIdx.x]=value; __syncthreads();
	for(int stride=128;stride;stride/=2) {
		if(threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
		__syncthreads();
	}
	if(!threadIdx.x) partial[blockIdx.x]=sums[0];
}
__global__ void columnDotSum_kernel(int count,const double* partial,double* result)
{
	double value=0.; for(int i=threadIdx.x;i<count;i+=blockDim.x) value+=partial[i];
	__shared__ double sums[256]; sums[threadIdx.x]=value; __syncthreads();
	for(int stride=128;stride;stride/=2) {
		if(threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
		__syncthreads();
	}
	if(!threadIdx.x) *result=2.*sums[0];
}
void columnDot_gpu(int count,const ColumnDotDescriptor* pairs,double* partial,double* result)
{
	columnDotPartial_kernel<<<count,256>>>(pairs,partial);
	columnDotSum_kernel<<<1,256>>>(count,partial,result); gpuErrorCheck();
}
void columnDiagDot_gpu(int count,const ColumnDotDescriptor* pairs,double* result)
{
	if(count) columnDotPartial_kernel<<<count,256>>>(pairs,result);
	gpuErrorCheck();
}

__global__ void columnCGStep_kernel(const ColumnCGStepDescriptor* descriptors)
{
	const auto descriptor=descriptors[blockIdx.y];
	size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
	if(i<descriptor.count) {
		descriptor.solution[i]+=descriptor.alpha*descriptor.direction[i];
		descriptor.residual[i]-=descriptor.alpha*descriptor.applied[i];
	}
}
void columnCGStep_gpu(int count,size_t maximum,const ColumnCGStepDescriptor* descriptors)
{
	if(count && maximum) columnCGStep_kernel<<<dim3((maximum+255)/256,count),256>>>(descriptors);
	gpuErrorCheck();
}
__global__ void columnUpdate_kernel(const ColumnUpdateDescriptor* descriptors)
{
	const auto descriptor=descriptors[blockIdx.y];
	size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
	if(i<descriptor.count) descriptor.output[i]=descriptor.outputScale*descriptor.output[i]
		+(descriptor.input ? descriptor.inputScale*descriptor.input[i] : complex());
}
void columnUpdate_gpu(int count,size_t maximum,const ColumnUpdateDescriptor* descriptors)
{
	if(count && maximum) columnUpdate_kernel<<<dim3((maximum+255)/256,count),256>>>(descriptors);
	gpuErrorCheck();
}
__global__ void columnFFTScatterBatch_kernel(int nr,const ColumnFFTDescriptor* descriptors,complex* values)
{
	const auto descriptor=descriptors[blockIdx.y];
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<descriptor.inputBasis) values[size_t(blockIdx.y)*nr+descriptor.inputIndices[i]]=descriptor.input[i];
}
void columnFFTScatterBatch_gpu(int count,int maximum,int nr,const ColumnFFTDescriptor* descriptors,complex* values)
{
	columnFFTScatterBatch_kernel<<<dim3((maximum+255)/256,count),256>>>(nr,descriptors,values); gpuErrorCheck();
}
__global__ void columnFFTGatherBatch_kernel(int nr,const ColumnFFTDescriptor* descriptors,const complex* values)
{
	const auto descriptor=descriptors[blockIdx.y];
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<descriptor.outputBasis) descriptor.output[i]=values[size_t(blockIdx.y)*nr+descriptor.outputIndices[i]];
}
void columnFFTGatherBatch_gpu(int count,int maximum,int nr,const ColumnFFTDescriptor* descriptors,const complex* values)
{
	columnFFTGatherBatch_kernel<<<dim3((maximum+255)/256,count),256>>>(nr,descriptors,values); gpuErrorCheck();
}

__global__ void columnPairDensity_kernel(int nr,int columns,const complex* ground,const complex* response,
	const double* fillings,complex* output,bool groundFirst)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i>=nr) return;
	complex value;
	for(int b=0;b<columns;b++) {
		size_t j=size_t(b)*nr+i;
		value+=fillings[b]*(groundFirst ? ground[j]*response[j].conj() : response[j]*ground[j].conj());
	}
	output[i]+=value;
}
void columnPairDensity_gpu(int nr,int columns,const complex* ground,const complex* response,
	const double* fillings,complex* output,bool groundFirst)
{
	columnPairDensity_kernel<<<(nr+255)/256,256>>>(nr,columns,ground,response,fillings,output,groundFirst); gpuErrorCheck();
}

__global__ void matrixColumnAccumulate_kernel(int count,const int* rows,int nr,int column,
	const complex* values,complex* output)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<count) output[rows[i]+nr*column]+=values[i];
}
void matrixColumnAccumulate_gpu(int count,const int* rows,int nr,int column,const complex* values,complex* output)
{
	matrixColumnAccumulate_kernel<<<(count+255)/256,256>>>(count,rows,nr,column,values,output); gpuErrorCheck();
}

__global__ void columnFFTScatter_kernel(size_t count,int nbasis,int nr,const int* indices,const complex* input,complex* output)
{
	size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
	if(i<count) output[(i/nbasis)*nr+indices[i%nbasis]]=input[i];
}
void columnFFTScatter_gpu(size_t count,int nbasis,int nr,const int* indices,const complex* input,complex* output)
{
	columnFFTScatter_kernel<<<(count+255)/256,256>>>(count,nbasis,nr,indices,input,output); gpuErrorCheck();
}
__global__ void columnFFTMultiply_kernel(size_t count,int nr,const double* realPotential,const complex* complexPotential,complex* values)
{
	size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
	if(i<count) values[i]*=complexPotential ? complexPotential[i%nr] : complex(realPotential[i%nr]);
}
void columnFFTMultiply_gpu(size_t count,int nr,const double* realPotential,const complex* complexPotential,complex* values)
{
	columnFFTMultiply_kernel<<<(count+255)/256,256>>>(count,nr,realPotential,complexPotential,values); gpuErrorCheck();
}
__global__ void columnFFTGather_kernel(size_t count,int nbasis,int nr,const int* indices,const complex* input,complex* output)
{
	size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
	if(i<count) output[i]=input[(i/nbasis)*nr+indices[i%nbasis]];
}
void columnFFTGather_gpu(size_t count,int nbasis,int nr,const int* indices,const complex* input,complex* output)
{
	columnFFTGather_kernel<<<(count+255)/256,256>>>(count,nbasis,nr,indices,input,output); gpuErrorCheck();
}
__global__ void fieldSymmetryAccumulate_kernel(int nr,const int* indices,const complex* phases,const complex* input,
	complex* output,complex weight,bool conjugate)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<nr) output[indices[i]]+=weight*phases[i]*(conjugate ? input[i].conj() : input[i]);
}
void fieldSymmetryAccumulate_gpu(int nr,const int* indices,const complex* phases,const complex* input,
	complex* output,complex weight,bool conjugate)
{
	fieldSymmetryAccumulate_kernel<<<(nr+255)/256,256>>>(nr,indices,phases,input,output,weight,conjugate); gpuErrorCheck();
}
