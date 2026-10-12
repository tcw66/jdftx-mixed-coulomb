/* Exact uniform-k-mesh correlations. GPL v3 or later; part of JDFTx. */
#include <electronic/KMeshCorrelation.h>
#include <core/GpuKernelUtils.h>

__device__ complex kMeshPhase(int r,const vector3<int>& shape,const vector3<>& k)
{
	int z=r%shape[2],xy=r/shape[2],y=xy%shape[1],x=xy/shape[1];
	return cis((2*M_PI)*(k[0]*x/shape[0]+k[1]*y/shape[1]+k[2]*z/shape[2]));
}
__global__ void kMeshPackOrbital_kernel(size_t total,int nk,int centers,int n,int start,int count,
	vector3<int> shape,const int* slotToK,const vector3<>* mesh,const complex* const* orbital,complex* values)
{
	size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
	if(i<total) {
		int r=i/nk,ik=slotToK[i%nk];
		values[i]=r<count ? orbital[ik*centers+n][start+r]*kMeshPhase(start+r,shape,mesh[ik]) : complex();
	}
}
void kMeshPackOrbital_gpu(int nk,int centers,int n,int start,int count,int tile,
	const vector3<int>& shape,const int* slotToK,const vector3<>* mesh,const complex* const* orbital,complex* values)
{
	size_t total=size_t(nk)*tile;
	kMeshPackOrbital_kernel<<<(total+255)/256,256>>>(total,nk,centers,n,start,count,shape,slotToK,mesh,orbital,values); gpuErrorCheck();
}
__global__ void kMeshPower_kernel(size_t count,complex* values)
{
	size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
	if(i<count) values[i]=complex(values[i].norm());
}
void kMeshPower_gpu(size_t count,complex* values)
{
	kMeshPower_kernel<<<(count+255)/256,256>>>(count,values); gpuErrorCheck();
}
__global__ void kMeshStoreSource_kernel(size_t total,int nk,int nr,int start,vector3<int> shape,
	const int* qToSlot,const vector3<>* qmesh,const complex* values,complex* source)
{
	size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
	if(i<total) {
		int r=i/nk,iq=i%nk;
		source[size_t(iq)*nr+start+r]=values[size_t(r)*nk+qToSlot[iq]]*kMeshPhase(start+r,shape,-qmesh[iq])/(double(nk)*nk);
	}
}
void kMeshStoreSource_gpu(int nk,int nr,int start,int count,const vector3<int>& shape,
	const int* qToSlot,const vector3<>* qmesh,const complex* values,complex* source)
{
	size_t total=size_t(nk)*count;
	kMeshStoreSource_kernel<<<(total+255)/256,256>>>(total,nk,nr,start,shape,qToSlot,qmesh,values,source); gpuErrorCheck();
}
__global__ void kMeshPackPotential_kernel(size_t total,int nk,int start,int count,vector3<int> shape,
	const int* slotToQ,const vector3<>* qmesh,const complex* const* potential,complex* values)
{
	size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
	if(i<total) {
		int r=i/nk,iq=slotToQ[i%nk];
		values[i]=(r<count && potential[iq]) ? potential[iq][start+r].conj()*kMeshPhase(start+r,shape,-qmesh[iq]) : complex();
	}
}
void kMeshPackPotential_gpu(int nk,int start,int count,int tile,const vector3<int>& shape,
	const int* slotToQ,const vector3<>* qmesh,const complex* const* potential,complex* values)
{
	size_t total=size_t(nk)*tile;
	kMeshPackPotential_kernel<<<(total+255)/256,256>>>(total,nk,start,count,shape,slotToQ,qmesh,potential,values); gpuErrorCheck();
}
__global__ void kMeshMultiply_kernel(size_t count,const complex* a,complex* b)
{
	size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
	if(i<count) b[i]*=a[i];
}
void kMeshMultiply_gpu(size_t count,const complex* a,complex* b)
{
	kMeshMultiply_kernel<<<(count+255)/256,256>>>(count,a,b); gpuErrorCheck();
}
__global__ void kMeshConduction_kernel(int nk,int centers,int n,int start,int count,vector3<int> shape,
	const int* kToSlot,const vector3<>* mesh,const complex* const* orbital,
	int emptyCount,const int* emptyCenters,const complex* weighted,complex* const* correction,double scale)
{
	int ik=blockIdx.x/emptyCount,m=emptyCenters[blockIdx.x%emptyCount];
	complex value;
	for(int r=threadIdx.x;r<count;r+=blockDim.x)
		value+=orbital[ik*centers+m][start+r].conj()*kMeshPhase(start+r,shape,-mesh[ik])*weighted[size_t(r)*nk+kToSlot[ik]];
	__shared__ complex sums[256]; sums[threadIdx.x]=value; __syncthreads();
	for(int stride=128;stride;stride/=2) {
		if(threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
		__syncthreads();
	}
	if(!threadIdx.x) correction[ik][size_t(n)*centers+m]+=scale*sums[0];
}
void kMeshConduction_gpu(int nk,int centers,int n,int start,int count,const vector3<int>& shape,
	const int* kToSlot,const vector3<>* mesh,const complex* const* orbital,
	int emptyCount,const int* emptyCenters,const complex* weighted,complex* const* correction,double scale)
{
	kMeshConduction_kernel<<<nk*emptyCount,256>>>(nk,centers,n,start,count,shape,kToSlot,mesh,orbital,emptyCount,emptyCenters,weighted,correction,scale);
	gpuErrorCheck();
}
