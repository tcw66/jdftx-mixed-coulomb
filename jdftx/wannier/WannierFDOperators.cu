#include <core/GpuKernelUtils.h>

__global__ void wannierDiagonal_kernel(int n, int stride, const complex* in, complex* out)
{
    int i=kernelIndex1D();if(i<n) out[i]=in[i*stride];
}
void wannierDiagonal_gpu(int n, int stride, const complex* in, complex* out)
{
    GpuLaunchConfig1D glc(wannierDiagonal_kernel,n);
    wannierDiagonal_kernel<<<glc.nBlocks,glc.nPerBlock>>>(n,stride,in,out);
    gpuErrorCheck();
}

__global__ void wannierOverlapGradient_kernel(int n, int ni, int nj, int leftRows, int rightRows,
    const complex* left, const complex* right, const complex* d, complex* gi, complex* gj)
{
    int index=kernelIndex1D();
    const int i=index%n, j=index/n;
    // Each output element has one writer. This also works for self edges:
    // gi==gj then both updates are performed by the same thread.
    if(j<ni) gi[index]+=(left[j+i*leftRows]*d[i]).conj();
    if(j<nj) gj[index]+=d[i]*right[i+j*rightRows];
}
void wannierOverlapGradient_gpu(int n, int ni, int nj, int leftRows, int rightRows,
    const complex* left, const complex* right, const complex* d, complex* gi, complex* gj)
{
    GpuLaunchConfig1D glc(wannierOverlapGradient_kernel,n*std::max(ni,nj));
    wannierOverlapGradient_kernel<<<glc.nBlocks,glc.nPerBlock>>>(n,ni,nj,leftRows,rightRows,left,right,d,gi,gj);
    gpuErrorCheck();
}

__global__ void wannierCenterVariation_kernel(int n, int ni, int nj,
    const complex* left, const complex* right, const complex* ai, const complex* aj, complex* out)
{
    int i=kernelIndex1D();if(i>=n) return;
    complex value=0.;
    for(int j=0;j<nj;j++) value+=right[i+j*n]*aj[j+i*nj];
    for(int j=0;j<ni;j++) value-=ai[j+i*ni].conj()*left[j+i*ni];
    out[i]=value;
}
void wannierCenterVariation_gpu(int n, int ni, int nj, const complex* left, const complex* right,
    const complex* ai, const complex* aj, complex* out)
{
    GpuLaunchConfig1D glc(wannierCenterVariation_kernel,n);
    wannierCenterVariation_kernel<<<glc.nBlocks,glc.nPerBlock>>>(n,ni,nj,left,right,ai,aj,out);
    gpuErrorCheck();
}
