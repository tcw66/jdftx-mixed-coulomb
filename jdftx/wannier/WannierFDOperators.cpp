#include <wannier/WannierFDOperators.h>
#ifdef GPU_ENABLED
void wannierCenterVariation_gpu(int n, int ni, int nj, const complex* left, const complex* right,
    const complex* ai, const complex* aj, complex* out);
void wannierDiagonal_gpu(int n, int stride, const complex* in, complex* out);
void wannierOverlapGradient_gpu(int n, int ni, int nj, int leftRows, int rightRows,
    const complex* left, const complex* right, const complex* d, complex* gi, complex* gj);
#endif

void wannierDiagonal(const matrix& overlap, matrix& packed, size_t edge)
{
    const int n=packed.nRows();
    assert(n<=overlap.nRows() && n<=overlap.nCols());
    assert(edge<size_t(packed.nCols()));
#ifdef GPU_ENABLED
    wannierDiagonal_gpu(n,overlap.nRows()+1,overlap.dataGpu(),packed.dataGpu()+edge*n);
#else
    const complex* in=overlap.data();complex* out=packed.data()+edge*n;
    for(int i=0;i<n;i++) out[i]=in[i*(overlap.nRows()+1)];
#endif
}

void wannierOverlapGradient(const matrix& left, const matrix& right,
    const matrix& derivatives, size_t edge, matrix& gi, matrix& gj)
{
    const int n=derivatives.nRows(), ni=gi.nCols(), nj=gj.nCols();
    assert(gi.nRows()==n && gj.nRows()==n);
    assert(left.nRows()==ni && left.nCols()>=n);
    assert(right.nRows()>=n && right.nCols()==nj);
    assert(edge<size_t(derivatives.nCols()));
#ifdef GPU_ENABLED
    wannierOverlapGradient_gpu(n,ni,nj,left.nRows(),right.nRows(),
        left.dataGpu(),right.dataGpu(),derivatives.dataGpu()+edge*n,gi.dataGpu(),gj.dataGpu());
#else
    const complex* l=left.data();const complex* r=right.data();
    const complex* d=derivatives.data()+edge*n;
    complex* outi=gi.data();complex* outj=gj.data();
    for(int j=0;j<ni;j++) for(int i=0;i<n;i++)
        outi[i+j*n]+=(l[j+i*left.nRows()]*d[i]).conj();
    for(int j=0;j<nj;j++) for(int i=0;i<n;i++)
        outj[i+j*n]+=d[i]*r[i+j*right.nRows()];
#endif
}

void wannierCenterVariation(const matrix& left, const matrix& right,
    const matrix& ai, const matrix& aj, matrix& packed, size_t edge)
{
    const int n=packed.nRows(),ni=left.nRows(),nj=right.nCols();
    assert(left.nCols()==n && right.nRows()==n);
    assert(ai.nRows()==ni && ai.nCols()==n && aj.nRows()==nj && aj.nCols()==n);
    assert(edge<size_t(packed.nCols()));
#ifdef GPU_ENABLED
    wannierCenterVariation_gpu(n,ni,nj,left.dataGpu(),right.dataGpu(),ai.dataGpu(),aj.dataGpu(),packed.dataGpu()+edge*n);
#else
    const complex* l=left.data();const complex* r=right.data();
    const complex* a=ai.data();const complex* b=aj.data();complex* out=packed.data()+edge*n;
    for(int i=0;i<n;i++)
    {
        complex value=0.;
        for(int j=0;j<nj;j++) value+=r[i+j*n]*b[j+i*nj];
        for(int j=0;j<ni;j++) value-=a[j+i*ni].conj()*l[j+i*ni];
        out[i]=value;
    }
#endif
}
