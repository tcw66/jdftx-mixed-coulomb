#include <core/Util.h>
#include <wannier/WannierFDOperators.h>

int main(int argc,char** argv)
{
    initSystem(argc,argv);
    for(int n: {1,3,17}) for(bool self: {false,true}) for(bool rectangular: {false,true})
    {
        const int ni=n+2,nj=self ? ni : n+5;
        matrix left(ni,rectangular ? n : nj,isGpuEnabled());randomize(left);
        matrix right=rectangular ? matrix(n,nj,isGpuEnabled()) : left;
        if(rectangular) randomize(right);
        matrix gi=zeroes(n,ni),gj=zeroes(n,nj);
        matrix refi=gi,refj=gj;
        matrix derivatives(n,3);randomize(derivatives);
        // Accumulation over several edges exercises nonzero destinations and
        // packing offsets; a self edge deliberately aliases both destinations.
        for(size_t edge=0;edge<3;edge++)
        {
            std::vector<complex> d(n);
            for(int i=0;i<n;i++) d[i]=derivatives.data()[edge*n+i];
            refi+=dagger(matrix(left(0,ni,0,n))*d);
            if(self) refi+=d*matrix(right(0,n,0,nj));
            else refj+=d*matrix(right(0,n,0,nj));
        }
        for(size_t edge=0;edge<3;edge++)
            wannierOverlapGradient(left,right,derivatives,edge,gi,self ? gi : gj);
        if(nrm2(gi-refi)>1e-11 || (!self && nrm2(gj-refj)>1e-11))
            die("Wannier overlap derivative mismatch (n=%d self=%d rectangular=%d).\n",n,int(self),int(rectangular));
        if(rectangular)
        {
            matrix ai(ni,n,isGpuEnabled()),aj(nj,n,isGpuEnabled());randomize(ai);randomize(aj);
            matrix expected=columnDot(dagger(right),aj)-columnDot(ai,left);
            matrix variations(n,3,isGpuEnabled());
            for(size_t edge=0;edge<3;edge++) wannierCenterVariation(left,right,ai,aj,variations,edge);
            const complex* actual=variations.data();const complex* reference=expected.data();
            for(size_t edge=0;edge<3;edge++) for(int i=0;i<n;i++)
                if(abs(actual[edge*n+i]-reference[i])>1e-11) die("Center variation mismatch.\n");
        }
        matrix packed(n,3,isGpuEnabled());
        for(size_t edge=0;edge<3;edge++) wannierDiagonal(left,packed,edge);
        if(isGpuEnabled() && !left.isOnGpu()) die("Diagonal packing moved the overlap to the host.\n");
        const complex* diagonal=packed.data();const complex* values=left.data();
        for(size_t edge=0;edge<3;edge++) for(int i=0;i<n;i++)
            if(abs(diagonal[edge*n+i]-values[left.index(i,i)])>1e-14)
                die("Wannier diagonal packing mismatch.\n");
    }
    logPrintf("Wannier overlap derivative and diagonal packing regression passed.\n");
    finalizeSystem();return 0;
}
