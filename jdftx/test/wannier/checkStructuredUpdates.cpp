#include <core/Util.h>
#include <core/matrix.h>
#include <cmath>

int main(int argc,char** argv)
{
    initSystem(argc,argv);
    for(auto dimensions: {std::make_pair(3,17),std::make_pair(17,3),std::make_pair(4,4)})
    for(int rankCase=0;rankCase<3;rankCase++)
    for(double alpha: {0.,1e-9,.2,-2.,10.,1000.}) {
        int p=dimensions.first,q=dimensions.second;
        matrix B(p,q,isGpuEnabled()),A(29,p+q,isGpuEnabled()); randomize(B);randomize(A);
        if(rankCase==0) B*=0.;
        if(rankCase==1) for(int j=0;j<q;j++) for(int i=1;i<p;i++) B.data()[B.index(i,j)]=B.data()[B.index(0,j)];
        matrix full=zeroes(p+q,p+q);
        full.set(0,p,p,p+q,B);full.set(p,p+q,0,p,dagger(B));
        matrix actual=rightMultiplyOffDiagonalExp(A,B,alpha),reference=A*cis(alpha*full);
        double error=nrm2(actual-reference)/std::max(1.,nrm2(reference));
        if(error>1e-10) die("Structured update mismatch %.3le (p=%d q=%d rank=%d alpha=%g).\n",error,p,q,rankCase,alpha);
        matrix X(37,p+q,isGpuEnabled()),Y(37,p+q,isGpuEnabled());randomize(X);randomize(Y);
        matrix product=dagger(X)*Y,diagonal=columnDot(X,Y);
        for(int i=0;i<p+q;i++) if(abs(product.data()[product.index(i,i)]-diagonal.data()[i])>1e-11)
            die("Complex column contraction mismatch.\n");
    }
    {
        const int p=24,q=384;
        matrix B(p,q,isGpuEnabled()),A(600,p+q,isGpuEnabled());randomize(B);randomize(A);B*=.02;
        matrix full=zeroes(p+q,p+q);full.set(0,p,p,p+q,B);full.set(p,p+q,0,p,dagger(B));
        const double start=clock_sec();matrix reference=A*cis(.5*full);
        // Host read synchronizes GPU work for a meaningful elapsed comparison.
        volatile double sync=reference.data()[0].real(); (void)sync;
        const double denseSeconds=clock_sec()-start;
        const double structuredStart=clock_sec();matrix actual=rightMultiplyOffDiagonalExp(A,B,.5);
        sync=actual.data()[0].real();
        const double structuredSeconds=clock_sec()-structuredStart;
        if(nrm2(actual-reference)/std::max(1.,nrm2(reference))>1e-10) die("Large structured update mismatch.\n");
        logPrintf("Outer-window update benchmark (24+384): dense %.6lf s; structured %.6lf s.\n",denseSeconds,structuredSeconds);
    }
    logPrintf("Structured update and complex column contraction regression passed.\n");
    finalizeSystem();return 0;
}
