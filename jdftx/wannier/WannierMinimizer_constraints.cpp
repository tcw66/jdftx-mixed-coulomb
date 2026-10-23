/* Hard center equality constraints for reciprocal-space Wannier localization.
   Uses allowed inner-window-preserving rotations, not a penalty functional.
   Distributed gradients use the WannierGradient metric, including its B1 factor 2.
*/
#include <wannier/WannierMinimizer.h>
#include <algorithm>
#include <cmath>
#include <deque>

namespace {
double centerError(const std::vector<double>& c)
{
    double error=0.;
    for(size_t i=0;i<c.size();i+=3)
        error=std::max(error,std::sqrt(c[i]*c[i]+c[i+1]*c[i+1]+c[i+2]*c[i+2]));
    return error;
}

double localGradientDot(const WannierGradient& x,const WannierGradient& y)
{
    double value=0.;
    for(size_t ik=x.ikStart();ik<x.ikStop();ik++)
    {
        if(x.B1[ik]) value+=2.*dotc(x.B1[ik],y.B1[ik]).real();
        value+=dotc(x.B2[ik],y.B2[ik]).real();
    }
    return value;
}

// Pseudoinverse in equilibrated center coordinates permits redundant constraints. Feasibility is checked
// independently against ALL residuals, so a null-space residual is never ignored.
matrix gramInverse(const std::vector<WannierGradient>& J)
{
    int n=J.size(); matrix G=zeroes(n,n);
    std::vector<double> packed;
    for(int i=0;i<n;i++) for(int j=0;j<=i;j++) packed.push_back(localGradientDot(J[i],J[j]));
    mpiWorld->allReduceData(packed,MPIUtil::ReduceSum);
    size_t pos=0;
    for(int i=0;i<n;i++) for(int j=0;j<=i;j++)
    {
        double value=packed[pos++];
        G.data()[G.index(i,j)]=value; G.data()[G.index(j,i)]=value;
    }
    // Equilibrate center rows before truncating eigenvalues. Otherwise one
    // sensitive center can cause independent, weaker constraints to be lost.
    diagMatrix scale(n);
    for(int i=0;i<n;i++) scale[i]=1./std::sqrt(std::max(1e-20,G.data()[G.index(i,i)].real()));
    G=scale*G*scale;
    matrix V; diagMatrix eig;
    G.diagonalize(V,eig);
    double largest=eig.empty()?0.:eig.back();
    for(double& x:eig) x=(x>std::max(1e-20,largest*1e-10)) ? 1./x : 0.;
    return scale*(V*eig*dagger(V))*scale;
}

std::vector<double> applyConstraintMatrix(const matrix& A,const std::vector<double>& b)
{
    std::vector<double> result(b.size(),0.);
    const complex* data=A.data();
    for(size_t i=0;i<b.size();i++) for(size_t j=0;j<b.size();j++)
        result[i]+=data[A.index(i,j)].real()*b[j];
    return result;
}


}

bool WannierMinimizer::getCenterConstraints(std::vector<double>&,std::vector<WannierGradient>*)
{
    die("Hard center constraints require a reciprocal-space localization measure.\n");
    return false;
}

bool WannierMinimizer::prepareCenterJacobian(std::vector<double>&) { die("Center Jacobian unavailable.\n");return false; }
std::vector<double> WannierMinimizer::applyCenterJacobian(const WannierGradient&) { die("Center Jacobian unavailable.\n");return {}; }
WannierGradient WannierMinimizer::applyCenterJacobianAdjoint(const std::vector<double>&) { die("Center Jacobian unavailable.\n");return {}; }
std::vector<double> WannierMinimizer::centerJacobianScaling() { return {}; }
size_t WannierMinimizer::centerJacobianStorage() const { return 0; }

void WannierMinimizer::checkFixedCenterInvariants()
{
    double orthError=0.,innerError=0.;
    for(size_t ik=ikStart;ik<ikStop;ik++)
    {
        const KmeshEntry& ki=kMesh[ik];
        const matrix W=ki.U(0,nBands,0,nCenters);
        orthError=std::max(orthError,nrm2(dagger(W)*W-eye(nCenters)));
        if(ki.nFixed)
        {
            const matrix F=ki.U1(0,nBands,0,ki.nFixed);
            innerError=std::max(innerError,nrm2(F-W*(dagger(W)*F)));
        }
    }
    mpiWorld->allReduce(orthError,MPIUtil::ReduceMax);
    mpiWorld->allReduce(innerError,MPIUtil::ReduceMax);
    logPrintf("FixedCenters: orthonormality error %.3le; inner-window inclusion error %.3le\n",orthError,innerError);
    if(orthError>1e-7 || innerError>1e-7) die("Hard-center optimization lost orthonormality or inner-window inclusion.\n");
}

double WannierMinimizer::minimizeFixedCenters()
{
    const MinimizeParams& p=wannier.minParams;
    centerDiagnostics=p.fdTest;
    if(p.maxThreshold) die("Hard-center optimization supports knormThreshold, but not maxThreshold.\n");
    const double tolerance=wannier.fixedCenterTolerance;
    double restorationTolerance=tolerance;
    double updateSeconds=0.,objectiveSeconds=0.,restoreCommunicationSeconds=0.;
    size_t restorationCount=0;
    auto timedStep=[&](const WannierGradient& direction,double alpha) {
        const double start=clock_sec(); step(direction,alpha); updateSeconds+=clock_sec()-start;
    };
    auto timedOmega=[&]() {
        const double start=clock_sec();double E=getOmega();objectiveSeconds+=clock_sec()-start;return E;
    };
    auto timedCompute=[&](WannierGradient& grad) {
        const double start=clock_sec();double E=compute(&grad,0);objectiveSeconds+=clock_sec()-start;return E;
    };
    typedef std::vector<std::pair<matrix,matrix>> State;
    auto save=[&]() {
        State state;
        for(size_t ik=ikStart;ik<ikStop;ik++) state.emplace_back(kMesh[ik].U1,kMesh[ik].U2);
        return state;
    };
    auto restore=[&](const State& state,bool communicate=true) {
        for(size_t ik=ikStart;ik<ikStop;ik++)
        {
            auto& ki=kMesh[ik];
            ki.U1=state[ik-ikStart].first;ki.U2=state[ik-ikStart].second;ki.U=ki.U1*ki.U2;
        }
        if(communicate) {const double start=clock_sec();bcastU();restoreCommunicationSeconds+=clock_sec()-start;}
    };
    // One linearization is shared by normal correction and tangent projection.
    // Explicit remains available as a reference; MatrixFree never stores J rows.
    const bool matrixFree=wannier.fixedCenterMatrixFree;
    std::vector<WannierGradient> rows;matrix Gi;
    size_t normalSolves=0,normalProducts=0,peakStorage=0;
    double prepareSeconds=0.,solveSeconds=0.;
    auto prepare=[&](std::vector<double>& c) {
        const double start=clock_sec();size_t bytes=0;
        if(matrixFree)
        {
            if(!prepareCenterJacobian(c)) return false;
            bytes=centerJacobianStorage();
        }
        else
        {
            if(!getCenterConstraints(c,&rows)) return false;
            Gi=gramInverse(rows);bytes=Gi.nRows()*Gi.nCols()*sizeof(complex);
            if(p.fdTest)
            {
                if(!prepareCenterJacobian(c)) return false;
                bytes+=centerJacobianStorage();
            }
            for(const auto& row:rows) for(size_t ik=ikStart;ik<ikStop;ik++)
                bytes+=(row.B1[ik].nRows()*row.B1[ik].nCols()+row.B2[ik].nRows()*row.B2[ik].nCols())*sizeof(complex);
        }
        peakStorage=std::max(peakStorage,bytes);prepareSeconds+=clock_sec()-start;
        return true;
    };
    auto Jv=[&](const WannierGradient& v) {
        if(matrixFree) return applyCenterJacobian(v);
        std::vector<double> b(rows.size());
        for(size_t i=0;i<rows.size();i++) b[i]=localGradientDot(rows[i],v);
        mpiWorld->allReduceData(b,MPIUtil::ReduceSum);return b;
    };
    auto adjoint=[&](const std::vector<double>& x) {
        if(matrixFree) return applyCenterJacobianAdjoint(x);
        WannierGradient v=rows[0];v*=0.;
        for(size_t i=0;i<rows.size();i++) axpy(x[i],rows[i],v);
        return v;
    };
    auto scalarDot=[](const std::vector<double>& x,const std::vector<double>& y) {
        double a=0.;for(size_t i=0;i<x.size();i++) a+=x[i]*y[i];return a;
    };
    // Recycled positive-curvature secants approximate the inverse normal operator.
    // They are frozen within each PCG solve, keeping its preconditioner linear/SPD.
    std::deque<std::pair<std::vector<double>,std::vector<double>>> normalHistory;
    auto normal=[&](const std::vector<double>& b,WannierGradient& correction) {
        normalSolves++;const double start=clock_sec();
        if(!matrixFree) { correction=adjoint(applyConstraintMatrix(Gi,b));solveSeconds+=clock_sec()-start;return true; }
        const size_t m=b.size();
        const auto scale=centerJacobianScaling();
        std::vector<double> x(m,0.),r=b,z(m),d(m);
        std::deque<std::pair<std::vector<double>,std::vector<double>>> newHistory;
        auto preconditionNormal=[&](const std::vector<double>& input) {
            std::vector<double> result=input,alpha(normalHistory.size());
            for(int j=int(normalHistory.size())-1;j>=0;j--) {
                const auto& pair=normalHistory[j];
                alpha[j]=scalarDot(pair.first,result)/scalarDot(pair.first,pair.second);
                for(size_t i=0;i<m;i++) result[i]-=alpha[j]*pair.second[i];
            }
            for(size_t i=0;i<m;i++) result[i]/=scale[i];
            for(size_t j=0;j<normalHistory.size();j++) {
                const auto& pair=normalHistory[j];
                double beta=scalarDot(pair.second,result)/scalarDot(pair.first,pair.second);
                for(size_t i=0;i<m;i++) result[i]+=(alpha[j]-beta)*pair.first[i];
            }
            return result;
        };
        const double target=std::max(1e-13,1e-9*std::sqrt(scalarDot(b,b)));
        z=preconditionNormal(r);
        d=z;double rz=scalarDot(r,z);
        bool solved=std::sqrt(scalarDot(r,r))<=target;
        // PCG on the positive-semidefinite normal operator, starting at zero.
        // Consistent redundant constraints are allowed. Check the TRUE residual
        // before accepting; never ignore an inconsistent/null-space residual.
        for(size_t iter=0;!solved && iter<4*m+40;iter++)
        {
            auto Ad=applyCenterJacobian(adjoint(d));normalProducts++;
            const double curvature=scalarDot(d,Ad);
            if(!std::isfinite(curvature) || curvature<=0. || !std::isfinite(rz)) break;
            if(curvature>1e-12*std::sqrt(scalarDot(d,d)*scalarDot(Ad,Ad))) {
                newHistory.emplace_back(d,Ad);
                if(newHistory.size()>8) newHistory.pop_front();
            }
            const double alpha=rz/curvature;
            for(size_t i=0;i<m;i++) {x[i]+=alpha*d[i];r[i]-=alpha*Ad[i];}
            if(iter%100==99)
            {
                logPrintf("FixedCenters: normal CG iteration %lu residual %.3le (target %.3le).\n",iter+1,std::sqrt(scalarDot(r,r)),target);
                logFlush();
            }
            if(std::sqrt(scalarDot(r,r))<=target)
            {
                auto Ax=applyCenterJacobian(adjoint(x));normalProducts++;
                for(size_t i=0;i<m;i++) r[i]=b[i]-Ax[i];
                solved=std::sqrt(scalarDot(r,r))<=target;
                if(solved) break;
                z=preconditionNormal(r);
                d=z;rz=scalarDot(r,z);continue;
            }
            z=preconditionNormal(r);
            const double next=scalarDot(r,z),beta=next/rz;
            for(size_t i=0;i<m;i++) d[i]=z[i]+beta*d[i];
            rz=next;
        }
        if(!solved && !normalHistory.empty()) {
            // Revert to the original diagonal PCG if recycling is unhelpful.
            normalHistory.clear();newHistory.clear();
            std::fill(x.begin(),x.end(),0.); r=b;z=preconditionNormal(r);d=z;rz=scalarDot(r,z);
            for(size_t iter=0;!solved && iter<4*m+40;iter++) {
                auto Ad=applyCenterJacobian(adjoint(d));normalProducts++;
                const double curvature=scalarDot(d,Ad);
                if(!std::isfinite(curvature) || curvature<=0. || !std::isfinite(rz)) break;
                const double alpha=rz/curvature;
                for(size_t i=0;i<m;i++) {x[i]+=alpha*d[i];r[i]-=alpha*Ad[i];}
                if(std::sqrt(scalarDot(r,r))<=target) {
                    auto Ax=applyCenterJacobian(adjoint(x));normalProducts++;
                    for(size_t i=0;i<m;i++) r[i]=b[i]-Ax[i];
                    solved=std::sqrt(scalarDot(r,r))<=target;
                    if(solved) break;
                    z=preconditionNormal(r);d=z;rz=scalarDot(r,z);continue;
                }
                z=preconditionNormal(r);const double next=scalarDot(r,z), beta=next/rz;
                for(size_t i=0;i<m;i++) d[i]=z[i]+beta*d[i];
                rz=next;
            }
        }
        if(!solved) { solveSeconds+=clock_sec()-start;return false; }
        for(auto& pair:newHistory) {normalHistory.push_back(std::move(pair));if(normalHistory.size()>8) normalHistory.pop_front();}
        correction=adjoint(x);solveSeconds+=clock_sec()-start;return true;
    };
    auto project=[&](const WannierGradient& v,WannierGradient& result) {
        WannierGradient correction;
        if(!normal(Jv(v),correction)) return false;
        result=v;axpy(-1.,correction,result);
        // A small residual relative to J*v can still leave a large normal
        // component when the raw gradient/Jacobian is large. Check tangency
        // of the RESULT, and refine rather than reporting a false projection.
        for(int refine=0;refine<4;refine++)
        {
            const auto drift=Jv(result);
            const double target=1e-8*std::max(1.,std::sqrt(dot(result,result)));
            if(std::sqrt(scalarDot(drift,drift))<=target) return true;
            if(!normal(drift,correction)) return false;
            axpy(-1.,correction,result);
        }
        const auto drift=Jv(result);
        const double target=1e-8*std::max(1.,std::sqrt(dot(result,result)));
        if(std::sqrt(scalarDot(drift,drift))<=target) return true;
        logPrintf("FixedCenters: projection is ill-conditioned: tangent residual %.3le exceeds %.3le.\n",std::sqrt(scalarDot(drift,drift)),target);
        return false;
    };
    // Damped Newton restoration in the minimum-norm normal direction.
    auto feasible=[&](bool verbose) {
        for(int iter=0;iter<30;iter++)
        {
            std::vector<double> c;
            if(!getCenterConstraints(c)) return false;
            const double error=centerError(c);
            if(verbose) logPrintf("FixedCenters: restore %d residual %.12le bohr\n",iter,error);
            if(error<=restorationTolerance) return true;
            restorationCount++;
            if(!prepare(c)) return false;
            if(verbose) { logPrintf("FixedCenters: center linearization ready.\n");logFlush(); }
            WannierGradient correction;
            if(!normal(c,correction)) return false;
            correction*=-1.;
            if(dot(correction,correction)<1e-26) return false;
            State start=save(); bool accepted=false;
            double scale=1.;
            for(int trial=0;trial<16;trial++,scale*=.5)
            {
                restore(start,false);timedStep(correction,scale);
                std::vector<double> next;
                if(getCenterConstraints(next) && centerError(next)<error*(1.-1e-4*scale)) { accepted=true;break; }
            }
            if(!accepted) { restore(start);return false; }
        }
        return false;
    };
    logPrintf("\nFixedCenters: enforcing %lu center equalities (tolerance %.3le bohr).\n",wannier.fixedCenters.size(),tolerance);
    logPrintf("FixedCenters: using projected Polak-Ribiere CG with Armijo search and Newton restoration; generic direction/line-search options do not apply.\n");

    logPrintf("FixedCenters: %s center solver.\n",matrixFree?"MatrixFree (PCG, overlap row scaling)":"Explicit (dense pseudoinverse)");

    // Independent finite-difference check of raw objective and center Jacobian.
    if(p.fdTest)
    {
        WannierGradient g;compute(&g,0);
        WannierGradient dx=g;randomize(dx);constrain(dx);
        dx*=1./std::sqrt(dot(dx,dx));
        std::vector<double> c;
        if(!prepare(c)) die("Cannot differentiate an undefined fixed center (zero overlap). Improve trial orbitals or windows.\n");
        const auto analytic=Jv(dx);
        if(matrixFree && c.size()<=24)
        {
            std::vector<WannierGradient> reference;std::vector<double> check;
            if(!getCenterConstraints(check,&reference)) die("Undefined explicit reference center.\n");
            double error=0.;
            for(size_t i=0;i<check.size();i++) error=std::max(error,std::abs(analytic[i]-dot(reference[i],dx)));
            logPrintf("FixedCenters: fdTest matrix-free/explicit product error %.3le\n",error);
            if(error>1e-9) die("Matrix-free/explicit center derivative mismatch.\n");
        }
        std::vector<double> weights(c.size());for(size_t i=0;i<weights.size();i++) weights[i]=std::sin(1.+i);
        const double adjointError=std::abs(scalarDot(weights,analytic)-dot(adjoint(weights),dx));
        logPrintf("FixedCenters: fdTest adjoint identity error %.3le\n",adjointError);
        if(adjointError>1e-9) die("Fixed-center adjoint identity failed.\n");
        State start=save();const double eps=1e-5;
        timedStep(dx,eps);std::vector<double> plus;if(!getCenterConstraints(plus)) die("Undefined center in derivative check.\n");double Ep=timedOmega();
        restore(start,false);timedStep(dx,-eps);std::vector<double> minus;if(!getCenterConstraints(minus)) die("Undefined center in derivative check.\n");double Em=timedOmega();
        restore(start);
        double err=0.;
        for(size_t i=0;i<c.size();i++) err=std::max(err,std::abs((plus[i]-minus[i])/(2*eps)-analytic[i]));
        double objError=std::abs((Ep-Em)/(2*eps)-dot(g,dx));
        logPrintf("FixedCenters: fdTest center Jacobian error %.3le; objective derivative error %.3le\n",err,objError);
        if(err>1e-6 || objError>1e-5) die("Fixed-center derivative check failed.\n");
    }
    if(!feasible(true)) die("Cannot satisfy fixed-center targets within the allowed inner/outer-window subspace. Targets may be infeasible, locally singular, or need better trial orbitals.\n");
    if(p.fdTest)
    {
        // Test the derivative of the actual feasible search curve, not only
        // the unconstrained rotation derivative. Use tighter restoration so
        // fixed feasibility noise does not dominate a small difference step.
        restorationTolerance=std::min(tolerance,1e-12);
        if(!feasible(false)) die("Cannot restore centers for feasible derivative check.\n");
        WannierGradient raw;timedCompute(raw);std::vector<double> c;
        if(!prepare(c)) die("Cannot prepare feasible derivative check.\n");
        WannierGradient dx=raw;randomize(dx);constrain(dx);
        WannierGradient tangent;
        if(!project(dx,tangent)) die("Cannot project feasible derivative check.\n");
        const double norm=std::sqrt(dot(tangent,tangent));
        if(norm>1e-12)
        {
            tangent*=1./norm;
            const auto drift=Jv(tangent);double driftNorm=std::sqrt(scalarDot(drift,drift));
            if(driftNorm>1e-7) die("Projected direction is not tangent to the center constraints.\n");
            const double expected=dot(raw,tangent);State start=save();double error=0.;
            for(double h:{1e-3,5e-4})
            {
                restore(start,false);timedStep(tangent,h);
                if(!feasible(false)) die("Cannot restore positive feasible derivative step.\n");
                const double plus=timedOmega();
                restore(start,false);timedStep(tangent,-h);
                if(!feasible(false)) die("Cannot restore negative feasible derivative step.\n");
                const double minus=timedOmega();
                error=std::abs((plus-minus)/(2*h)-expected);
            }
            restore(start);
            logPrintf("FixedCenters: fdTest feasible derivative error %.3le; tangent residual %.3le\n",error,driftNorm);
            if(error>5e-5*std::max(1.,std::abs(expected))) die("Feasible localization derivative check failed.\n");
        }
        restorationTolerance=tolerance;
    }
    WannierGradient prevG,prevD;
    bool havePrevious=false;
    size_t trialCount=0;
    double E=timedOmega(),previous=E,alpha=std::max(1e-4,p.alphaTstart);
    int smallChanges=0;bool converged=false,acceptedLast=false;
    for(int iter=0;!killFlag;iter++)
    {
        WannierGradient raw;E=timedCompute(raw);
        std::vector<double> c;
        if(!prepare(c) || centerError(c)>tolerance) die("Fixed-center feasibility lost.\n");
        WannierGradient g;
        if(!project(raw,g)) die("Fixed-center normal solve failed to converge. Try Explicit solver or improve the trial subspace.\n");
        const double gSq=dot(g,g),gNorm=std::sqrt(gSq/std::max(size_t(1),p.nDim));
        if(p.fdTest)
        {
            const auto drift=Jv(g);
            logPrintf("FixedCenters: stationarity rawNorm %.6le; tangentNorm %.6le; tangentResidual %.6le; phaseMargin %.6le; minOverlap %.6le\n",std::sqrt(dot(raw,raw)),std::sqrt(gSq),std::sqrt(scalarDot(drift,drift)),centerPhaseMargin,centerOverlapMinimum);
        }
        logPrintf("%sIter: %3d  Omega: %22.15le  |grad|_K: %.3le  centerError: %.3le  t[s]: %.2lf\n",p.linePrefix,iter,E,gNorm,centerError(c),clock_sec());logFlush();
        if(acceptedLast && p.energyDiffThreshold>0. && std::abs(E-previous)<p.energyDiffThreshold) smallChanges++;else smallChanges=0;
        acceptedLast=false;
        const bool gradDone=p.knormThreshold>0. && gNorm<p.knormThreshold;
        const bool energyDone=p.energyDiffThreshold>0. && smallChanges>=p.nEnergyDiff;
        const bool done=p.convergeAll ? ((p.knormThreshold<=0. || gradDone) && (p.energyDiffThreshold<=0. || energyDone) && (p.knormThreshold>0. || p.energyDiffThreshold>0.)) : (gradDone || energyDone);
        if(done || gSq<1e-28) {converged=true;break;}
        if(iter>=p.nIterations) break;
        WannierGradient d=g;d*=-1.;
        if(havePrevious)
        {
            WannierGradient diff=g;axpy(-1.,prevG,diff);
            double beta=std::max(0.,dot(g,diff)/std::max(1e-30,dot(prevG,prevG)));
            WannierGradient transported;
            if(project(prevD,transported)) axpy(beta,transported,d);
            else
            {
                // Transport is optional acceleration. A failed normal solve
                // must not abort a valid current projected-gradient step.
                logPrintf("FixedCenters: restarting CG after direction-transport solve failure.\n");
                havePrevious=false;
            }
        }
        double slope=dot(g,d);
        if(slope>=-1e-3*gSq) {d=g;d*=-1.;slope=-gSq;}
        State start=save();bool accepted=false;
        // Let backtracking determine the scale. A hard cap of three stalls
        // soft localization modes even when much larger steps decrease Omega.
        double trialAlpha=alpha*1.5;
        for(int trial=0;trial<20;trial++,trialAlpha*=.5)
        {
            restore(start,false);timedStep(d,trialAlpha);
            trialCount++;
            if(!feasible(false)) continue;
            double next=timedOmega();
            if(std::isfinite(next) && next<=E+1e-4*trialAlpha*slope)
            {accepted=true;break;}
        }
        if(!accepted)
        {
            restore(start);
            if(havePrevious) { havePrevious=false;alpha=std::max(1e-6,alpha*.25);continue; }
            die("Fixed-center line search failed; constraints remain satisfied, but localization did not converge.\n");
        }
        previous=E;alpha=trialAlpha;prevG=g;prevD=d;havePrevious=true;acceptedLast=true;
    }
    std::vector<double> finalC;
    if(!getCenterConstraints(finalC) || centerError(finalC)>tolerance) die("Final fixed-center tolerance not met.\n");
    E=timedOmega();
    logPrintf("FixedCenters: final maximum center residual %.12le bohr.\n",centerError(finalC));
    logPrintf("%s%s\n",p.linePrefix,converged?"Converged (feasible projected optimization).":"Not converged (iteration limit or interruption; center constraints satisfied).");
    mpiWorld->allReduce(updateSeconds,MPIUtil::ReduceMax);
    mpiWorld->allReduce(objectiveSeconds,MPIUtil::ReduceMax);
    mpiWorld->allReduce(restoreCommunicationSeconds,MPIUtil::ReduceMax);
    logPrintf("FixedCenters: orbital updates (including step communication) %.6lf s; objective/gradient %.6lf s; restore communication %.6lf s.\n",updateSeconds,objectiveSeconds,restoreCommunicationSeconds);
    logPrintf("FixedCenters: line-search trials %lu; Newton restoration corrections %lu.\n",trialCount,restorationCount);
    logPrintf("FixedCenters: normal solves %lu; matrix-free normal products %lu.\n",normalSolves,normalProducts);
    double storageMiB=peakStorage/(1024.*1024.);
    mpiWorld->allReduce(storageMiB,MPIUtil::ReduceMax);
    mpiWorld->allReduce(prepareSeconds,MPIUtil::ReduceMax);mpiWorld->allReduce(solveSeconds,MPIUtil::ReduceMax);
    logPrintf("FixedCenters: peak persistent derivative storage %.6lf MiB/rank; prepare %.6lf s; normal solves %.6lf s.\n",storageMiB,prepareSeconds,solveSeconds);
    checkFixedCenterInvariants();
    return E;
}
