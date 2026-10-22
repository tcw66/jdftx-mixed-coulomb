/* Matrix-free center derivatives in the inner-window-preserving rotation metric. */
#include <wannier/WannierMinimizerFD.h>
#include <algorithm>
#include <cmath>

bool WannierMinimizerFD::prepareCenterJacobian(std::vector<double>& residual)
{
    if(!getCenterConstraints(residual)) return false;
    const size_t nb=edges[0].size(), nc=wannier.fixedCenters.size();
    const bool tdc=wannier.localizationMeasure==Wannier::LM_TruncatedDC;
    centerLeft.clear(); centerRight.clear();
    centerLeft.resize((ikStop-ikStart)*nb);centerRight.resize(centerLeft.size());
    centerZ.assign(centerLeft.size()*nc,complex(0.,0.));
    std::vector<complex> rho(nb*nc,complex(0.,0.));
    // Communicate the coordinate frame once per linearization. Keeping the
    // overlap factors in the U1 frame avoids U2 conversions in every product.
    std::vector<matrix> frames(kMesh.size());
    for(size_t ik=0;ik<kMesh.size();ik++) if(kMesh[ik].U)
        frames[ik]=(ik>=ikStart && ik<ikStop) ? kMesh[ik].U2 : zeroes(kMesh[ik].nIn,kMesh[ik].nIn);
    std::vector<MPIUtil::Request> frameRequests;
    for(size_t ik=0;ik<kMesh.size();ik++) if(kMesh[ik].mpi)
    {
        MPIUtil::Request request;kMesh[ik].mpi->bcastData(frames[ik],0,&request);frameRequests.push_back(request);
    }
    MPIUtil::waitAll(frameRequests);
    size_t pos=0;
    for(size_t ik=ikStart;ik<ikStop;ik++) for(size_t ib=0;ib<nb;ib++,pos++)
    {
        const auto& edge=edges[ik][ib];
        const auto& Ui=kMesh[ik].U;const auto& Uj=kMesh[edge.ik].U;
        centerLeft[pos]=dagger(Ui)*(edge.M0*Uj(0,nBands,0,nCenters));
        centerRight[pos]=dagger(Ui(0,nBands,0,nCenters))*edge.M0*Uj;
        size_t ic=0;
        for(const auto& target:wannier.fixedCenters)
        {
            const int n=target.first;
            complex z=centerLeft[pos].data()[centerLeft[pos].index(n,n)];
            centerZ[pos*nc+ic]=z;
            if(tdc) rho[ib*nc+ic]+=kMesh[ik].point.weight*z;
            ic++;
        }
        centerLeft[pos]=frames[ik]*centerLeft[pos];
        centerRight[pos]=centerRight[pos]*dagger(frames[edge.ik]);
    }
    if(tdc)
    {
        mpiWorld->allReduceData(rho,MPIUtil::ReduceSum);
        for(size_t p=0;p<centerLeft.size();p++) for(size_t ic=0;ic<nc;ic++)
            centerZ[p*nc+ic]=rho[(p%nb)*nc+ic];
    }
    // Cheap overlap-based row equilibration, not a stored Jacobian diagonal.
    centerScaling.assign(3*nc,0.);pos=0;
    for(size_t ik=ikStart;ik<ikStop;ik++) for(size_t ib=0;ib<nb;ib++,pos++)
        for(size_t ic=0;ic<nc;ic++) for(int d=0;d<3;d++)
        {
            const auto& edge=edges[ik][ib];
            const double a=kMesh[ik].point.weight*edge.wb*edge.b[d];
            centerScaling[3*ic+d]+=a*a/centerZ[pos*nc+ic].norm();
        }
    mpiWorld->allReduceData(centerScaling,MPIUtil::ReduceSum);
    double largest=0.;for(double x:centerScaling) largest=std::max(largest,x);
    for(double& x:centerScaling) x=std::max(x,std::max(1e-20,largest*1e-8));
    return true;
}

std::vector<double> WannierMinimizerFD::centerJacobianScaling() { return centerScaling; }

size_t WannierMinimizerFD::centerJacobianStorage() const
{
    size_t bytes=centerZ.size()*sizeof(complex)+centerScaling.size()*sizeof(double);
    for(const auto& M:centerLeft) bytes+=M.nRows()*M.nCols()*sizeof(complex);
    for(const auto& M:centerRight) bytes+=M.nRows()*M.nCols()*sizeof(complex);
    return bytes;
}

std::vector<double> WannierMinimizerFD::applyCenterJacobian(const WannierGradient& v)
{
    assert(v.wmin==this);
    // dW = i U1 (H1+H2) Q, Q=U2(:,Wannier). The cache is in the U1
    // frame, so only HQ's rectangular columns need to be formed/broadcast.
    std::vector<matrix> A(kMesh.size());
    for(size_t ik=0;ik<kMesh.size();ik++) if(kMesh[ik].U)
    {
        const auto& ki=kMesh[ik];A[ik]=zeroes(ki.nIn,nCenters);
        if(ik<ikStart || ik>=ikStop) continue;
        const matrix Q=ki.U2(0,ki.nIn,0,nCenters);
        matrix HQ=zeroes(ki.nIn,nCenters);
        if(ki.nIn>nCenters)
        {
            HQ.set(ki.nFixed,nCenters,0,nCenters,v.B1[ik]*Q(nCenters,ki.nIn,0,nCenters));
            HQ.set(nCenters,ki.nIn,0,nCenters,dagger(v.B1[ik])*Q(ki.nFixed,nCenters,0,nCenters));
        }
        HQ.set(nFrozen,nCenters,0,nCenters,HQ(nFrozen,nCenters,0,nCenters)+v.B2[ik]*Q(nFrozen,nCenters,0,nCenters));
        A[ik]=HQ;
    }
    std::vector<MPIUtil::Request> requests;
    for(size_t ik=0;ik<kMesh.size();ik++) if(kMesh[ik].mpi)
    {
        MPIUtil::Request request;kMesh[ik].mpi->bcastData(A[ik],0,&request);requests.push_back(request);
    }
    MPIUtil::waitAll(requests);
    const size_t nb=edges[0].size(),nc=wannier.fixedCenters.size();
    std::vector<double> result(3*nc,0.);size_t pos=0;
    for(size_t ik=ikStart;ik<ikStop;ik++) for(size_t ib=0;ib<nb;ib++,pos++)
    {
        const auto& edge=edges[ik][ib];
        const matrix dM=complex(0.,1.)*(centerRight[pos]*A[edge.ik]-dagger(A[ik])*centerLeft[pos]);
        size_t ic=0;
        for(const auto& target:wannier.fixedCenters)
        {
            int n=target.first;
            const double phaseDerivative=(dM.data()[dM.index(n,n)]/centerZ[pos*nc+ic]).imag();
            for(int d=0;d<3;d++) result[3*ic+d]-=kMesh[ik].point.weight*edge.wb*edge.b[d]*phaseDerivative;
            ic++;
        }
    }
    // All constraints travel in one collective, independently of their count.
    mpiWorld->allReduceData(result,MPIUtil::ReduceSum);
    return result;
}

WannierGradient WannierMinimizerFD::applyCenterJacobianAdjoint(const std::vector<double>& v)
{
    const size_t nb=edges[0].size(),nc=wannier.fixedCenters.size();
    assert(v.size()==3*nc);
    for(auto& ki:kMesh) if(ki.U) ki.Omega_UdotU=zeroes(nCenters,ki.nIn);
    size_t pos=0;
    for(size_t ik=ikStart;ik<ikStop;ik++) for(size_t ib=0;ib<nb;ib++,pos++)
    {
        const auto& edge=edges[ik][ib];
        std::vector<complex> derivative(nCenters,complex(0.,0.));size_t ic=0;
        for(const auto& target:wannier.fixedCenters)
        {
            double coefficient=0.;for(int d=0;d<3;d++) coefficient-=v[3*ic+d]*kMesh[ik].point.weight*edge.wb*edge.b[d];
            derivative[target.first]=coefficient*complex(0.,-1.)/centerZ[pos*nc+ic];ic++;
        }
        kMesh[ik].Omega_UdotU+=dagger(centerLeft[pos]*derivative);
        kMesh[edge.ik].Omega_UdotU+=derivative*centerRight[pos];
    }
    WannierGradient result;collectRotationGradient(result,true);return result;
}
