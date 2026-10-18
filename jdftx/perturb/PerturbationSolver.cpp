/*-------------------------------------------------------------------
Copyright 2022 Brandon Li

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#include <perturb/PerturbationSolver.h>
#include <perturb/DFPTGroundState.h>
#include <perturb/TestPerturbation.h>
#include <electronic/Everything.h>
#include <electronic/ElecMinimizer.h>
#include <electronic/ColumnBundle.h>
#include <electronic/ColumnBundleTransform.h>
#include <electronic/ColumnBundleFFT.h>
#include <electronic/ExCorr.h>
#include <core/matrix.h>
#include <core/Units.h>
#include <core/Minimize.h>
#include <core/ScalarField.h>
#include <core/ScalarFieldIO.h>
#include <core/Operators.h>
#include <core/MPIUtil.h>
#include <cstdio>
#include <cmath>
#include <core/Operators.h>


void PerturbationGradient::init(const Everything& e)
{	eInfo = &e.eInfo;
	pInfo = &e.pertInfo;
	if (pInfo->commensurate)
		X.resize(eInfo->nStates);
	else
		X.resize(eInfo->nStates*2);
}

PerturbationGradient& PerturbationGradient::operator*=(double alpha)
{
	int nStates = eInfo->nStates;

	for(int q=eInfo->qStart; q<eInfo->qStop; q++)
	{
		if( X[q]) X[q] *= alpha;
		if (!pInfo->commensurate) if( X[q+nStates]) X[q+nStates] *= alpha;
	}

	return *this;
}

void axpy(double alpha, const PerturbationGradient& x, PerturbationGradient& y)
{	assert(x.eInfo == y.eInfo);

	int nStates = x.eInfo->nStates;

	for(int q=x.eInfo->qStart; q<x.eInfo->qStop; q++)
	{
		if(x.X[q]) { if(y.X[q]) axpy(alpha, x.X[q], y.X[q]); else y.X[q] = alpha*x.X[q]; }
		if (!x.pInfo->commensurate) if(x.X[q+nStates]) { if(y.X[q+nStates]) axpy(alpha, x.X[q+nStates], y.X[q+nStates]); else y.X[q+nStates] = alpha*x.X[q+nStates]; }
	}
}

double dot(const PerturbationGradient& x, const PerturbationGradient& y)
{	assert(x.eInfo == y.eInfo);

	int nStates = x.eInfo->nStates;
	double result = 0.0;

	#ifdef GPU_ENABLED
	if(x.pInfo->fullMeshResponse && x.pInfo->optimizeFrozenResponse) {
		result=batchedColumnDot(x.X,y.X,x.eInfo->qStart,x.eInfo->qStop,nStates,x.pInfo->commensurate || x.pInfo->oneSidedResponse);
		if(x.pInfo->oneSidedResponse) result*=2.;
		mpiWorld->allReduce(result,MPIUtil::ReduceSum);
		return result;
	}
	#endif
	for(int q=x.eInfo->qStart; q<x.eInfo->qStop; q++)
	{
		if(x.X[q] && y.X[q]) result += dotc(x.X[q], y.X[q]).real()*2.0;
		if (!x.pInfo->commensurate && !x.pInfo->oneSidedResponse) if(x.X[q+nStates] && y.X[q+nStates]) result += dotc(x.X[q+nStates], y.X[q+nStates]).real()*2.0;
	}

	if(x.pInfo->oneSidedResponse) result*=2.;
	mpiWorld->allReduce(result, MPIUtil::ReduceSum);
	return result;
}

double dot(const std::vector<ColumnBundle>& x, const std::vector<ColumnBundle>& y, PerturbationInfo* pInfo, ElecInfo *eInfo)
{
	int nStates = eInfo->nStates;
	double result = 0.0;

	for(int q=eInfo->qStart; q<eInfo->qStop; q++)
	{
		if(x[q] && y[q]) result += dotc(x[q], y[q]).real()*2.0;
		if (!pInfo->commensurate && !pInfo->oneSidedResponse) if(x[q+nStates] && y[q+nStates]) result += dotc(x[q+nStates], y[q+nStates]).real()*2.0;
	}

	if(pInfo->oneSidedResponse) result*=2.;
	mpiWorld->allReduce(result, MPIUtil::ReduceSum);
	return result;
}

PerturbationGradient clone(const PerturbationGradient& x) { return x; } //!< create a copy

void randomize(PerturbationGradient& x) { randomize(x.X, *x.eInfo); } //!< initialize with random numbers

ScalarField randomRealSpaceVector(ColumnBundle& ref, const GridInfo* gInfo) {
	ColumnBundle res = ref;
	res.randomize(0, 1);
	ScalarField vec = Real(I(res.getColumn(0, 0)));
	ScalarFieldTilde vtilde = J(vec);
	vtilde->setGzero(0);
	return changeGrid(I(vtilde), *gInfo);
}


PerturbationSolver::PerturbationSolver(Everything& e) : e(e), eVars(e.eVars), eInfo(e.eInfo), iInfo(e.iInfo), pInfo(e.pertInfo)
{
}

void PerturbationSolver::setResponseSymmetries(const std::vector<SpaceGroupOp>& operations)
{ responseSymmetries=operations; }

double PerturbationSolver::responseWeight(int k) const
{ return kReductionActive ? responseWeights[k] : eInfo.qnums[k].weight; }

bool PerturbationSolver::pInfoInKBatch(int k) const { return pInfo.inPhononKBatch(k); }

int PerturbationSolver::responseSigns() const
{ return (pInfo.commensurate || pInfo.oneSidedResponse) ? 1 : 2; }

void PerturbationSolver::prepareTimeReversal()
{
	if(timeReversalChecked || !pInfo.fullMeshResponse || !pInfo.optimizeFrozenResponse) return;
	timeReversalChecked=true;
	pInfo.oneSidedResponse=false;
	const char* setting=getenv("JDFTX_KI_TIME_REVERSAL");
	if(setting && string(setting)!="0" && string(setting)!="1")
		die("JDFTX_KI_TIME_REVERSAL must be 0 or 1.\n");
	if((setting && string(setting)=="0") || metallic() || pInfo.commensurate
		|| eInfo.spinType!=SpinNone) return;
	// Only the fully occupied source manifold can be changed by an arbitrary
	// sewing rotation without rotating a nontrivial occupation matrix.
	for(int k=0;k<eInfo.nStates;k++) for(double f:eVars.F[k]) if(fabs(f-1.)>1e-12) return;
	std::vector<int> opposite(eInfo.nStates,-1);
	for(int k=0;k<eInfo.nStates;k++) for(int j=0;j<eInfo.nStates;j++) {
		vector3<> delta=eInfo.qnums[k].k+eInfo.qnums[j].k;
		for(int d=0;d<3;d++) delta[d]-=round(delta[d]);
		if(delta.length_squared()<1e-20) { opposite[k]=j; break; }
	}
	for(int j:opposite) if(j<0) return;
	const SpaceGroupOp identity;
	double error=0.;
	for(int k=0;k<eInfo.nStates;k++) {
		ColumnBundle remote;
		const int owner=eInfo.whose(k);
		ColumnBundle& source=eInfo.isMine(k) ? eVars.C[k] : remote;
		if(!eInfo.isMine(k)) source.init(eInfo.nBands,e.basis[k].nbasis,&e.basis[k],&eInfo.qnums[k],isGpuEnabled());
		if(mpiWorld->nProcesses()>1) mpiWorld->bcastData(source,owner);
		const int j=opposite[k];
		if(!eInfo.isMine(j)) continue;
		const ColumnBundle& target=eVars.C[j];
		ColumnBundle transformed=target.similar(); transformed.zero();
		ColumnBundleTransform::BasisWrapper wrapper(*target.basis);
		ColumnBundleTransform(source.qnum->k,*source.basis,target.qnum->k,wrapper,1,identity,-1)
			.scatterAxpy(1.,source,transformed,0,1);
		// The common-basis reconstruction need not preserve exact orthonormality.
		// The raw overlap then includes the Gram-matrix error, even for two
		// exactly time-reversed states. Its polar factor is the best unitary
		// sewing rotation and preserves the actual occupied projector and Hsub.
		matrix left,right; diagMatrix singular;
		(transformed^O(target)).svd(left,singular,right);
		const matrix sewing=left*right;
		ColumnBundle difference=target;
		difference-=transformed*sewing;
		double norm=dotc(target,target).real();
		error=std::max(error,sqrt(std::max(0.,dotc(difference,difference).real())/std::max(1e-30,norm)));
		matrix hError=eVars.Hsub[j]-dagger(sewing)*conj(eVars.Hsub[k])*sewing;
		error=std::max(error,nrm2(hError)/std::max(1.,nrm2(eVars.Hsub[j])));
	}
	mpiWorld->allReduce(error,MPIUtil::ReduceMax);
	// A tolerance tied to the requested response accuracy rejects ground states
	// that would make the reduction less accurate than the original solve.
	const double tolerance=std::min(1e-8,0.05*pInfo.solverParams.residualTol);
	pInfo.oneSidedResponse=error<=tolerance;
	logPrintf("KI time-reversal response: %s (occupied projector/Hamiltonian error %.3g, tolerance %.3g).\n",
		pInfo.oneSidedResponse ? "one branch" : "two branches",error,tolerance);
	if(pInfo.oneSidedResponse) for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
		pInfo.Cinc[k+eInfo.nStates].free(); pInfo.dC[k+eInfo.nStates].free();
	}
}

void PerturbationSolver::prepareKReduction()
{
	kReductionActive=false;
	physicalResponse=false;
	kSymmetries.clear(); kCharacters.clear(); densitySymmetryMaps.clear();
	if(!pInfo.phononReduceK || !pInfo.phononGrid.isNonzero() || pInfo.commensurate
		|| !pInfo.datom || pInfo.testing) return;
	const auto& mode=pInfo.datom->mode;
	const vector3<>& position=iInfo.species[mode.sp]->atpos[mode.at];
	const auto& g=e.gInfo;
	for(const auto& op:responseSymmetries) {
		// Use one-dimensional characters of a Cartesian displacement.
		// Translations and reciprocal q wraps need additional Bloch factors;
		// general Cartesian mixing needs multi-component irreps.
		if(op.a.length_squared()>1e-20 || (pInfo.qvec*op.inv().rot-pInfo.qvec).length_squared()>1e-20
			|| (op.rot*position-position).length_squared()>1e-20) continue;
		vector3<> rotated=g.R*op.rot*g.invR*mode.dirCartesian;
		double character;
		if((rotated-mode.dirCartesian).length_squared()<1e-20) character=1.;
		else if((rotated+mode.dirCartesian).length_squared()<1e-20) character=-1.;
		else continue;
		kSymmetries.push_back(op); kCharacters.push_back(character);
	}
	if(kSymmetries.size()<2) return;
	auto matches=[](vector3<> a,const vector3<>& b) {
		a-=b; for(int d=0;d<3;d++) a[d]-=round(a[d]);
		return a.length_squared()<1e-16;
	};
	std::vector<std::vector<int>> maps(kSymmetries.size(),std::vector<int>(eInfo.nStates,-1));
	for(size_t s=0;s<kSymmetries.size();s++) for(int k=0;k<eInfo.nStates;k++) {
		vector3<> target=eInfo.qnums[k].k*kSymmetries[s].inv().rot;
		for(int j=0;j<eInfo.nStates;j++) if(matches(target,eInfo.qnums[j].k)) { maps[s][k]=j; break; }
		if(maps[s][k]<0) die("DFPT response symmetry does not preserve the k mesh.\n");
	}
	kRepresentatives.assign(eInfo.nStates,-1); kOperations.assign(eInfo.nStates,-1);
	responseWeights.assign(eInfo.nStates,0.);
	std::vector<int> load(mpiWorld->nProcesses(),0);
	int reduced=0;
	for(int seed=0;seed<eInfo.nStates;seed++) {
		if(kRepresentatives[seed]>=0) continue;
		std::vector<int> orbit;
		for(const auto& map:maps) orbit.push_back(map[seed]);
		std::sort(orbit.begin(),orbit.end()); orbit.erase(std::unique(orbit.begin(),orbit.end()),orbit.end());
		int representative=orbit[0];
		for(int k:orbit) if(load[eInfo.whose(k)]<load[eInfo.whose(representative)]) representative=k;
		load[eInfo.whose(representative)]++; reduced++;
		for(int k:orbit) {
			if(kRepresentatives[k]>=0) die("DFPT response k orbits overlap.\n");
			kRepresentatives[k]=representative;
			responseWeights[representative]+=eInfo.qnums[k].weight;
			for(size_t s=0;s<maps.size();s++) if(maps[s][representative]==k) { kOperations[k]=s; break; }
			if(kOperations[k]<0) die("DFPT response k orbit is not closed.\n");
		}
	}
	if(reduced==eInfo.nStates) return;
	// The driver already checked that these space-group operations exactly
	// permute the FFT mesh and preserve the frozen Hamiltonian.
	for(const auto& op:kSymmetries) {
		matrix3<int> rotation;
		for(int i=0;i<3;i++) for(int j=0;j<3;j++) rotation(i,j)=int(round(double(op.rot(i,j))*g.S[i]/g.S[j]));
		std::vector<int> map(g.nr);
		vector3<int> x;
		for(x[0]=0;x[0]<g.S[0];x[0]++) for(x[1]=0;x[1]<g.S[1];x[1]++) for(x[2]=0;x[2]<g.S[2];x[2]++) {
			vector3<int> y=rotation*x;
			for(int d=0;d<3;d++) y[d]=(y[d]%g.S[d]+g.S[d])%g.S[d];
			map[(x[0]*g.S[1]+x[1])*g.S[2]+x[2]]=(y[0]*g.S[1]+y[1])*g.S[2]+y[2];
		}
		densitySymmetryMaps.push_back(std::move(map));
	}
	kReductionActive=true;
	logPrintf("DFPT electronic response reduction: %d/%d k points, %zu displacement symmetries.\n",reduced,eInfo.nStates,kSymmetries.size());
}

void PerturbationSolver::symmetrizeResponseDensity(complexScalarFieldArray& density) const
{
	if(!kReductionActive) return;
	for(auto& field:density) {
		std::vector<complex> original(field->data(),field->data()+e.gInfo.nr);
		complex* result=field->data();
		for(int x=0;x<e.gInfo.nr;x++) {
			complex sum=0.;
			for(size_t s=0;s<densitySymmetryMaps.size();s++) sum+=kCharacters[s]*original[densitySymmetryMaps[s][x]];
			result[x]=sum/double(densitySymmetryMaps.size());
		}
	}
}

void PerturbationSolver::expandKResponse()
{
	if(!kReductionActive) return;
	// Force contractions need every Cartesian source on the full mesh. The
	// sewing matrix rotates source-band columns to each target's eigenbasis;
	// a spatial rotation alone would be wrong at degeneracies / arbitrary gauges.
	if(metallic()) metallicProject(pInfo.dC);
	for(int representative=0;representative<eInfo.nStates;representative++) {
		if(kRepresentatives[representative]!=representative) continue;
		ColumnBundle source(eInfo.nBands,e.basis[representative].nbasis,&e.basis[representative],&eInfo.qnums[representative],isGpuEnabled());
		if(eInfo.isMine(representative)) source=eVars.C[representative];
		if(mpiWorld->nProcesses()>1) mpiWorld->bcastData(source,eInfo.whose(representative));
		std::vector<ColumnBundle> shifts(2);
		for(int sign=0;sign<2;sign++) {
			int index=representative+sign*eInfo.nStates;
			const Basis& basis=sign ? pInfo.kminusq_basis[representative] : pInfo.kplusq_basis[representative];
			const QuantumNumber& qnum=sign ? pInfo.kminusq_vectors[representative] : pInfo.kplusq_vectors[representative];
			shifts[sign].init(eInfo.nBands,basis.nbasis,&basis,&qnum,isGpuEnabled());
			if(eInfo.isMine(representative)) shifts[sign]=metallic()
				? metallicMap(pInfo.dC[index],representative,sign) : pInfo.dC[index];
			if(mpiWorld->nProcesses()>1) mpiWorld->bcastData(shifts[sign],eInfo.whose(representative));
			if(metallic() && eInfo.isMine(representative)) pInfo.dC[index]=shifts[sign];
		}
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
			if(kRepresentatives[k]!=representative || k==representative) continue;
			int s=kOperations[k];
			SpaceGroupOp operation=kSymmetries[s].inv();
			ColumnBundle rotated=eVars.C[k].similar(); rotated.zero();
			ColumnBundleTransform::BasisWrapper wrapper(e.basis[k]);
			ColumnBundleTransform(eInfo.qnums[representative].k,e.basis[representative],eInfo.qnums[k].k,wrapper,1,operation,1)
				.scatterAxpy(1.,source,rotated,0,1);
			matrix sewing=rotated^O(eVars.C[k]);
			for(int sign=0;sign<2;sign++) {
				int index=k+sign*eInfo.nStates;
				ColumnBundle target=pInfo.Cinc[index].similar(); target.zero();
				ColumnBundleTransform::BasisWrapper offsetWrapper(*target.basis);
				ColumnBundleTransform(shifts[sign].qnum->k,*shifts[sign].basis,target.qnum->k,offsetWrapper,1,operation,1)
					.scatterAxpy(kCharacters[s],shifts[sign],target,0,1);
				target=target*sewing;
				pInfo.dC[index]=std::move(target);
			}
		}
	}
	kReductionActive=false;
	physicalResponse=metallic();
}

complex PerturbationSolver::forceResponse()
{
	// Metallic variational amplitudes depend on the stored empty-band
	// projector. Contract expanded physical responses directly; inverting
	// their susceptibility map would amplify roundoff in empty bands.
	if(!physicalResponse) calcdGradTau();
	complex response=0.;
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) for(int sign=0;sign<responseSigns();sign++) {
		int index=k+sign*eInfo.nStates;
		complex value;
		if(physicalResponse) {
			ColumnBundle derivative=pInfo.Cinc[index].similar(); derivative.zero();
			dHtau(*derivative.qnum,derivative,eVars.C[k],sign ? -1 : 1);
			value=eInfo.qnums[k].weight*dotc(derivative,pInfo.dC[index]);
		} else value=dotc(sourceGradient().X[index],pInfo.dC[index]);
		response+=sign ? conj(value) : value;
	}
	if(pInfo.oneSidedResponse) response*=2.;
	mpiWorld->allReduce(response,MPIUtil::ReduceSum);
	return response;
}

PerturbationGradient PerturbationSolver::screeningSource(const complexScalarFieldArray& potential)
{
	PerturbationGradient result; result.init(e);
	complexScalarFieldArray negative=conj(potential);
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<responseSigns();sign++) {
		int index=k+sign*eInfo.nStates;
		ColumnBundle derivative=pInfo.Cinc[index].similar(); derivative.zero();
		dHpsi(*derivative.qnum,derivative,eVars.C[k],sign ? negative : potential);
		if(metallic()) result.X[index]=metallicMap(derivative,k,sign);
		else {
			const ColumnBundle& C=pInfo.Cinc[index];
			result.X[index]=(derivative-O(C*(C^derivative)))*eVars.F[k];
		}
	}
	if(metallic()) metallicProject(result.X);
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<responseSigns();sign++)
		result.X[k+sign*eInfo.nStates]*=responseWeight(k);
	return result;
}

ColumnBundle PerturbationSolver::bareBandOperator(const ColumnBundle& X,int k,const std::vector<int>& bands)
{
	int sign=(X.qnum==&pInfo.kminusq_vectors[k]) ? 1 : 0;
	const ColumnBundle& C=pInfo.Cinc[k+sign*eInfo.nStates];
	ColumnBundle low=C*(C^O(X)), high=X-low, result;
	applyH(*X.qnum,eye(X.nCols()),result,high);
	diagMatrix energies(bands.size());
	for(size_t i=0;i<bands.size();i++) energies[i]=eVars.Hsub_eigs[k][bands[i]];
	result-=O(high)*energies;
	result-=C*(C^O(result));
	if(metallic()) result+=O(low);
	return result;
}

ColumnBundle PerturbationSolver::bareBandPrecondition(const ColumnBundle& X,int k,int sign,const std::vector<int>* bands)
{
	const ColumnBundle& C=pInfo.Cinc[k+sign*eInfo.nStates];
	ColumnBundle low=C*(C^O(X)), result=X-low;
	const int index=k+sign*eInfo.nStates;
	const ColumnBundle* coarse=bandCoarseSpace.empty() ? nullptr : &bandCoarseSpace[index];
	matrix coefficients;
	if(coarse && *coarse) {
		coefficients=*coarse^O(result);
		result-=*coarse*coefficients;
	}
	precond_inv_kinetic(result,1.);
	if(coarse && *coarse) {
		result-=*coarse*(*coarse^O(result));
		const matrix& inverse=bandCoarseInverse[index];
		if(!bands || X.nCols()==eInfo.nBands)
			callPref(eblas_zmul)(coefficients.nData(),inverse.dataPref(),1,coefficients.dataPref(),1);
		else for(int col=0;col<X.nCols();col++)
			callPref(eblas_zmul)(coarse->nCols(),inverse.dataPref()+size_t((*bands)[col])*coarse->nCols(),1,
				coefficients.dataPref()+size_t(col)*coarse->nCols(),1);
		result+=*coarse*coefficients;
	}
	result-=C*(C^O(result));
	if(metallic()) result+=low;
	return result;
}

void PerturbationSolver::clearBandCoarseSpace()
{ bandCoarseSpace.clear(); bandCoarseInverse.clear(); }

void PerturbationSolver::setBandCoarseSpace(std::vector<ColumnBundle>&& space)
{
	clearBandCoarseSpace();
	if(!pInfo.fullMeshResponse || !pInfo.optimizeFrozenResponse || !pInfo.nestedResponse
		|| !pInfo.independentBandCG || metallic()) return;
	bandCoarseSpace=std::move(space); bandCoarseInverse.resize(bandCoarseSpace.size());
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) for(int sign=0;sign<responseSigns();sign++) {
		int index=k+sign*eInfo.nStates;
		ColumnBundle& U=bandCoarseSpace[index];
		if(!U) continue;
		const ColumnBundle& C=pInfo.Cinc[index];
		for(int pass=0;pass<2;pass++) U-=C*(C^O(U));
		bool singular=false;
		matrix normalization=invsqrt(dagger_symmetrize(U^O(U)),nullptr,nullptr,&singular);
		if(singular) { U.free(); continue; }
		U=U*normalization;
		ColumnBundle HU=U.similar();
		applyH(*U.qnum,eye(U.nCols()),HU,U);
		matrix vectors; diagMatrix energies;
		dagger_symmetrize(U^HU).diagonalize(vectors,energies);
		U=U*vectors;
		// Cache positive per-source-band spectral denominators once per q.
		// Applying them requires no device-to-host transfer, also after pruning
		// converged bands. The full Sternheimer operator is unchanged.
		matrix inverse(energies.size(),eInfo.nBands);
		complex* values=inverse.data();
		// Coefficients above include O=detR for NC states, while the bare
		// Hessian is detR*(H-epsilon). Match the kinetic inverse's volume scale.
		const double volume=U.basis->gInfo->detR;
		for(int col=0;col<eInfo.nBands;col++) for(size_t row=0;row<energies.size();row++)
			values[inverse.index(row,col)]=1./(volume*std::max(1e-3,energies[row]-eVars.Hsub_eigs[k][col]));
		bandCoarseInverse[index]=std::move(inverse);
	}
}

namespace {
double responseFieldDot(const complexScalarFieldArray& a,const complexScalarFieldArray& b)
{
	double result=0.;
	for(size_t s=0;s<a.size();s++) result+=dot(a[s],b[s]).real();
	return result;
}

size_t responseRecycleBudget(const char* name,double defaultMiB)
{
	const char* value=getenv(name);
	char* end=nullptr;
	double mib=value ? strtod(value,&end) : defaultMiB;
	if(!std::isfinite(mib) || mib<0. || mib>65536. || (value && (end==value || *end)))
		die("%s must be a finite number between 0 and 65536 MiB.\n",name);
	return size_t(mib*1024.*1024.);
}
}

void PerturbationSolver::prepareResponseRecycling()
{
	if(!pInfo.fullMeshResponse || !pInfo.optimizeFrozenResponse || !pInfo.nestedResponse
		|| pInfo.commensurate || metallic()) return;
	if(pInfo.oneSidedResponse && !bandCoarseSpace.empty())
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
			bandCoarseSpace[k+eInfo.nStates].free(); bandCoarseInverse[k+eInfo.nStates]=matrix();
		}
	std::vector<size_t> generations={size_t(eInfo.nStates),size_t(eInfo.nBands),size_t(responseSigns())};
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) for(int sign=0;sign<responseSigns();sign++)
		generations.push_back(pInfo.Cinc[k+sign*eInfo.nStates].basis->generation());
	if(!recycleReady || !(recycleQ==pInfo.qvec) || generations!=recycleGenerations) {
		for(auto& entries:dielectricRecycle) entries.clear();
		recycleQ=pInfo.qvec; recycleGenerations=std::move(generations);
		if(!recycleReady) {
			dielectricRecycleLimit=responseRecycleBudget("JDFTX_KI_DIELECTRIC_RECYCLE_MIB",16.);
			logPrintf("KI fixed-q dielectric recycling: %.1f MiB/rank.\n",
				dielectricRecycleLimit/(1024.*1024.));
		}
		recycleReady=true;
	}
}

bool PerturbationSolver::predictScreeningPotential(complexScalarFieldArray& potential)
{
	const auto& entries=dielectricRecycle[pInfo.magneticResponse ? 1 : 0];
	if(!recycleReady || entries.empty() || !pInfo.dVext) return false;
	// External fields are physical potentials; induced fields and outer errors
	// use the quadrature-weighted JdagOJ representation consumed by dHpsi.
	complexScalarFieldArray source(pInfo.dVext->dVextpq.size());
	for(size_t s=0;s<source.size();s++) source[s]=JdagOJ(pInfo.dVext->dVextpq[s]);
	double norm=responseFieldDot(source,source), captured=0.;
	for(const auto& entry:entries) {
		double coefficient=responseFieldDot(entry.source,source);
		captured+=coefficient*coefficient;
		axpy(coefficient,entry.response,potential);
	}
	double predictedNorm=responseFieldDot(potential,potential);
	if(!std::isfinite(predictedNorm) || predictedNorm>1e4*norm || captured<1e-4*norm) {
		initZero(potential); return false;
	}
	if(!pInfo.fullMeshResponse || pInfo.kiVerbose) logPrintf("KI recycled screening potential: %d directions, %.1f%% source norm captured (%s sector).\n",
		int(entries.size()),100.*std::min(1.,captured/std::max(1e-300,norm)),pInfo.magneticResponse ? "magnetic" : "charge");
	return true;
}

void PerturbationSolver::preconditionScreeningError(complexScalarFieldArray& error)
{
	const auto& entries=dielectricRecycle[pInfo.magneticResponse ? 1 : 0];
	if(entries.empty()) return;
	// If L maps an external source to its converged induced potential, the
	// inverse fixed-point Jacobian is I+L. Reuse the measured part of L for
	// outer mixing as well as the starting potential, without new inner solves.
	double norm=responseFieldDot(error,error);
	auto corrected=clone(error);
	for(const auto& entry:entries)
		axpy(responseFieldDot(entry.source,error),entry.response,corrected);
	double correctedNorm=responseFieldDot(corrected,corrected);
	if(std::isfinite(correctedNorm) && correctedNorm<=1e4*norm) error=std::move(corrected);
}

void PerturbationSolver::recordScreeningResponse()
{
	if(!recycleReady || !dielectricRecycleLimit || !pInfo.dVext
		|| !std::isfinite(pInfo.lastResponseResidual) || pInfo.lastResponseResidual>pInfo.solverParams.residualTol) return;
	size_t bytes=2*pInfo.dVext->dVextpq.size()*size_t(e.gInfo.nr)*sizeof(complex);
	auto& entries=dielectricRecycle[pInfo.magneticResponse ? 1 : 0];
	size_t total=dielectricRecycle[0].size()+dielectricRecycle[1].size();
	if(entries.size()>=size_t(std::max(1,pInfo.phononMixHistory))
		|| bytes>dielectricRecycleLimit/(total+1)) return;
	DielectricRecycle entry={complexScalarFieldArray(pInfo.dVext->dVextpq.size()),clone(pInfo.dVsclocpq)};
	for(size_t s=0;s<entry.source.size();s++) entry.source[s]=JdagOJ(pInfo.dVext->dVextpq[s]);
	double norm=responseFieldDot(entry.source,entry.source);
	// The converged dielectric map is linear at fixed q. Orthogonalize both
	// sides together, rejecting dependent sources rather than magnifying errors.
	for(int pass=0;pass<2;pass++) for(const auto& previous:entries) {
		double coefficient=responseFieldDot(previous.source,entry.source);
		axpy(-coefficient,previous.source,entry.source);
		axpy(-coefficient,previous.response,entry.response);
	}
	double remainder=responseFieldDot(entry.source,entry.source);
	if(!std::isfinite(remainder) || norm<=1e-30 || remainder<1e-6*norm) return;
	double scale=1./sqrt(remainder);
	entry.source*=scale; entry.response*=scale;
	entries.push_back(std::move(entry));
}

bool PerturbationSolver::solveIndependentBands(const PerturbationGradient& rhs,const SolverParams& parameters)
{
	// Independent columns require a diagonal occupied Hamiltonian in the
	// supplied ground-state gauge. A general occupied rotation uses MINRES.
	if(pInfo.fullMeshResponse) {
		if(!bandDiagonalChecked) {
			bandDiagonalReady=true;
			for(int k=eInfo.qStart;k<eInfo.qStop;k++)
				if(nrm2(eVars.Hsub[k]-matrix(eVars.Hsub_eigs[k]))>1e-9)
					bandDiagonalReady=false;
			mpiWorld->allReduce(bandDiagonalReady,MPIUtil::ReduceLAnd);
			bandDiagonalChecked=true;
			if(!bandDiagonalReady) logPrintf("KI occupied Hamiltonian is not diagonal; nested inner solves use MINRES.\n");
		}
		if(!bandDiagonalReady) return false;
	}
	// Each MPI rank owns complete plane-wave vectors for its k points. With
	// screening fixed, these blocks require no inter-k density reductions.
	// Pack only unconverged source bands into subsequent Hamiltonian calls.
	struct Block {
		int k,sign;
		std::vector<int> bands;
		ColumnBundle solution,residual,direction;
		diagMatrix rho,reference;
	};
	std::vector<Block> blocks;
	double referenceNorm=0.; int totalBands=0;
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<responseSigns();sign++) {
		int index=k+sign*eInfo.nStates;
		Block block; block.k=k; block.sign=sign;
		block.residual=rhs.X[index]; block.residual*=(-1./responseWeight(k));
		block.solution=block.residual.similar(); block.solution.zero();
		block.direction=bareBandPrecondition(block.residual,k,sign);
		totalBands+=eInfo.nBands;
		for(int band=0;band<eInfo.nBands;band++) block.bands.push_back(band);
		blocks.push_back(std::move(block));
	}
	std::vector<const ColumnBundle*> residuals,directions;
	for(const auto& block:blocks) { residuals.push_back(&block.residual); directions.push_back(&block.direction); }
	auto initialRho=batchedDiagDot(residuals,directions);
	for(size_t b=0;b<blocks.size();b++) {
		blocks[b].rho=std::move(initialRho[b]); blocks[b].reference=blocks[b].rho;
		for(double rho:blocks[b].rho) referenceNorm+=std::max(0.,rho)*std::pow(responseWeight(blocks[b].k),2);
	}
	mpiWorld->allReduce(referenceNorm,MPIUtil::ReduceSum);
	mpiWorld->allReduce(totalBands,MPIUtil::ReduceSum);
	// Use a common absolute residual budget, as in QE. Relative tests against
	// each tiny empty-band source would oversolve bands with negligible weight.
	for(auto& block:blocks) for(double& reference:block.reference)
		reference=referenceNorm/(std::max(1,totalBands)*std::pow(responseWeight(block.k),2));
	auto select=[](const ColumnBundle& X,const std::vector<int>& indices) {
		ColumnBundle result=X.similar(indices.size());
		for(size_t i=0;i<indices.size();i++) result.copyColumnFrom(i,X,indices[i]);
		return result;
	};
	int applications=0;
	for(int iteration=0;iteration<parameters.nIterations;iteration++) {
		bool failed=false,unfinished=false;
		std::vector<Block*> active;
		for(auto& block:blocks) {
			if(block.bands.empty()) continue;
			std::vector<int> retain;
			for(size_t i=0;i<block.bands.size();i++) {
				if(block.rho[i]<=parameters.residualTol*parameters.residualTol*block.reference[i] || block.reference[i]<=1e-30)
					state.X[block.k+block.sign*eInfo.nStates].copyColumnFrom(block.bands[i],block.solution,i);
				else retain.push_back(i);
			}
			if(retain.size()!=block.bands.size()) {
				std::vector<int> bands; diagMatrix rho(retain.size()), reference(retain.size());
				for(size_t i=0;i<retain.size();i++) { bands.push_back(block.bands[retain[i]]); rho[i]=block.rho[retain[i]]; reference[i]=block.reference[retain[i]]; }
				block.bands=std::move(bands); block.rho=std::move(rho); block.reference=std::move(reference);
				if(block.bands.empty()) continue;
				block.solution=select(block.solution,retain); block.residual=select(block.residual,retain); block.direction=select(block.direction,retain);
			}
			active.push_back(&block); unfinished=true;
		}
		mpiWorld->allReduce(unfinished,MPIUtil::ReduceLOr);
		if(!unfinished) {
			mpiWorld->allReduce(applications,MPIUtil::ReduceSum);
			if(!pInfo.fullMeshResponse || pInfo.kiVerbose) logPrintf("DFPT independent-band solve: %d iterations, %d band Hamiltonian applications.\n",iteration,applications);
			return true;
		}
		// Pack Hamiltonian FFTs and dot products across every local k/sign block.
		// Only two scalar-array transfers are required for a complete CG step.
		std::vector<ColumnBundle> high(active.size()),applied,preconditioned(active.size());
		std::vector<const ColumnBundle*> inputs;
		std::vector<const ColumnBundle*> projectors;
		std::vector<ColumnBundle*> highMutable;
		const bool batchProject=pInfo.fullMeshResponse && pInfo.optimizeFrozenResponse && !metallic();
		for(size_t b=0;b<active.size();b++) {
			const auto& block=*active[b];
			const ColumnBundle& C=pInfo.Cinc[block.k+block.sign*eInfo.nStates];
			high[b]=block.direction;
			if(batchProject) { projectors.push_back(&C); highMutable.push_back(&high[b]); }
			else high[b]-=C*(C^O(block.direction));
			inputs.push_back(&high[b]); applications+=block.bands.size();
		}
		if(batchProject) projectOutBatched(projectors,highMutable);
		applyHBatched(inputs,applied);
		std::vector<const ColumnBundle*> appliedPointers,directionPointers;
		std::vector<ColumnBundle*> solutionPointers,residualPointers,directionMutable;
		std::vector<ColumnBundle*> appliedMutable;
		for(size_t b=0;b<active.size();b++) {
			auto& block=*active[b];
			const ColumnBundle& C=pInfo.Cinc[block.k+block.sign*eInfo.nStates];
			diagMatrix energies(block.bands.size());
			for(size_t i=0;i<block.bands.size();i++) energies[i]=eVars.Hsub_eigs[block.k][block.bands[i]];
			applied[b]-=O(high[b])*energies;
			if(batchProject) appliedMutable.push_back(&applied[b]);
			else applied[b]-=C*(C^O(applied[b]));
			if(metallic()) applied[b]+=O(block.direction-high[b]);
			appliedPointers.push_back(&applied[b]); directionPointers.push_back(&block.direction);
			solutionPointers.push_back(&block.solution); residualPointers.push_back(&block.residual);
			directionMutable.push_back(&block.direction);
		}
		if(batchProject) projectOutBatched(projectors,appliedMutable);
		high.clear(); // Recycle projection work before allocating the preconditioned residuals.
		auto curvature=batchedDiagDot(directionPointers,appliedPointers);
		std::vector<diagMatrix> alpha; alpha.reserve(active.size());
		for(size_t b=0;b<active.size();b++) {
			const auto& block=*active[b]; alpha.emplace_back(block.bands.size());
			for(size_t i=0;i<block.bands.size();i++) {
				if(!std::isfinite(curvature[b][i]) || curvature[b][i]<=0. || !std::isfinite(block.rho[i]) || block.rho[i]<=0.) { failed=true; break; }
				alpha.back()[i]=block.rho[i]/curvature[b][i];
			}
		}
		mpiWorld->allReduce(failed,MPIUtil::ReduceLOr);
		if(failed) {
			state*=0.;
			logPrintf("DFPT independent-band CG encountered nonpositive curvature; using MINRES for this inner solve.\n");
			return false;
		}
		batchedCGStep(solutionPointers,residualPointers,directionPointers,appliedPointers,alpha);
		applied.clear();
		std::vector<const ColumnBundle*> residualConst,preconditionedPointers;
		for(size_t b=0;b<active.size();b++) {
			const auto& block=*active[b]; preconditioned[b]=bareBandPrecondition(block.residual,block.k,block.sign,&block.bands);
			residualConst.push_back(&block.residual); preconditionedPointers.push_back(&preconditioned[b]);
		}
		auto rho=batchedDiagDot(residualConst,preconditionedPointers);
		std::vector<diagMatrix> beta; beta.reserve(active.size());
		for(size_t b=0;b<active.size();b++) {
			auto& block=*active[b]; beta.emplace_back(rho[b].size());
			for(size_t i=0;i<rho[b].size();i++) beta.back()[i]=rho[b][i]/block.rho[i];
			block.rho=std::move(rho[b]);
		}
		batchedCGDirection(directionMutable,preconditionedPointers,beta);
	}
	state*=0.;
	logPrintf("DFPT independent-band solve reached its iteration limit; using MINRES for this inner solve.\n");
	return false;
}

void PerturbationSolver::solveNestedResponse()
{
	complexScalarFieldArray potential(pInfo.dVsclocpq.size());
	nullToZero(potential,e.gInfo); initZero(potential);
	const bool predicted=predictScreeningPotential(potential);
	std::vector<complexScalarFieldArray> inputs, outputs, errors;
	auto potentialDot=[](const complexScalarFieldArray& a,const complexScalarFieldArray& b) {
		double result=0.; for(size_t s=0;s<a.size();s++) result+=dot(a[s],b[s]).real(); return result;
	};
	const double sourceNorm=dot(sourceGradient(),sourceGradient());
	const char* reuseSetting=getenv("JDFTX_KI_REUSE_BARE_HESSIAN");
	if(reuseSetting && string(reuseSetting)!="0" && string(reuseSetting)!="1")
		die("JDFTX_KI_REUSE_BARE_HESSIAN must be 0 or 1.\n");
	const bool reuseBare=pInfo.fullMeshResponse && pInfo.optimizeFrozenResponse && !metallic()
		&& (!reuseSetting || string(reuseSetting)!="0");
	PerturbationGradient barePrevious;
	double relative=1.;
	for(int outer=0;outer<pInfo.phononMixIterations;outer++) {
		PerturbationGradient rhs=clone(sourceGradient());
		if(outer || predicted) axpy(1.,screeningSource(potential),rhs);
		// Solve a correction about the previous *physical* response. MINRES
		// stores preconditioned coordinates internally, so passing its previous
		// physical state directly as a warm start would apply Ksqrt twice.
		PerturbationGradient previous=std::move(state), correctionRhs;
		screeningEnabled=false;
		if(reuseBare) {
			if(outer) { correctionRhs=std::move(barePrevious); axpy(1.,rhs,correctionRhs); }
			else correctionRhs=clone(rhs); // The initial physical response is zero.
		} else { hessian(correctionRhs,previous); axpy(1.,rhs,correctionRhs); }
		state.init(e);
		state.X.resize(previous.X.size());
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) for(int sign=0;sign<responseSigns();sign++) {
			int index=k+sign*eInfo.nStates;
			if(previous.X[index]) {
				const ColumnBundle& reference=previous.X[index];
				state.X[index].init(reference.nCols(),reference.colLength(),reference.basis,reference.qnum,isGpuEnabled());
				state.X[index].zero();
			} else state.X[index].free();
		}
		SolverParams inner=pInfo.solverParams;
		inner.algorithm=SolverParams::MINRES;
		inner.residualDiffThreshold=0.;
		// This is a correction equation, not the original source equation. Use
		// an absolute error budget relative to the original source; otherwise
		// late corrections are solved far below floating-point accuracy.
		double correctionNorm=dot(correctionRhs,correctionRhs);
		double absoluteBudget=correctionNorm>0. ? 0.1*pInfo.solverParams.residualTol*sqrt(sourceNorm/correctionNorm) : 0.1;
		inner.residualTol=std::min(0.1,std::max(absoluteBudget,std::min(1e-2,0.1*relative)));
		if(!pInfo.independentBandCG || !solveIndependentBands(correctionRhs,inner)) solve(correctionRhs,inner);
		if(!metallic()) for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<responseSigns();sign++) {
			int index=k+sign*eInfo.nStates; const ColumnBundle& C=pInfo.Cinc[index];
			state.X[index]-=C*(C^O(state.X[index]));
		}
		axpy(1.,previous,state);
		previous.X.clear(); rhs.X.clear(); correctionRhs.X.clear();
		screeningEnabled=true;
		PerturbationGradient residual;
		if(reuseBare) { barePrevious.init(e); bareHessianCapture=&barePrevious; }
		hessian(residual,state); axpy(1.,sourceGradient(),residual);
		bareHessianCapture=nullptr;
		double norm=dot(residual,residual);
		relative=sourceNorm ? sqrt(norm/sourceNorm) : sqrt(norm);
		if(!pInfo.fullMeshResponse || pInfo.kiVerbose) logPrintf("DFPT nested iteration %d: coupled relative residual %.6le, inner tolerance %.6le.\n",outer+1,relative,inner.residualTol);
		if(!std::isfinite(relative)) die("Nonfinite nested DFPT response residual.\n");
		if(relative<=pInfo.solverParams.residualTol) return;
		complexScalarFieldArray output=clone(pInfo.dVsclocpq), error=output-potential;
		preconditionScreeningError(error);
		output=potential+error;
		inputs.push_back(clone(potential)); outputs.push_back(std::move(output)); errors.push_back(std::move(error));
		if(int(errors.size())>pInfo.phononMixHistory) {
			inputs.erase(inputs.begin()); outputs.erase(outputs.begin()); errors.erase(errors.begin());
		}
		int count=errors.size();
		matrix gram=zeroes(count,count), ones=zeroes(count,1);
		double scale=0.;
		for(int i=0;i<count;i++) {
			ones.set(i,0,1.);
			for(int j=0;j<=i;j++) { double value=potentialDot(errors[i],errors[j]); gram.set(i,j,value); gram.set(j,i,value); }
			scale=std::max(scale,gram(i,i).real());
		}
		for(int i=0;i<count;i++) gram.set(i,i,gram(i,i)+std::max(1e-30,1e-10*scale));
		matrix coefficients=invApply(gram,ones);
		double total=0.; for(int i=0;i<count;i++) total+=coefficients(i,0).real();
		bool valid=std::isfinite(total) && total>0.;
		for(int i=0;i<count;i++) valid &= std::isfinite(coefficients(i,0).real()) && fabs(coefficients(i,0).real()/total)<20.;
		initZero(potential);
		if(valid) for(int i=0;i<count;i++) {
			double coefficient=coefficients(i,0).real()/total;
			axpy(coefficient*(1.-pInfo.phononMixFraction),inputs[i],potential);
			axpy(coefficient*pInfo.phononMixFraction,outputs[i],potential);
		} else {
			axpy(1.-pInfo.phononMixFraction,inputs.back(),potential);
			axpy(pInfo.phononMixFraction,outputs.back(),potential);
			inputs.clear(); outputs.clear(); errors.clear();
		}
	}
	if(pInfo.fullMeshResponse) {
		logPrintf("KI nested screening reached its outer iteration limit; falling back to coupled MINRES.\n");
		state*=0.; solve(sourceGradient(),pInfo.solverParams);
		return;
	}
	die("Nested DFPT response did not converge; increase phonon-dfpt-mixing iterations or use the coupled solver.\n");
}

void PerturbationSolver::solvePerturbation()
{
	if (pInfo.testing)
	{	TestPerturbation(e, *this).testVPT();
		return;
	}

	if (!pInfo.solverParams.nIterations)
		die("Error: Must specify nIterations in command perturb-minimize to enable the variational perturbation solver.\n")
	
	if(!pInfo.fullMeshResponse || pInfo.kiVerbose) logPrintf("Variational perturbation solver is starting.\n");
	
	if (pInfo.datom) {
		AtomicMode m = pInfo.datom->mode;
		if (m.sp >= iInfo.species.size() || m.at >= iInfo.species[m.sp]->atpos.size())
			die("Perturbed atom does not exist! Check species and atom number.\n");
		logPrintf("Perturbing species %d atom %d in direction %g %g %g.\n", m.sp, m.at, m.dirCartesian[0], m.dirCartesian[1], m.dirCartesian[2]);
	}
	
	if (pInfo.dVext && (!pInfo.fullMeshResponse || pInfo.kiVerbose))
		logPrintf("Perturbing Vexternal.\n");
	
	if(!pInfo.groundStateReady && e.eVars.wfnsFilename.empty() && !e.eVars.skipWfnsInit) {
		logPrintf("No input wavefunctions given. Performing electronic structure calculation first.\n");
		elecFluidMinimize(e);
	}
	
	// Phonon and full-mesh orbital drivers supply the frozen ground-state
	// density, potential and subspace Hamiltonians; responses modify dC/Cinc.
	if((!pInfo.phononGrid.isNonzero() && !pInfo.fullMeshResponse) || !pInfo.groundStateReady)
		eVars.elecEnergyAndGrad(e.ener, nullptr, nullptr, true);
	pInfo.groundStateReady = true;
	if(!pInfo.commensurate && pInfo.wfnsFilename.empty() && !pInfo.offsetStatesReady)
	{ computeIncommensurateWfns(); pInfo.offsetStatesReady=true; }
	if(metallic() && !metallicReady) setupMetallicResponse();
	prepareTimeReversal();
	prepareKReduction();
	prepareResponseRecycling();

	state.X.clear();
	state.init(e);
	if (pInfo.commensurate) {
		init(state.X, eInfo.nStates, eInfo.nBands, &e.basis[0], &eInfo);
		for(int q=eInfo.qStart; q<eInfo.qStop; q++)
			state.X[q].zero();
	} else {
		pInfo.initInc(state.X, 2*eInfo.nStates, eInfo.nBands, &eInfo);
		for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
			if(!activeResponse(q)) { state.X[q].free(); state.X[q+eInfo.nStates].free(); continue; }
			state.X[q].zero();
			if(!pInfo.oneSidedResponse) state.X[q+eInfo.nStates].zero();
		}
	}
	
	if(!pInfo.phononGrid.isNonzero() && !pInfo.fullMeshResponse) eVars.elecEnergyAndGrad(e.ener, nullptr, nullptr, true);

	const bool frozenCache = pInfo.fullMeshResponse && pInfo.optimizeFrozenResponse && pInfo.groundStateReady;
	if((!usePhononCache() && !frozenCache) || !groundCachesReady) {
		updateExcorrCache();
		updateHC();
		groundCachesReady=true;
	}
	updateNonlocalDerivs();
	prepareFrozenOrbitals();
	
	pInfo.dGradTau.init(e);
	pInfo.dGradPsi.init(e);
	
	calcdGradTau();
	
	SolverParams responseParams=pInfo.solverParams;
	// The band-space and plane-wave blocks have different energy scales;
	// leave margin between the preconditioned and verified Euclidean residual.
	if(metallic()) responseParams.residualTol*=0.05;
	if((pInfo.phononNestedSolver && pInfo.phononGrid.isNonzero()) || pInfo.nestedResponse) solveNestedResponse();
	else solve(sourceGradient(), responseParams);
	
	hessian(pInfo.dGradPsi, state);
	if(metallic() && pInfo.qvec.length_squared()<1e-20) {
		complex charge(integral(Real(pInfo.dnpq[0])),integral(Imag(pInfo.dnpq[0])));
		complex shift=0.;
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
			if(!activeResponse(k)) continue;
			ColumnBundle bare=pInfo.Cinc[k].similar(), induced=bare.similar();
			bare.zero(); induced.zero();
			dHtau(*pInfo.Cinc[k].qnum,bare,eVars.C[k],1);
			dHpsi(*pInfo.Cinc[k].qnum,induced,eVars.C[k],pInfo.dVsclocpq);
			matrix dH=eVars.C[k]^(bare+induced);
			for(int n=0;n<eInfo.nBands;n++)
				shift-=responseWeight(k)*eInfo.smearPrime(metallicMu,eVars.Hsub_eigs[k][n])*dH(n,n);
		}
		mpiWorld->allReduce(shift,MPIUtil::ReduceSum);
		if(metallicDOS>1e-30) shift*=1./metallicDOS;
		logPrintf("Metallic DFPT Gamma charge response: %.6le; Fermi-level shift [Hartree/bohr]: %.6le %+.6lei\n",charge.abs(),shift.real(),shift.imag());
	}
	PerturbationGradient residual = clone(pInfo.dGradPsi);
	axpy(1.,sourceGradient(),residual);
	double bnorm = dot(sourceGradient(),sourceGradient());
	double rnorm = dot(residual,residual);
	pInfo.lastResponseResidual=bnorm ? sqrt(rnorm/bnorm) : sqrt(rnorm);
	logPrintf("DFPT final relative residual: %.6le\n",bnorm ? sqrt(rnorm/bnorm) : sqrt(rnorm));
	recordScreeningResponse();
	if(pInfo.phononGrid.isNonzero() && (!std::isfinite(rnorm) || rnorm > std::pow(10*pInfo.solverParams.residualTol,2)*bnorm))
		die("DFPT phonon response did not converge; increase perturb-minimize nIterations or tighten the ground-state solve.\n");
	expandKResponse();
}

// Eq. (70) of Baroni et al.: the divided difference of occupations is
// the density-matrix susceptibility, including its finite degenerate limit.
// Square-root weights make its coupling to the XC/Hartree kernel Hermitian.
// The energy-ordering smearing partitions each pair between its two sources.
// It approaches 1 for upward transitions, 0 for downward transitions and 1/2
// at degeneracy. A uniform half weight would depend on which empty bands
// happen to be included in the target projector, and fails off the k mesh.
bool PerturbationSolver::metallic() const
{ return pInfo.phononGrid.isNonzero() && eInfo.fillingsUpdate==ElecInfo::FillingsHsub; }

void PerturbationSolver::setupMetallicResponse()
{
	double Bz=0.;
	metallicMu=eInfo.findMu(eVars.Hsub_eigs,eInfo.nElectrons,Bz);
	metallicWeights.resize(2*eInfo.nStates);
	metallicEnergies.resize(2*eInfo.nStates);
	metallicSqrtF.resize(eInfo.nStates);
	double tail=0.;
	metallicDOS=0.;
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
		metallicSqrtF[k]=eVars.F[k];
		for(double& f:metallicSqrtF[k]) f=sqrt(std::max(0.,f));
		for(double eps:eVars.Hsub_eigs[k])
			metallicDOS-=eInfo.qnums[k].weight*eInfo.smearPrime(metallicMu,eps);
		if(!activeResponse(k)) continue;
		for(int sign=0;sign<2;sign++) {
			int index=k+sign*eInfo.nStates;
			ColumnBundle& C=pInfo.Cinc[index];
			ColumnBundle HC;
			applyH(*C.qnum,eye(eInfo.nBands),HC,C);
			matrix U;
			dagger_symmetrize(C^HC).diagonalize(U,metallicEnergies[index]);
			C=C*U;
			const diagMatrix& eps=metallicEnergies[index];
			tail=std::max(tail,eInfo.smear(metallicMu,eps.back()));
			matrix W(eInfo.nBands,eInfo.nBands);
			for(int n=0;n<eInfo.nBands;n++) for(int m=0;m<eInfo.nBands;m++) {
				double en=eVars.Hsub_eigs[k][n], em=eps[m];
				double diff=en-em;
				double divided=(fabs(diff)<1e-7*eInfo.smearingWidth)
					? eInfo.smearPrime(metallicMu,0.5*(en+em))
					: (eInfo.smear(metallicMu,en)-eInfo.smear(metallicMu,em))/diff;
				W.set(m,n,sqrt(std::max(0.,-divided*eInfo.smear(0.,diff))));
			}
			metallicWeights[index]=W;
		}
	}
	mpiWorld->allReduce(tail,MPIUtil::ReduceMax);
	mpiWorld->allReduce(metallicDOS,MPIUtil::ReduceSum);
	if(tail>1e-6) die("Metallic DFPT offset states need more empty bands: highest-band occupation exceeds 1e-6. Increase elec-n-bands.\n");
	logPrintf("Metallic DFPT: Fermi energy %.12g Hartree, DOS %.6le electrons/Hartree.\n",metallicMu,metallicDOS);
	metallicReady=true;
}

namespace {
// Elementwise multiplication, kept in small band-space matrices. Accessing
// data() synchronizes these matrices correctly in both CPU and GPU builds.
matrix metallicHadamard(const matrix& a,const matrix& b)
{
	matrix result(a.nRows(),a.nCols());
	const complex* ad=a.data(); const complex* bd=b.data();
	complex* rd=result.data();
	for(int i=0;i<a.nRows()*a.nCols();i++) rd[i]=ad[i]*bd[i];
	return result;
}
}

void PerturbationSolver::metallicProject(std::vector<ColumnBundle>& X) const
{
	if(streamedUnprojected) return;
	if(pInfo.qvec.length_squared()>1e-20) return;
	// Constrain the Gamma response to zero total charge (Eq. 75). In this
	// square-root representation this is an orthogonal projection along the
	// DOS vector, equivalent to the self-consistent Fermi-energy shift.
	// Use overlaps rather than assuming a gauge within degenerate subspaces.
	for(int sign=0;sign<2;sign++) {
		complex charge=0.; double norm=0.;
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
			if(!activeResponse(k)) continue;
			int index=k+sign*eInfo.nStates;
			const ColumnBundle& C=pInfo.Cinc[index];
			matrix t=metallicHadamard(metallicWeights[index],C^O(eVars.C[k]));
			charge+=responseWeight(k)*trace(dagger(t)*(C^O(X[index])));
			norm+=responseWeight(k)*std::pow(nrm2(t),2);
		}
		mpiWorld->allReduce(charge,MPIUtil::ReduceSum);
		mpiWorld->allReduce(norm,MPIUtil::ReduceSum);
		if(norm<=1e-30) continue; // zero DOS: insulating limit
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
			if(!activeResponse(k)) continue;
			int index=k+sign*eInfo.nStates;
			const ColumnBundle& C=pInfo.Cinc[index];
			matrix t=metallicHadamard(metallicWeights[index],C^O(eVars.C[k]));
			X[index]-=C*(t*(charge/norm));
		}
	}
}

ColumnBundle PerturbationSolver::metallicMap(const ColumnBundle& X,int k,int sign) const
{
	int index=k+sign*eInfo.nStates;
	const ColumnBundle& C=pInfo.Cinc[index];
	matrix low=C^O(X);
	ColumnBundle high=X;
	high-=C*low;
	ColumnBundle result=high*metallicSqrtF[k];
	result+=C*metallicHadamard(metallicWeights[index],low);
	return result;
}

void PerturbationSolver::metallicHessian(PerturbationGradient& Av,const PerturbationGradient& v)
{
	if(screeningEnabled) {
		std::vector<ColumnBundle> projected=v.X, response(2*eInfo.nStates);
		metallicProject(projected);
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<2;sign++) {
			int index=k+sign*eInfo.nStates;
			response[index]=metallicMap(projected[index],k,sign);
		}
		getdnInc(&response,nullptr,pInfo.dnpq,pInfo.dnmq);
		getdVsclocPsi(pInfo.dnpq,pInfo.dVsclocpq,&pInfo.qvec);
		pInfo.dVsclocmq=conj(pInfo.dVsclocpq);
	}
	std::vector<ColumnBundle> induced(2*eInfo.nStates), kinetic(2*eInfo.nStates);
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) for(int sign=0;sign<2;sign++) {
		if(!activeResponse(k)) continue;
		int index=k+sign*eInfo.nStates;
		const ColumnBundle& C=pInfo.Cinc[index];
		// Preserve the variational amplitudes for force-matrix contractions.
		pInfo.dC[index]=v.X[index];
		ColumnBundle low=C*(C^O(v.X[index])), high=v.X[index]-low;
		ColumnBundle Hhigh;
		applyH(*C.qnum,eye(eInfo.nBands),Hhigh,high);
		Hhigh-=O(high)*eVars.Hsub_eigs[k];
		Hhigh-=C*(C^O(Hhigh));
		kinetic[index]=Hhigh+O(low);
		if(screeningEnabled) {
			ColumnBundle dHC=C.similar(); dHC.zero();
			dHpsi(*C.qnum,dHC,eVars.C[k],sign ? pInfo.dVsclocmq : pInfo.dVsclocpq);
			induced[index]=metallicMap(dHC,k,sign);
		}
	}
	if(screeningEnabled) metallicProject(induced);
	Av.init(e);
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) for(int sign=0;sign<2;sign++) {
		int index=k+sign*eInfo.nStates;
		if(!activeResponse(k)) { Av.X[index].free(); continue; }
		if(screeningEnabled) Av.X[index]=(kinetic[index]+induced[index])*responseWeight(k);
		else Av.X[index]=kinetic[index]*responseWeight(k);
	}
}

void PerturbationSolver::metallicSource()
{
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) for(int sign=0;sign<2;sign++) {
		int index=k+sign*eInfo.nStates;
		if(!activeResponse(k)) { pInfo.dGradTau.X[index].free(); continue; }
		ColumnBundle dHC=pInfo.Cinc[index].similar(); dHC.zero();
		dHtau(*pInfo.Cinc[index].qnum,dHC,eVars.C[k],sign ? -1 : 1);
		pInfo.dGradTau.X[index]=metallicMap(dHC,k,sign);
	}
	metallicProject(pInfo.dGradTau.X);
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<2;sign++)
		pInfo.dGradTau.X[k+sign*eInfo.nStates]*=responseWeight(k);
}

void PerturbationSolver::precondition(PerturbationGradient& v) {
	if(metallic()) { Ksqrt(v); Ksqrt(v); return; }
	//Apply inverse kinetic preconditioner
	for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
		if(!activeResponse(q)) continue;
		precond_inv_kinetic(v.X[q], 1);
		if (!pInfo.commensurate && !pInfo.oneSidedResponse) precond_inv_kinetic(v.X[q + eInfo.nStates], 1);
		if(kReductionActive) { v.X[q]*=1./responseWeight(q); v.X[q+eInfo.nStates]*=1./responseWeight(q); }
	}
}

void PerturbationSolver::Ksqrt(PerturbationGradient& v) {
	if(metallic()) {
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) for(int sign=0;sign<2;sign++) {
			if(!activeResponse(k)) continue;
			int index=k+sign*eInfo.nStates;
			const ColumnBundle& C=pInfo.Cinc[index];
			ColumnBundle low=C*(C^O(v.X[index]));
			v.X[index]-=low;
			precond_inv_kinetic(v.X[index],1,true);
			v.X[index]-=C*(C^O(v.X[index]));
			v.X[index]+=low;
			if(kReductionActive) v.X[index]*=1./sqrt(responseWeight(k));
		}
		return;
	}
	//Apply inverse kinetic preconditioner
	for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
		if(!activeResponse(q)) continue;
		precond_inv_kinetic(v.X[q], 1, true);
		if (!pInfo.commensurate && !pInfo.oneSidedResponse) precond_inv_kinetic(v.X[q + eInfo.nStates], 1, true);
		if(kReductionActive) { double scale=1./sqrt(responseWeight(q)); v.X[q]*=scale; v.X[q+eInfo.nStates]*=scale; }
	}
}

void PerturbationSolver::hessian(PerturbationGradient& Av, const PerturbationGradient& v) {
	if(metallic()) { metallicHessian(Av,v); return; }
	static StopWatch watch("dGradPsi"); watch.start();
	Av.init(e);
	
	if (pInfo.commensurate) {
		for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
			//pInfo.dU[q] = (pInfo.dY[q]^O(eVars.C[q])) + (eVars.C[q]^O(pInfo.dY[q]));
			pInfo.dU[q] = 2*dagger_symmetrize(v.X[q]^pInfo.OC[q]);
			pInfo.dC[q] = v.X[q];
			pInfo.dC[q] -= 0.5*(eVars.C[q]*pInfo.dU[q]);
		}
	} else {
		for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
			int kpq_k = q;
			if(!activeResponse(q)) { pInfo.dC[q].free(); pInfo.dC[q+eInfo.nStates].free(); continue; }
			int kmq_k = q+e.eInfo.nStates;
			int kpq = q ;
			int kmq = q + e.eInfo.nStates;

			pInfo.dC[kpq_k] = v.X[kpq_k] - O(pInfo.Cinc[kpq]*(pInfo.Cinc[kpq]^v.X[kpq_k]));
			if(!pInfo.oneSidedResponse) pInfo.dC[kmq_k] = v.X[kmq_k] - O(pInfo.Cinc[kmq]*(pInfo.Cinc[kmq]^v.X[kmq_k]));
		}
	}
	if (pInfo.commensurate) {
		getdn(pInfo.dn, &pInfo.dC);
		getdVsclocPsi(pInfo.dn, pInfo.dVscloc);
		e.iInfo.augmentDensityGridGrad(pInfo.dVscloc);
		pInfo.E_nAug_dVsclocpsi = iInfo.getE_nAug();
	} else {
		if(screeningEnabled) {
			getdnInc(&pInfo.dC, 0, pInfo.dnpq, pInfo.dnmq);
			getdVsclocPsi(pInfo.dnpq, pInfo.dVsclocpq, &pInfo.qvec);
			pInfo.dVsclocmq = conj(pInfo.dVsclocpq);
		}
	}

	if (pInfo.commensurate) {
		for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
			const QuantumNumber& qnum = e.eInfo.qnums[q];
			ColumnBundle dwGradEq, dHCq = eVars.C[q].similar(), HdCq = eVars.C[q].similar();

			diagMatrix F = eVars.F[q];
			applyH(e.eInfo.qnums[q], eVars.F[q], HdCq, pInfo.dC[q]);
			dHpsi(e.eInfo.qnums[q], dHCq, eVars.C[q], pInfo.dVscloc, &eVars.VdagC[q]);

			pInfo.CdagdHC[q] = eVars.C[q]^dHCq;
			pInfo.dHsub[q] = (pInfo.dC[q]^pInfo.HC[q]) + pInfo.CdagdHC[q] + (eVars.C[q]^HdCq);
			
			dwGradEq = dHCq;
			dwGradEq += HdCq;
			dwGradEq -= O(eVars.C[q]*pInfo.dHsub[q]+pInfo.dC[q]*eVars.Hsub[q]);
			
			dwGradEq = dwGradEq*eVars.F[q];
			
			dwGradEq += (-0.5)*pInfo.grad[q]*(eVars.F[q]*pInfo.dU[q]); //Second order correction term to be used if C does not minimize energy exactly

			Av.X[q] = dwGradEq*qnum.weight;
		}
	} else {
		std::vector<const ColumnBundle*> input;
		std::vector<ColumnBundle> applied;
		const bool batch=pInfo.fullMeshResponse && pInfo.optimizeFrozenResponse;
		if(batch) {
			for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<responseSigns();sign++)
				input.push_back(&pInfo.dC[k+sign*eInfo.nStates]);
			applyHBatched(input,applied);
		}
		size_t block=0;
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
			if(!activeResponse(k)) { Av.X[k].free(); Av.X[k+eInfo.nStates].free(); continue; }
			for(int sign=0;sign<responseSigns();sign++) {
				int index=k+sign*eInfo.nStates;
				const ColumnBundle& C=pInfo.Cinc[index];
				const QuantumNumber& qnum=sign ? pInfo.kminusq_vectors[k] : pInfo.kplusq_vectors[k];
				ColumnBundle dHC=C.similar(), HdC;
				dHC.zero();
				if(screeningEnabled) dHpsi(qnum,dHC,eVars.C[k],sign ? pInfo.dVsclocmq : pInfo.dVsclocpq);
				if(batch) HdC=std::move(applied[block++]);
				else { HdC=C.similar(); applyH(qnum,eVars.F[k],HdC,pInfo.dC[index]); }
				if(bareHessianCapture) {
					ColumnBundle bare=HdC-O(pInfo.dC[index]*eVars.Hsub[k]);
					bareHessianCapture->X[index]=(bare-O(C*(C^bare)))*eVars.F[k]*responseWeight(k);
				}
				ColumnBundle gradient=dHC+HdC;
				gradient-=O(C*(C^dHC));
				gradient-=O(pInfo.dC[index]*eVars.Hsub[k]);
				Av.X[index]=(gradient-O(C*(C^gradient)))*eVars.F[k]*responseWeight(k);
			}
			if(pInfo.oneSidedResponse) Av.X[k+eInfo.nStates].free();
		}
	}

	watch.stop();
}

void PerturbationSolver::calcdGradTau() {
	cachedSource=nullptr;
	if(usePhononCache() && !kReductionActive) {
		auto cached=sourceCache.find(pInfo.datom.get());
		if(cached!=sourceCache.end()) { pInfo.dGradTau.X.clear(); cachedSource=&cached->second; sourceCacheHits++; return; }
	}
	// Cache insertion transfers X; force contractions can request another source
	// before the next solve initializes the perturbation gradients.
	pInfo.dGradTau.init(e);
	sourceBuilds++;
	if(metallic()) { metallicSource(); cacheSource(); return; }
	static StopWatch watch("dGradTau"); watch.start();
	if (pInfo.commensurate) {
		
		if (pInfo.datom && pInfo.datom->isUltrasoft(iInfo)) {
			getdnatom(pInfo.datom->dnatom);
			
			for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
				ColumnBundle dOC = eVars.C[q].similar();
				dOC.zero();
				
				iInfo.species[pInfo.datom->mode.sp]->augmentOverlapDeriv(eVars.C[q], dOC, pInfo.datom->Vatom[q], pInfo.datom->dVatom[q]);
				pInfo.dUmhalfatom[q] = (-0.5)*eVars.C[q]^dOC;
				
				pInfo.datom->dCatom[q] = eVars.C[q]*pInfo.dUmhalfatom[q];
			}
			
			ScalarFieldArray dnCatom(eVars.n.size());
			nullToZero(dnCatom, e.gInfo);
			getdn(dnCatom, &pInfo.datom->dCatom);
			pInfo.datom->dnatom += dnCatom;
		}
		
		nullToZero(pInfo.dVsclocTau, e.gInfo);
		
		if (pInfo.datom && pInfo.datom->isUltrasoft(iInfo))
			getdVsclocTau(pInfo.dVsclocTau, &pInfo.datom->dnatom);
		else
			getdVsclocTau(pInfo.dVsclocTau);
		
		iInfo.augmentDensityGridGrad(pInfo.dVsclocTau);
		pInfo.E_nAug_dVscloctau = iInfo.getE_nAug();
			
		for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
			const QuantumNumber& qnum = e.eInfo.qnums[q];
			ColumnBundle dwGradEq, dHCq = eVars.C[q].similar(), HdCq = eVars.C[q].similar();
			diagMatrix F = eVars.F[q];

			dHtau(e.eInfo.qnums[q], dHCq, eVars.C[q], pInfo.dVsclocTau);
			pInfo.dHsubatom[q] = eVars.C[q]^dHCq;
			pInfo.CdagdHCatom[q] = pInfo.dHsubatom[q];
			
			if(pInfo.datom && pInfo.datom->isUltrasoft(iInfo)) {
				applyH(e.eInfo.qnums[q], eVars.F[q], HdCq, pInfo.datom->dCatom[q]);
				pInfo.dHsubatom[q] += (pInfo.datom->dCatom[q]^pInfo.HC[q]) + (eVars.C[q]^HdCq);
			}

			dwGradEq = dHCq;
			dwGradEq -= pInfo.OC[q]*pInfo.dHsubatom[q];
			
			if(pInfo.datom && pInfo.datom->isUltrasoft(iInfo)) {
				//Atom perturbation causes wfns to become unnormalized, so dC needs to be factored in the gradient
				dwGradEq += HdCq;
				dwGradEq -= O(pInfo.datom->dCatom[q]*eVars.Hsub[q]);
				
				//Derivative of overlap operator w.r.t atpos
				iInfo.species[pInfo.datom->mode.sp]->augmentOverlapDeriv((-1)*eVars.C[q]*eVars.Hsub[q], dwGradEq, pInfo.datom->Vatom[q], pInfo.datom->dVatom[q]);
			}
			
			dwGradEq = dwGradEq*F;
			
			if(pInfo.datom && pInfo.datom->isUltrasoft(iInfo))
				dwGradEq += pInfo.grad[q]*(pInfo.dUmhalfatom[q]*F); //Correction term 
			
			pInfo.dGradTau.X[q] = dwGradEq*qnum.weight;
		}
	} else {
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
			if(!activeResponse(k)) { pInfo.dGradTau.X[k].free(); pInfo.dGradTau.X[k+eInfo.nStates].free(); continue; }
			for(int sign=0;sign<responseSigns();sign++) {
				int index=k+sign*eInfo.nStates;
				const ColumnBundle& C=pInfo.Cinc[index];
				const QuantumNumber& qnum=sign ? pInfo.kminusq_vectors[k] : pInfo.kplusq_vectors[k];
				ColumnBundle dHC=C.similar(); dHC.zero();
				dHtau(qnum,dHC,eVars.C[k],sign ? -1 : 1);
				pInfo.dGradTau.X[index]=(dHC-O(C*(C^dHC)))*eVars.F[k]*responseWeight(k);
			}
			if(pInfo.oneSidedResponse) pInfo.dGradTau.X[k+eInfo.nStates].free();
		}
	}

	watch.stop();
	cacheSource();
}

bool PerturbationSolver::usePhononCache() const
{
	// Cache lifetimes are limited to the frozen, norm-conserving phonon path.
	// General perturbations and finite-difference tests may change the state.
	return pInfo.phononCache && pInfo.phononGrid.isNonzero()
		&& !pInfo.testing && !pInfo.commensurate && pInfo.datom
		&& !pInfo.dVext && !pInfo.drhoExt;
}

void PerturbationSolver::cacheSource()
{
	if(!usePhononCache() || kReductionActive || pInfo.phononSourceCacheMiB<=0.) return;
	size_t bytes=0;
	for(const ColumnBundle& X:pInfo.dGradTau.X)
		bytes+=X.nCols()*X.colLength()*sizeof(complex);
	// Metallic sources include a collective charge projection. All ranks must
	// therefore cache the same entries, even when their local basis sizes differ.
	double maximumBytes=double(bytes);
	mpiWorld->allReduce(maximumBytes,MPIUtil::ReduceMax);
	double budget=pInfo.phononSourceCacheMiB*1024.*1024.;
	if(double(sourceCacheBytes)+maximumBytes>budget) return;
	auto inserted=sourceCache.emplace(pInfo.datom.get(),std::move(pInfo.dGradTau));
	cachedSource=&inserted.first->second;
	sourceCacheBytes+=size_t(maximumBytes);
}

const PerturbationGradient& PerturbationSolver::sourceGradient() const
{ return cachedSource ? *cachedSource : pInfo.dGradTau; }

void PerturbationSolver::releasePhononResponse()
{
	assert(pInfo.phononGrid.isNonzero());
	cachedSource=nullptr; sourceCache.clear(); sourceCacheBytes=0;
	state.X.clear(); pInfo.dGradTau.X.clear(); pInfo.dGradPsi.X.clear();
	pInfo.dC.clear(); pInfo.Cinc.clear();
	pInfo.grad.clear(); pInfo.HC.clear(); pInfo.OC.clear();
	pInfo.dU.clear(); pInfo.dUmhalfatom.clear(); pInfo.dHsub.clear(); pInfo.dHsubatom.clear();
	pInfo.CdagdHC.clear(); pInfo.CdagdHCatom.clear();
	pInfo.dnpq.clear(); pInfo.dnmq.clear(); pInfo.dVsclocpq.clear(); pInfo.dVsclocmq.clear();
	pInfo.kplusq_basis.clear(); pInfo.kminusq_basis.clear();
	pInfo.kplusq_vectors.clear(); pInfo.kminusq_vectors.clear(); pInfo.offsetStatesReady=false;
	metallicWeights.clear(); metallicEnergies.clear(); metallicSqrtF.clear();
	barePotentialPlus.clear(); barePotentialMinus.clear();
	barePotentialReady=false; barePotentialAtom=nullptr;
}

void PerturbationSolver::reportPhononCache() const
{
	logPrintf("DFPT q cache: %zu retained sources, %d hits, %d source builds, %d bare potential pairs, maximum rank storage %.3f MiB.\n",
		sourceCache.size(),sourceCacheHits,sourceBuilds,barePotentialBuilds,sourceCacheBytes/(1024.*1024.));
}

void PerturbationSolver::getGrad(std::vector<ColumnBundle> *grad, std::vector<ColumnBundle> Y) {
	assert(pInfo.testing);
	
	for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
		const QuantumNumber& qnum = e.eInfo.qnums[q];
		ColumnBundle HC, gradq, HCFUmhalf;
		matrix dHtilde, HtildeCommF;

		HCFUmhalf.zero();
		gradq.zero();

		diagMatrix F = eVars.F[q];

		applyH(qnum, F, HC, eVars.C[q]);

		//HtildeCommF = eVars.Hsub[q]*F - F*eVars.Hsub[q];
		matrix Usqrtinv = invsqrt(Y[q]^O(Y[q]));
		//ColumnBundle HCFUsqrtinv = HC*(F*Usqrtinv);
		gradq = HC;
		gradq -= O(eVars.C[q]*eVars.Hsub[q]);
		//gradq += 0.5*O(eVars.C[q]*HtildeCommF);
		(*grad)[q]  = gradq*F*Usqrtinv*qnum.weight;
	}
}

// Fixed-potential eigensolver: keeps primitive-cell states and density untouched.
namespace {
class OffsetBandMinimizer : public Minimizable<ColumnBundle>
{
	PerturbationSolver& solver;
	ColumnBundle& C;
public:
	OffsetBandMinimizer(PerturbationSolver& solver, ColumnBundle& C) : solver(solver), C(C) {}
	void step(const ColumnBundle& dir, double alpha) override
	{ C += alpha*dir; C = C*invsqrt(C^O(C)); }
	void constrain(ColumnBundle& dir) override { dir -= C*(C^O(dir)); }
	double compute(ColumnBundle* grad, ColumnBundle* Kgrad) override
	{
		ColumnBundle HC;
		solver.applyH(*C.qnum, eye(C.nCols()), HC, C);
		matrix Hsub = C^HC;
		if(grad)
		{
			*grad = HC;
			*grad -= O(C)*Hsub;
			if(Kgrad) { *Kgrad = *grad; precond_inv_kinetic(*Kgrad, 1.); }
		}
		return real(trace(Hsub));
	}
};

// Rank-revealing orthonormalization of trial vectors. Residuals can become
// linearly dependent at degeneracies, so a full inverse square root is unsafe.
ColumnBundle orthonormalOffset(const ColumnBundle& trial)
{
	matrix vectors;
	diagMatrix norms;
	dagger_symmetrize(trial^O(trial)).diagonalize(vectors,norms);
	int first=0;
	double cutoff=std::max(1e-14,1e-12*norms.back());
	while(first<int(norms.size()) && norms[first]<=cutoff) first++;
	if(first==int(norms.size())) return ColumnBundle();
	diagMatrix scale=norms(first,norms.size());
	for(double& value:scale) value=1./sqrt(value);
	return trial*(vectors(0,vectors.nRows(),first,vectors.nCols())*scale);
}

// A restarted block Davidson solve avoids the nonlinear line-search failures
// of CG for offset states far from the original k point. Extra random trial
// vectors also allow entry into symmetry sectors absent from the initial guess.
double minimizeOffsetDavidson(PerturbationSolver& solver, ColumnBundle& occupied, int nIterations)
{
	int nOccupied=occupied.nCols();
	int nKeep=std::min(int(occupied.basis->nbasis),nOccupied+std::max(2,nOccupied/2));
	ColumnBundle C=occupied.similar(nKeep);
	C.setSub(0,occupied);
	C.randomize(nOccupied,nKeep);
	C=orthonormalOffset(C);
	double residual=INFINITY;
	for(int iter=0;iter<=nIterations;iter++) {
		ColumnBundle HC;
		solver.applyH(*C.qnum,eye(C.nCols()),HC,C);
		matrix vectors;
		diagMatrix eigs;
		dagger_symmetrize(C^HC).diagonalize(vectors,eigs);
		C=C*vectors; HC=HC*vectors;
		ColumnBundle R=HC-O(C)*eigs;
		diagMatrix rNorm=diagDot(R,R);
		residual=0.;
		for(int b=0;b<nOccupied;b++) {
			double value=sqrt(std::max(0.,rNorm[b])/C.basis->gInfo->detR);
			if(!std::isfinite(rNorm[b])) {residual=INFINITY; break;}
			residual=std::max(residual,value);
		}
		if(residual<1e-7 || !std::isfinite(residual) || iter==nIterations) break;
		diagMatrix KEref=(-0.5)*diagDot(C,L(C));
		precond_inv_kinetic_band(R,KEref);
		// Reorthogonalize twice before selecting independent search directions.
		for(int pass=0;pass<2;pass++) R-=C*(C^O(R));
		R=orthonormalOffset(R);
		if(!R) break;
		ColumnBundle HR;
		solver.applyH(*R.qnum,eye(R.nCols()),HR,R);
		int nc=C.nCols(),nr=R.nCols();
		matrix H=zeroes(nc+nr,nc+nr),cross=C^HR;
		H.set(0,nc,0,nc,eigs);
		H.set(0,nc,nc,nc+nr,cross);
		H.set(nc,nc+nr,0,nc,dagger(cross));
		H.set(nc,nc+nr,nc,nc+nr,R^HR);
		dagger_symmetrize(H).diagonalize(vectors,eigs);
		int next=std::min(nKeep,nc+nr);
		C=C*vectors(0,nc,0,next)+R*vectors(nc,nc+nr,0,next);
	}
	occupied=C.getSub(0,nOccupied);
	return residual;
}
}

void PerturbationSolver::computeMetallicGroundStates(bool updateDensity)
{
	// An energy minimization barely constrains exponentially empty bands.
	// Their eigenvalues and wavefunctions still enter the divided differences,
	// so converge the entire stored subspace in the frozen SCF Hamiltonian.
	logPrintf("Converging metallic DFPT source bands, including empty states.\n");
	double residual=0.;
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
		double current=minimizeOffsetDavidson(*this,eVars.C[k],std::max(200,e.elecMinParams.nIterations));
		residual=std::max(residual,std::isfinite(current) ? current : INFINITY);
		iInfo.project(eVars.C[k],eVars.VdagC[k]);
	}
	mpiWorld->allReduce(residual,MPIUtil::ReduceMax);
	logPrintf("Metallic source-state maximum residual [Hartree]: %.6le\n",residual);
	if(!std::isfinite(residual) || residual>1e-6)
		die("Metallic DFPT source bands did not converge; increase electronic-minimize nIterations.\n");
	if(updateDensity) eVars.elecEnergyAndGrad(e.ener,nullptr,nullptr,true);
	else for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
		ColumnBundle HC;
		eVars.applyHamiltonian(k,eye(eInfo.nBands),HC,e.ener,true);
	}
}

void PerturbationSolver::computeIncommensurateWfns()
{
	logPrintf("Computing occupied states at k+q and k-q in the ground-state potential.\n");
	MinimizeParams params = e.elecMinParams;
	params.fpLog = nullLog;
	params.nIterations = std::max(200, params.nIterations);
	params.energyDiffThreshold = params.energyDiffThreshold > 0 ? std::min(1e-13,params.energyDiffThreshold) : 1e-13;
	params.knormThreshold = 0.;
	params.fdTest = false;
	std::vector<int> reused(2*eInfo.nStates,-1);
	std::vector<int> selected(eInfo.nStates,1);
	if(pInfo.phononKBatchStart>=0) {
		std::fill(selected.begin(),selected.end(),0);
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(pInfo.inPhononKBatch(k)) selected[k]=1;
		mpiWorld->allReduceData(selected,MPIUtil::ReduceMax);
	}
	if(pInfo.phononGrid.isNonzero() && pInfo.phononReuseOffsetStates) {
		for(int sign=0;sign<2;sign++) for(int k=0;k<eInfo.nStates;k++) {
			if(!selected[k]) continue;
			const vector3<>& target=sign ? pInfo.kminusq_vectors[k].k : pInfo.kplusq_vectors[k].k;
			for(int source=0;source<eInfo.nStates;source++) {
				vector3<> delta=target-eInfo.qnums[source].k;
				for(int d=0;d<3;d++) delta[d]-=round(delta[d]);
				if(delta.length_squared()<1e-16) {reused[k+sign*eInfo.nStates]=source; break;}
			}
		}
		// Broadcast one source at a time, keeping memory bounded and allowing
		// source and destination k points to belong to different MPI ranks.
		if(pInfo.phononGroundState) {
			std::vector<std::pair<int,ColumnBundle*>> requests;
			for(int sign=0;sign<2;sign++) for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
				int index=k+sign*eInfo.nStates;
				if(pInfo.inPhononKBatch(k) && reused[index]>=0) requests.emplace_back(reused[index],&pInfo.Cinc[index]);
			}
			pInfo.phononGroundState->reconstruct(requests);
		} else for(int source=0;source<eInfo.nStates;source++) {
			if(std::find(reused.begin(),reused.end(),source)==reused.end()) continue;
			ColumnBundle received;
			ColumnBundle& Csource=eInfo.isMine(source) ? eVars.C[source] : received;
			if(!eInfo.isMine(source)) Csource.init(eInfo.nBands,e.basis[source].nbasis,&e.basis[source],&eInfo.qnums[source],isGpuEnabled());
			if(mpiWorld->nProcesses()>1) mpiWorld->bcastData(Csource,eInfo.whose(source));
			for(int sign=0;sign<2;sign++) for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
				if(!pInfo.inPhononKBatch(k)) continue;
				int index=k+sign*eInfo.nStates;
				if(reused[index]!=source) continue;
				ColumnBundle& target=pInfo.Cinc[index];
				target.zero();
				ColumnBundleTransform::BasisWrapper wrapper(*target.basis);
				ColumnBundleTransform(eInfo.qnums[source].k,e.basis[source],target.qnum->k,wrapper,
					1,SpaceGroupOp(),1).scatterAxpy(1.,Csource,target,0,1);
			}
		}
	}
	double maxResidual = 0.;
	int nRetries=0,nReused=0,nRefined=0;
	for(int q=eInfo.qStart; q<eInfo.qStop; q++)
	for(int sign=0; sign<2; sign++)
	{
		if(!pInfo.inPhononKBatch(q)) continue;
		ColumnBundle& C = pInfo.Cinc[q+sign*eInfo.nStates];
		bool reuse=reused[q+sign*eInfo.nStates]>=0;
		if(!reuse) // Transfer periodic parts into an off-mesh basis.
			for(int b=0; b<C.nCols(); b++) C.setColumn(b,0,eVars.C[q].getColumn(b,0));
		C = C*invsqrt(C^O(C));
		OffsetBandMinimizer minimizer(*this,C);
		ColumnBundle grad;
		minimizer.compute(&grad,nullptr);
		double residual=sqrt(dotc(grad,grad).real()/(C.basis->gInfo->detR*C.nCols()));
		if(reuse) nReused++;
		if(!reuse || !std::isfinite(residual) || residual>1e-6) {
			if(reuse) nRefined++;
			minimizer.minimize(params);
			minimizer.compute(&grad,nullptr);
			residual=sqrt(dotc(grad,grad).real()/(C.basis->gInfo->detR*C.nCols()));
		}
		// A stationary transferred subspace can miss a lower symmetry sector.
		// Metallic divided differences need the actual lowest stored bands;
		// always verify non-reused offsets with the expanded Davidson subspace.
		if((metallic() && !reuse) || !std::isfinite(residual) || residual>1e-6) {
			nRetries++;
			residual=minimizeOffsetDavidson(*this,C,params.nIterations);
			if(!std::isfinite(residual) || residual>1e-6)
				fprintf(stderr,"Offset-state solve failed on rank %d, k index %d, sign %s, residual %.6le Hartree.\n",
					mpiWorld->iProcess(),q,sign ? "-q" : "+q",residual);
		}
		maxResidual=std::max(maxResidual,std::isfinite(residual) ? residual : INFINITY);
	}
	mpiWorld->allReduce(maxResidual,MPIUtil::ReduceMax);
	mpiWorld->allReduce(nRetries,MPIUtil::ReduceSum);
	mpiWorld->allReduce(nReused,MPIUtil::ReduceSum);
	mpiWorld->allReduce(nRefined,MPIUtil::ReduceSum);
	if(pInfo.phononGrid.isNonzero()) logPrintf("DFPT offset-state reuse: %d/%d k/sign pairs (%d required refinement).\n",nReused,2*eInfo.nStates,nRefined);
	if(nRetries) logPrintf("Offset-state Davidson fallback used for %d k/sign pairs.\n",nRetries);
	logPrintf("Offset-state maximum residual [Hartree]: %.6le\n",maxResidual);
	if(!std::isfinite(maxResidual) || maxResidual>1e-6)
		die("Offset occupied states did not converge; increase electronic-minimize nIterations.\n");
}

// Bloch derivative (grad + i*q_cart) of a lattice-periodic field.
complexScalarField blochDerivative(const complexScalarField& f, const vector3<>& q, int idir)
{
	vector3<> dir; dir[idir]=1.;
	return -I(ionicDerivative(J(f),q,dir));
}

#ifdef GPU_ENABLED
void fullBlochDerivative_gpu(const vector3<int> S, complex* field, vector3<> q, vector3<> Ge);
#endif
// Keep reciprocal differentiation on the preferred device. Its full-grid
// Nyquist convention is identical to ionicDerivative, including at G+q.
static complexScalarField frozenBlochDerivative(const complexScalarFieldTilde& fG,
	const vector3<>& q, int idir)
{
	complexScalarFieldTilde result=clone(fG);
	const GridInfo& g=fG->gInfo;
	const vector3<> Ge=g.G.column(idir);
#ifdef GPU_ENABLED
	fullBlochDerivative_gpu(g.S,result->dataGpu(),q,Ge);
#else
	complex* data=result->data();
	for(int ix=0; ix<g.S[0]; ix++)
	for(int iy=0; iy<g.S[1]; iy++)
	for(int iz=0; iz<g.S[2]; iz++) {
		vector3<int> iG(ix,iy,iz);
		for(int j=0; j<3; j++) if(2*iG[j]>g.S[j]) iG[j]-=g.S[j];
		data[g.fullGindex(iG)] *= complex(0,dot(iG+q,Ge));
	}
#endif
	return I(result);
}

// Derivative of a full reciprocal-space field at G+q along a Cartesian direction.
complexScalarFieldTilde ionicDerivative(const complexScalarFieldTilde& V, const vector3<>& q, const vector3<>& dir)
{
	const GridInfo& g = V->gInfo;
	complexScalarFieldTilde result = clone(V);
	complex* data = result->data();
	for(int ix=0; ix<g.S[0]; ix++)
	for(int iy=0; iy<g.S[1]; iy++)
	for(int iz=0; iz<g.S[2]; iz++)
	{
		vector3<int> iG(ix,iy,iz);
		for(int j=0;j<3;j++) if(2*iG[j]>g.S[j]) iG[j]-=g.S[j];
		data[g.fullGindex(iG)] *= complex(0,-dot((iG+q)*g.G,dir));
	}
	return result;
}

void PerturbationSolver::updateExcorrCache()
{
	ScalarFieldArray nXC = eVars.get_nXC();
	
	if (nXC.size() > 1)
		return;
	
	nullToZero(pInfo.sigma_cached,e.gInfo);
	initZero(pInfo.sigma_cached);
	const ScalarFieldTilde nG = J(nXC[0]);
	for(int i=0;i<3;i++) {
		pInfo.IDJn_cached[i]=I(D(nG,i));
		pInfo.sigma_cached+=pInfo.IDJn_cached[i]*pInfo.IDJn_cached[i];
	}

	e.exCorr.getSecondDerivatives(nXC[0], pInfo.e_nn_cached, pInfo.e_sigma_cached, pInfo.e_nsigma_cached, pInfo.e_sigmasigma_cached, 1e-9, &pInfo.sigma_cached);
	if(pInfo.kiSpinResponse)
		e.exCorr.getSpinSecondDerivatives(nXC[0],pInfo.sigma_cached,
			pInfo.spin_e_mm_cached,pInfo.spin_e_mu_cached,pInfo.spin_e_uu_cached,pInfo.spin_e_v_cached);
	// Finite-q KI needs no commensurate HC cache. Its driver keeps the
	// density fixed even when q and the external orbital source change.
	if(pInfo.fullMeshResponse && pInfo.optimizeFrozenResponse && !pInfo.commensurate)
		groundCachesReady=true;
}

void PerturbationSolver::updateHC() {
	if (!pInfo.commensurate) return;
	
	iInfo.augmentDensityGridGrad(eVars.Vscloc);
	pInfo.E_nAug_cached = iInfo.getE_nAug();
	
	for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
		QuantumNumber qnum = eInfo.qnums[q];
		applyH(qnum, eVars.F[q], pInfo.HC[q], eVars.C[q]);
		pInfo.OC[q] = O(eVars.C[q]);
		
		pInfo.grad[q] = pInfo.HC[q];
		pInfo.grad[q] -= pInfo.OC[q]*(eVars.Hsub[q]);
	}
}

void PerturbationSolver::updateNonlocalDerivs()
{
	if (!pInfo.datom || !pInfo.commensurate) return;
	AtomPerturbation& datom = *pInfo.datom;
	AtomicMode m = datom.mode;
	auto sp = iInfo.species[m.sp];
	for(int q=eInfo.qStart; q<eInfo.qStop; q++)
	{	const ColumnBundle& Cq = eVars.C[q];
		datom.Vatom[q] = sp->getVatom(Cq, m.at);
		datom.dVatom[q] = -D(datom.Vatom[q], m.dirCartesian);
		datom.VdagCatom[q] = datom.Vatom[q] ^ Cq;
		datom.dVdagCatom[q] = datom.dVatom[q] ^ Cq;
	}
	sp->augmentDensityGridGradDeriv(eVars.Vscloc, m.at, ManagedArray<vector3<>>(&m.dirLattice, 1).dataPref());
	datom.E_nAug_datom = sp->E_nAug;
}

void PerturbationSolver::getdn(ScalarFieldArray& dn, const std::vector<ColumnBundle>* dC, const std::vector<ColumnBundle>* C)
{	static StopWatch watch("getdn"); watch.start();

	initZero(dn);
	//Runs over all states and accumulates density to the corresponding spin channel of the total density
	e.iInfo.augmentDensityInit();
	
	std::vector<matrix> VdagdCq;

	for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++)
	{
		if (pInfo.densityAugRequired(e)) {
			VdagdCq.resize(e.iInfo.species.size());
			e.iInfo.project((*dC)[q], VdagdCq);
		}
		
		if (C) {
			std::vector<matrix> VdagCq(e.iInfo.species.size());
			e.iInfo.project((*C)[q], VdagCq); //update the atomic projections
			
			dn += 2.0*e.eInfo.qnums[q].weight * Real(diagouterI(eVars.F[q], (*C)[q], (*dC)[q], dn.size(), &e.gInfo));
			e.iInfo.augmentDensitySpherical(e.eInfo.qnums[q], eVars.F[q], VdagCq, &VdagdCq);
		} else {
			dn += 2.0*e.eInfo.qnums[q].weight * Real(diagouterI(eVars.F[q], eVars.C[q], (*dC)[q], dn.size(), &e.gInfo));
			e.iInfo.augmentDensitySpherical(e.eInfo.qnums[q], eVars.F[q], eVars.VdagC[q], &VdagdCq);
		}
	}
	e.iInfo.augmentDensityGrid(dn);
	for(ScalarField& ns: dn)
	{	nullToZero(ns, e.gInfo);
		ns->allReduceData(mpiWorld, MPIUtil::ReduceSum);
	}
	e.symm.symmetrize(dn);
	watch.stop();
}

void PerturbationSolver::getdnInc(const std::vector<ColumnBundle>* dC, const std::vector<ColumnBundle>* C, complexScalarFieldArray& dnpq, complexScalarFieldArray& dnmq)
{   static StopWatch watch("getdnInc"); watch.start();

	initZero(dnpq);
	initZero(dnmq);

	for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++)
	{
		int kpq_k = q;
		if(!activeResponse(q)) continue;
		int kmq_k = q+e.eInfo.nStates;

		if(pInfo.oneSidedResponse) {
			if(!C && frozenOrbitals.count(&eVars.C[q])) {
				const auto& fixed=frozenOrbitals.at(&eVars.C[q]);
				dnpq+=2.*responseWeight(q)*diagouterI_cached((*dC)[kpq_k],fixed.values,fixed.fillings,false,dnpq.size(),&e.gInfo);
			} else dnpq+=2.*responseWeight(q)*diagouterI(eVars.F[q],(*dC)[kpq_k],C ? (*C)[q] : eVars.C[q],dnpq.size(),&e.gInfo);
		} else if (C) {
			dnpq += responseWeight(q) * diagouterI((metallic() ? eye(eInfo.nBands) : eVars.F[q]), (*C)[q], (*dC)[kmq_k], dnpq.size(), &e.gInfo);
			dnpq += responseWeight(q) * diagouterI((metallic() ? eye(eInfo.nBands) : eVars.F[q]), (*dC)[kpq_k], (*C)[q], dnpq.size(), &e.gInfo);
		} else if(frozenOrbitals.count(&eVars.C[q])) {
			const auto& fixed=frozenOrbitals.at(&eVars.C[q]);
			dnpq+=responseWeight(q)*diagouterI_cached((*dC)[kmq_k],fixed.values,fixed.fillings,true,dnpq.size(),&e.gInfo);
			dnpq+=responseWeight(q)*diagouterI_cached((*dC)[kpq_k],fixed.values,fixed.fillings,false,dnpq.size(),&e.gInfo);
		} else {
			dnpq += responseWeight(q) * diagouterI((metallic() ? eye(eInfo.nBands) : eVars.F[q]), eVars.C[q], (*dC)[kmq_k], dnpq.size(), &e.gInfo);
			dnpq += responseWeight(q) * diagouterI((metallic() ? eye(eInfo.nBands) : eVars.F[q]), (*dC)[kpq_k], eVars.C[q], dnpq.size(), &e.gInfo);
		}
	}

	for(complexScalarField& ns: dnpq)
	{	nullToZero(ns, e.gInfo);
		ns->allReduceData(mpiWorld, MPIUtil::ReduceSum);
	}

	if(!pInfo.fullMeshResponse) e.symm.symmetrize(dnpq);
	symmetrizeResponseDensity(dnpq);
	
	dnmq = conj(dnpq);
	
	watch.stop();
}

void PerturbationSolver::getdnatom(ScalarFieldArray& dnatom)
{	static StopWatch watch("getdn"); watch.start();
	nullToZero(dnatom, e.gInfo, eVars.n.size());
	for (unsigned s = 0; s < dnatom.size(); s++)
		initZero(dnatom[s]);
	
	if (pInfo.datom && pInfo.datom->isUltrasoft(iInfo))
	{	const AtomPerturbation& datom = *(pInfo.datom);
		AtomicMode m = datom.mode;
		auto sp = iInfo.species[m.sp];
		//Runs over all states and accumulates density to the corresponding spin channel of the total density
		sp->augmentDensityInit(m.at);
		for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++)
			sp->augmentDensitySpherical(e.eInfo.qnums[q], eVars.F[q], datom.VdagCatom[q], &datom.dVdagCatom[q], 0, m.at);
		sp->augmentDensityGrid(dnatom, m.at);
		
		sp->augmentDensityInit(m.at);
		for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++)
			sp->augmentDensitySpherical(e.eInfo.qnums[q], eVars.F[q], datom.VdagCatom[q], 0, 0, m.at);
		sp->augmentDensityGrid(dnatom, m.at, ManagedArray<vector3<>>(&m.dirLattice, 1).dataPref());
		
		for(ScalarField& ns: dnatom)
		{	nullToZero(ns, e.gInfo);
			ns->allReduceData(mpiWorld, MPIUtil::ReduceSum);
		}
		e.symm.symmetrize(dnatom);
	}
	watch.stop();
}


void PerturbationSolver::applyH(const QuantumNumber& qnum, const diagMatrix& Fq, ColumnBundle& HCq, const ColumnBundle& Cq, const ColumnBundle* localH)
{	static StopWatch watch("applyH"); watch.start();
	assert(Cq); //make sure wavefunction is available for this state
	
	std::vector<matrix> HVdagCq(e.iInfo.species.size());
	std::vector<matrix> VdagCq(e.iInfo.species.size());
	
	HCq.zero();
	
	iInfo.setE_nAug(pInfo.E_nAug_cached);

	if(localH) HCq+=*localH;
	else HCq+=Idag_DiagV_I(Cq,eVars.Vscloc);

	e.iInfo.project(Cq, VdagCq); //update the atomic projections
	e.iInfo.augmentDensitySphericalGrad(qnum, VdagCq, HVdagCq);

	HCq += (-0.5) * L(Cq);

	//Nonlocal pseudopotentials:

	e.iInfo.EnlAndGrad(qnum, Fq, VdagCq, HVdagCq);
	e.iInfo.projectGrad(HVdagCq, Cq, HCq);
	
	watch.stop();
}

void PerturbationSolver::applyHBatched(const std::vector<const ColumnBundle*>& input,std::vector<ColumnBundle>& output)
{
	output.resize(input.size());
	if(input.empty()) return;
	#ifdef GPU_ENABLED
	bool batch=pInfo.fullMeshResponse && pInfo.optimizeFrozenResponse && eVars.Vscloc.size()==1 && useBatchedColumnFFT();
	const GridInfo& grid=*input[0]->basis->gInfo;
	for(const auto* C:input) batch &= !C->isSpinor() && C->basis->gInfo==&grid;
	if(batch) {
		std::vector<ColumnBundle> local(input.size()); std::vector<ColumnBundle*> destinations;
		for(size_t i=0;i<input.size();i++) { local[i]=input[i]->similar(); destinations.push_back(&local[i]); }
		ScalarField potential=eVars.Vscloc[0];
		if(&potential->gInfo!=&grid) potential=Jdag(changeGrid(Idag(potential),grid),true);
		batchedLocalPotentials(input,destinations,potential->dataGpu(),nullptr);
		for(size_t i=0;i<input.size();i++) {
			output[i]=input[i]->similar();
			applyH(*input[i]->qnum,eye(input[i]->nCols()),output[i],*input[i],&local[i]);
			local[i].free();
		}
		return;
	}
	#endif
	for(size_t i=0;i<input.size();i++) {
		output[i]=input[i]->similar();
		applyH(*input[i]->qnum,eye(input[i]->nCols()),output[i],*input[i]);
	}
}

void PerturbationSolver::dH_Vscloc(const QuantumNumber& qnum, ColumnBundle& HCq, const ColumnBundle& Cq, const ScalarFieldArray& dVscloc, const std::vector<matrix>* VdagCq)
{	static StopWatch watch("dH_Vscloc"); watch.start();
	assert(Cq);

	std::vector<matrix> HVdagCq(e.iInfo.species.size());

	HCq.zero();

	HCq += Idag_DiagV_I(Cq, dVscloc, &HCq);

	if (VdagCq) {
		e.iInfo.augmentDensitySphericalGrad(qnum, *VdagCq, HVdagCq);
	} else {
		std::vector<matrix> VdagCq_fresh(e.iInfo.species.size());
		e.iInfo.project(Cq, VdagCq_fresh); //update the atomic projections
		e.iInfo.augmentDensitySphericalGrad(qnum, VdagCq_fresh, HVdagCq);
	}
	e.iInfo.projectGrad(HVdagCq, HCq, HCq);
	
	watch.stop();
}

void PerturbationSolver::dHpsi(const QuantumNumber& qnum, ColumnBundle& HCq, const ColumnBundle& Cq, const ScalarFieldArray& dVscloc, const std::vector<matrix>* VdagCq)
{   static StopWatch watch("dHpsi"); watch.start();
	
	iInfo.setE_nAug(pInfo.E_nAug_dVsclocpsi);
	
	dH_Vscloc(qnum, HCq, Cq, dVscloc, VdagCq);
	watch.stop();
}

void PerturbationSolver::dHpsi(const QuantumNumber& qnum, ColumnBundle& HCq, const ColumnBundle& Cq, const complexScalarFieldArray& dVscloc)
{
	auto cached=frozenOrbitals.find(&Cq);
	if(cached!=frozenOrbitals.end()) {
		HCq=Idag_DiagV_I_cached(Cq,dVscloc,cached->second.values,HCq);
		return;
	}
	if((pInfo.fullMeshResponse && pInfo.optimizeFrozenResponse)
		|| (pInfo.phononGrid.isNonzero() && pInfo.phononOptimize && !pInfo.testing)) {
		// The native complex operator preserves quadrature and target-basis
		// projection while avoiding the duplicated wavefunction FFTs.
		HCq = Idag_DiagV_I(Cq, dVscloc, &HCq);
		return;
	}
	ColumnBundle HCr = HCq.similar(), HCi = HCq.similar();
	HCr = Idag_DiagV_I(Cq, Real(dVscloc), &HCr);
	HCi = Idag_DiagV_I(Cq, Imag(dVscloc), &HCi);
	HCq = HCr + complex(0,1)*HCi;
}

void PerturbationSolver::prepareFrozenOrbitals()
{
	if(frozenOrbitalsReady || !pInfo.fullMeshResponse || !pInfo.optimizeFrozenResponse
		|| pInfo.frozenOrbitalCacheMiB<=0. || metallic()) return;
	frozenOrbitalsReady=true;
	size_t bytes=0; const double budget=pInfo.frozenOrbitalCacheMiB*1024.*1024.;
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
		const ColumnBundle& C=eVars.C[k];
		if(C.isSpinor()) continue;
		size_t required=size_t(C.nCols())*(C.basis->gInfo->nr*sizeof(complex)+sizeof(double));
		if(double(bytes)+required>budget) continue;
		auto& fixed=frozenOrbitals[&C];
		fixed.values=realSpaceColumns(C); fixed.fillings=ManagedArray<double>(eVars.F[k]);
		bytes+=required;
	}
	logPrintf("KI frozen orbital cache: %zu/%d local k points, %.1f MiB (budget %.1f MiB/rank).\n",
		frozenOrbitals.size(),eInfo.qStop-eInfo.qStart,bytes/(1024.*1024.),pInfo.frozenOrbitalCacheMiB);
}

void PerturbationSolver::dHtau(const QuantumNumber& qnum, ColumnBundle& HCq, const ColumnBundle& Cq, const ScalarFieldArray& dVscloc)
{   static StopWatch watch("dHtau"); watch.start();
	assert(pInfo.commensurate);

	iInfo.setE_nAug(pInfo.E_nAug_dVscloctau);
	
	dH_Vscloc(qnum, HCq, Cq, dVscloc);
	
	if (pInfo.datom) {
		//dAugatom(qnum, HC, C);
		
		matrix VatomdagCq, dVatomdagCq, HVatomdagCq, HdVatomdagCq;
		ColumnBundle dVHVdagCq, VHdVdagCq;
		
		AtomicMode m = pInfo.datom->mode;
		auto sp = iInfo.species[m.sp];
		ColumnBundle Vatom = sp->getVatom(Cq, m.at);
		ColumnBundle dVatom = -D(Vatom, m.dirCartesian);
		VatomdagCq = Vatom ^ Cq;
		dVatomdagCq = dVatom ^ Cq;
		
		e.iInfo.setE_nAug(pInfo.E_nAug_cached);
		sp->augmentDensitySphericalGrad(qnum, dVatomdagCq, HdVatomdagCq, m.at);
		sp->augmentDensitySphericalGrad(qnum, VatomdagCq, HVatomdagCq, m.at);
		
		sp->EnlAndGrad(qnum, eVars.F[qnum.index()], dVatomdagCq, HdVatomdagCq, m.at);
		sp->EnlAndGrad(qnum, eVars.F[qnum.index()], VatomdagCq, HVatomdagCq, m.at);
		
		dVHVdagCq = dVatom * HVatomdagCq;
		VHdVdagCq = Vatom * HdVatomdagCq;
		
		HCq += VHdVdagCq;
		HCq += dVHVdagCq;
		
		if (pInfo.datom->isUltrasoft(iInfo)) {
			matrix dHVatomdagCq;
			sp->E_nAug = pInfo.datom->E_nAug_datom;
			sp->augmentDensitySphericalGrad(qnum, VatomdagCq, dHVatomdagCq, m.at);
			HCq += Vatom * dHVatomdagCq;
		}
	}
	
	watch.stop();
}

void PerturbationSolver::dHtau(const QuantumNumber& qnum, ColumnBundle& HCq, const ColumnBundle& Cq, int qsign)
{
	complexScalarFieldArray dVscloc(eVars.Vscloc.size());
	if(usePhononCache()) {
		if(!barePotentialReady || barePotentialAtom!=pInfo.datom.get()
			|| (barePotentialQ-pInfo.qvec).length_squared()>1e-24) {
			barePotentialPlus.resize(eVars.Vscloc.size());
			barePotentialMinus.resize(eVars.Vscloc.size());
			getdVsclocTau(barePotentialPlus,1);
			getdVsclocTau(barePotentialMinus,-1);
			barePotentialBuilds++;
			barePotentialAtom=pInfo.datom.get();
			barePotentialQ=pInfo.qvec;
			barePotentialReady=true;
		}
		dVscloc=qsign==1 ? barePotentialPlus : barePotentialMinus;
	} else getdVsclocTau(dVscloc, qsign);
	dHpsi(qnum, HCq, Cq, dVscloc);
	if(pInfo.datom)
	{
		const AtomicMode& m = pInfo.datom->mode;
		auto sp = iInfo.species[m.sp];
		ColumnBundle Vin = sp->getVatom(Cq,m.at);
		ColumnBundle Vout = sp->getVatom(HCq,m.at);
		ColumnBundle dVin = -D(Vin,m.dirCartesian), dVout = -D(Vout,m.dirCartesian);
		if(Vin.nCols())
		{
			HCq += dVout*(sp->MnlAll*(Vin^Cq));
			HCq += Vout*(sp->MnlAll*(dVin^Cq));
		}
	}
}

void PerturbationSolver::getdVsclocPsi(const ScalarFieldArray dn, ScalarFieldArray& dVscloc)
{	static StopWatch watch("getdVsclocPsi"); watch.start();
	assert(pInfo.commensurate);

	ScalarFieldTilde dnTilde = J(get_nTot(dn));

	// Hartree term:
	ScalarFieldTilde dVsclocTilde = (*e.coulomb)(dnTilde); //Note: external charge and nuclear charge contribute to d_vac as well (see below)

	//Second derivative of excorr energy
	ScalarFieldArray dVxc(eVars.Vxc.size());
	e.exCorr.get_dVxc(eVars.get_nXC(), &dVxc, false, &eVars.tau, &eVars.Vtau, dn);

	for(unsigned s=0; s<eVars.Vscloc.size(); s++)
	{	dVscloc[s] = JdagOJ(dVxc[s]);

		if(s<2) //Include all the spin-independent contributions along the diagonal alone
			dVscloc[s] += Jdag(O(dVsclocTilde), true);
	}

	e.symm.symmetrize(dVscloc);
	watch.stop();
}


void PerturbationSolver::getdVsclocPsi(const complexScalarFieldArray dn, complexScalarFieldArray& dVscloc, vector3<>* q)
{	static StopWatch watch("getdVsclocPsi"); watch.start();

	assert(!pInfo.commensurate);

	complexScalarFieldTilde dnTilde = J(get_nTot(dn));
	if(pInfo.magneticResponse) {
		assert(pInfo.kiSpinResponse && dn.size()==1 && dVscloc.size()==1);
		// A nonmagnetic state's charge and longitudinal magnetic responses
		// decouple. The latter has no Hartree term and uses the same 2*chi_spin
		// Bloch susceptibility, driven by V_m=(V_up-V_down)/2.
		complexScalarField dVm=pInfo.spin_e_mm_cached*dn[0];
		if(pInfo.spin_e_v_cached) {
			const bool optimized=pInfo.fullMeshResponse && pInfo.optimizeFrozenResponse;
			complexScalarField dmGradient[3],du;
			nullToZero(du,e.gInfo); initZero(du);
			for(int i=0;i<3;i++) {
				dmGradient[i]=optimized ? frozenBlochDerivative(dnTilde,*q,i) : blochDerivative(dn[0],*q,i);
				du+=pInfo.IDJn_cached[i]*dmGradient[i];
			}
			dVm+=pInfo.spin_e_mu_cached*du;
			for(int i=0;i<3;i++) {
				complexScalarField term=2*pInfo.spin_e_v_cached*dmGradient[i]
					+pInfo.IDJn_cached[i]*(pInfo.spin_e_mu_cached*dn[0]+pInfo.spin_e_uu_cached*du);
				dVm-=optimized ? frozenBlochDerivative(J(term),*q,i) : blochDerivative(term,*q,i);
			}
		}
		dVscloc[0]=JdagOJ(dVm);
		watch.stop(); return;
	}

	// Hartree term:
	complexScalarFieldTilde dVsclocTilde;
	
	dVsclocTilde = e.coulomb->applyBloch(dnTilde,*q);

	if(e.exCorr.needFiniteDifference_dVxc())
		die("Finite-q response requires an analytic LDA/GGA XC kernel.\n");
	complexScalarField dVxc = pInfo.e_nn_cached*dn[0];
	if(pInfo.e_sigma_cached) // LDA has no gradient-density derivatives.
	{
		const bool optimized=pInfo.fullMeshResponse && pInfo.optimizeFrozenResponse;
		complexScalarField dnGradient[3];
		if(optimized) for(int i=0;i<3;i++) dnGradient[i]=frozenBlochDerivative(dnTilde,*q,i);
		complexScalarField dsigma;
		nullToZero(dsigma,e.gInfo);
		initZero(dsigma);
		for(int i=0;i<3;i++) {
			// Bind as const: a conditional expression creates a temporary
			// shared field pointer that destructive arithmetic could consume.
			const complexScalarField& gradient=optimized ? dnGradient[i] : blochDerivative(dn[0],*q,i);
			dsigma += 2*pInfo.IDJn_cached[i]*gradient;
		}
		dVxc += pInfo.e_nsigma_cached*dsigma;
		for(int i=0;i<3;i++)
		{
			const complexScalarField& gradient=optimized ? dnGradient[i] : blochDerivative(dn[0],*q,i);
			complexScalarField term = pInfo.e_sigma_cached*gradient
				+ pInfo.IDJn_cached[i]*(pInfo.e_nsigma_cached*dn[0]+pInfo.e_sigmasigma_cached*dsigma);
			dVxc -= 2*(optimized ? frozenBlochDerivative(J(term),*q,i) : blochDerivative(term,*q,i));
		}
	}
	for(unsigned s=0;s<dVscloc.size();s++)
		dVscloc[s] = JdagOJ(dVxc) + Jdag(O(dVsclocTilde),true);

	if(!pInfo.fullMeshResponse) e.symm.symmetrize(dVscloc);
	watch.stop();
}


void PerturbationSolver::getdVsclocTau(ScalarFieldArray& dVscloc, ScalarFieldArray* dn)
{   static StopWatch watch("getdVsclocTau"); watch.start();
	
	nullToZero(dVscloc, e.gInfo);
	initZero(dVscloc);
	
	assert(!e.exCorr.orbitalDep);
		
	if (pInfo.datom && pInfo.datom->isUltrasoft(iInfo)) {
		assert(dn);
		getdVsclocPsi(*dn, dVscloc);
	}
	
	for(unsigned s=0; s<eVars.Vscloc.size(); s++) {
		if (pInfo.dVext) dVscloc[s] += JdagOJ(pInfo.dVext->dVext[s]);
		if (pInfo.drhoExt) dVscloc[s] += Jdag(O((*e.coulomb)(pInfo.drhoExt->drhoExt)), true);
		if (pInfo.dElectricField) {
			CoulombParams params = e.coulombParams;
			params.Efield = pInfo.dElectricField->Efield;
			pInfo.dElectricField->C = params.createCoulomb(e.gInfo, e.gInfoWfns, e.coulombWfns);
			dVscloc[s] += JdagOJ(pInfo.dElectricField->C->getEfieldPotential());
		}
	}
		
		
	if (pInfo.datom) {
		ScalarField dVlocps, drhoIon, dnCore;
		nullToZero(dVlocps, e.gInfo);
		nullToZero(drhoIon, e.gInfo);
		
		assert(!iInfo.nChargeball);
		
		ScalarFieldTilde rhoIon, nCoreTilde, tauCoreTilde, nChargeball;
		nullToZero(pInfo.datom->Vlocps, e.gInfo);
		initZero(pInfo.datom->Vlocps);
		nullToZero(rhoIon, e.gInfo);
		
		AtomicMode m = pInfo.datom->mode;
		auto sp = iInfo.species[m.sp];
		sp->updateLocal(pInfo.datom->Vlocps, rhoIon, nChargeball, nCoreTilde, tauCoreTilde, m.at);
		
		ScalarFieldTilde derivative=-D(pInfo.datom->Vlocps,m.dirCartesian)
			+(*e.coulomb)(-D(rhoIon,m.dirCartesian),Coulomb::PointChargeRight);
		// Retain the total undisplaced potential for commensurate response bookkeeping.
		pInfo.datom->Vlocps += (*(e.coulomb))(rhoIon, Coulomb::PointChargeRight);
		assert(!(iInfo.computeStress and iInfo.ionWidth));
		//rhoIon = gaussConvolve(rhoIon, ionWidth);
		
		assert(!tauCoreTilde);
		
		//Process partial core density:
		if(nCoreTilde) {
			dnCore = -I(D(nCoreTilde, m.dirCartesian));
			ScalarFieldArray dVxc(eVars.Vxc.size());
			e.exCorr.get_dVxc(eVars.get_nXC(), &dVxc, false, &eVars.tau, &eVars.Vtau, getdnXC(dnCore));
			for(unsigned s=0; s<eVars.Vscloc.size(); s++)
				dVscloc[s] += JdagOJ(dVxc[s]);
			pInfo.dnCoreA = dnCore;
		}
		
		for(unsigned s=0; s<eVars.Vscloc.size(); s++)
			dVscloc[s] += Jdag(O(derivative), true);
	}

	e.symm.symmetrize(dVscloc);
	watch.stop();
}

void PerturbationSolver::getdVsclocTau(complexScalarFieldArray& dVscloc, int qsign)
{	static StopWatch watch("getdVsclocTau"); watch.start();

	assert(!pInfo.commensurate);
	
	nullToZero(dVscloc, e.gInfo);
	initZero(dVscloc);

	for(unsigned s=0; s<eVars.Vscloc.size(); s++)
	{
		vector3<> qvec = pInfo.qvec*qsign;
		if(pInfo.datom)
		{
			const AtomicMode& m = pInfo.datom->mode;
			auto sp = iInfo.species[m.sp];
			dVscloc[s] += Jdag(O(sp->getLocalPotential(qvec,m.at,&m.dirCartesian)),true);
		}

		
		if (pInfo.dVext) {
			if (qsign == 1) dVscloc[s] += JdagOJ(pInfo.dVext->dVextpq[s]);
			else if (qsign == -1) dVscloc[s] += JdagOJ(pInfo.dVext->dVextmq[s]);
		}
		
		if (pInfo.drhoExt) {
			if (qsign == 1) dVscloc[s] += Jdag(O(e.coulomb->applyBloch(pInfo.drhoExt->drhoExtpq,qvec)), true);
			else if (qsign == -1) dVscloc[s] += Jdag(O(e.coulomb->applyBloch(pInfo.drhoExt->drhoExtmq,qvec)), true);
		}
	}

	if(!pInfo.fullMeshResponse) e.symm.symmetrize(dVscloc);
	watch.stop();
}

ScalarFieldArray PerturbationSolver::getdnXC(const ScalarField dnCore) const
{
	ScalarFieldArray dnXC(eVars.n.size());
	nullToZero(dnXC, e.gInfo);
	if(e.iInfo.nCore)
	{
		int nSpins = std::min(int(dnXC.size()), 2); //1 for unpolarized and 2 for polarized
		for(int s=0; s<nSpins; s++) //note that off-diagonal components of spin-density matrix are excluded
		{	initZero(dnXC[s]);
			dnXC[s] += (1./nSpins) * dnCore; //add core density
		}
	}
	
	return dnXC;
}
