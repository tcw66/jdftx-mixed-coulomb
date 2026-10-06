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
#include <perturb/TestPerturbation.h>
#include <electronic/Everything.h>
#include <electronic/ElecMinimizer.h>
#include <electronic/ColumnBundle.h>
#include <electronic/ColumnBundleTransform.h>
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

	for(int q=x.eInfo->qStart; q<x.eInfo->qStop; q++)
	{
		if(x.X[q] && y.X[q]) result += dotc(x.X[q], y.X[q]).real()*2.0;
		if (!x.pInfo->commensurate) if(x.X[q+nStates] && y.X[q+nStates]) result += dotc(x.X[q+nStates], y.X[q+nStates]).real()*2.0;
	}

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
		if (!pInfo->commensurate) if(x[q+nStates] && y[q+nStates]) result += dotc(x[q+nStates], y[q+nStates]).real()*2.0;
	}

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

void PerturbationSolver::prepareKReduction()
{
	kReductionActive=false;
	physicalResponse=false;
	kSymmetries.clear(); kCharacters.clear(); densitySymmetryMaps.clear();
	if(!pInfo.phononReduceK || !pInfo.phononGrid.isNonzero() || pInfo.commensurate
		|| !pInfo.datom || pInfo.phononElectronPhonon || pInfo.testing) return;
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
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) for(int sign=0;sign<2;sign++) {
		int index=k+sign*eInfo.nStates;
		complex value;
		if(physicalResponse) {
			ColumnBundle derivative=pInfo.Cinc[index].similar(); derivative.zero();
			dHtau(*derivative.qnum,derivative,eVars.C[k],sign ? -1 : 1);
			value=eInfo.qnums[k].weight*dotc(derivative,pInfo.dC[index]);
		} else value=dotc(pInfo.dGradTau.X[index],pInfo.dC[index]);
		response+=sign ? conj(value) : value;
	}
	mpiWorld->allReduce(response,MPIUtil::ReduceSum);
	return response;
}

PerturbationGradient PerturbationSolver::screeningSource(const complexScalarFieldArray& potential)
{
	PerturbationGradient result; result.init(e);
	complexScalarFieldArray negative=conj(potential);
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<2;sign++) {
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
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<2;sign++)
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

ColumnBundle PerturbationSolver::bareBandPrecondition(const ColumnBundle& X,int k,int sign)
{
	const ColumnBundle& C=pInfo.Cinc[k+sign*eInfo.nStates];
	ColumnBundle low=C*(C^O(X)), result=X-low;
	precond_inv_kinetic(result,1.);
	result-=C*(C^O(result));
	if(metallic()) result+=low;
	return result;
}

bool PerturbationSolver::solveIndependentBands(const PerturbationGradient& rhs,const SolverParams& parameters)
{
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
	for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<2;sign++) {
		int index=k+sign*eInfo.nStates;
		Block block; block.k=k; block.sign=sign;
		block.residual=rhs.X[index]; block.residual*=(-1./responseWeight(k));
		block.solution=block.residual.similar(); block.solution.zero();
		block.direction=bareBandPrecondition(block.residual,k,sign);
		block.rho=diagDot(block.residual,block.direction); block.reference=block.rho;
		for(double rho:block.rho) referenceNorm+=std::max(0.,rho)*std::pow(responseWeight(k),2);
		totalBands+=eInfo.nBands;
		for(int band=0;band<eInfo.nBands;band++) block.bands.push_back(band);
		blocks.push_back(std::move(block));
	}
	mpiWorld->allReduce(referenceNorm,MPIUtil::ReduceSum);
	mpiWorld->allReduce(totalBands,MPIUtil::ReduceSum);
	// Use a common absolute residual budget, as in QE. Relative tests against
	// each tiny empty-band source would oversolve bands with negligible weight.
	for(auto& block:blocks) for(double& reference:block.reference)
		reference=referenceNorm/(std::max(1,totalBands)*std::pow(responseWeight(block.k),2));
	auto select=[](const ColumnBundle& X,const std::vector<int>& indices) {
		ColumnBundle result=X.similar(indices.size());
		for(size_t i=0;i<indices.size();i++) result.setSub(i,X.getSub(indices[i],indices[i]+1));
		return result;
	};
	int applications=0;
	for(int iteration=0;iteration<parameters.nIterations;iteration++) {
		bool failed=false,unfinished=false;
		for(auto& block:blocks) {
			if(block.bands.empty()) continue;
			std::vector<int> retain;
			for(size_t i=0;i<block.bands.size();i++) {
				if(block.rho[i]<=parameters.residualTol*parameters.residualTol*block.reference[i] || block.reference[i]<=1e-30)
					state.X[block.k+block.sign*eInfo.nStates].setSub(block.bands[i],block.solution.getSub(i,i+1));
				else retain.push_back(i);
			}
			if(retain.size()!=block.bands.size()) {
				std::vector<int> bands; diagMatrix rho(retain.size()), reference(retain.size());
				for(size_t i=0;i<retain.size();i++) { bands.push_back(block.bands[retain[i]]); rho[i]=block.rho[retain[i]]; reference[i]=block.reference[retain[i]]; }
				block.bands=std::move(bands); block.rho=std::move(rho); block.reference=std::move(reference);
				if(block.bands.empty()) continue;
				block.solution=select(block.solution,retain); block.residual=select(block.residual,retain); block.direction=select(block.direction,retain);
			}
			unfinished=true;
			ColumnBundle applied=bareBandOperator(block.direction,block.k,block.bands);
			applications+=block.bands.size();
			diagMatrix curvature=diagDot(block.direction,applied), alpha(block.bands.size());
			for(size_t i=0;i<block.bands.size();i++) {
				if(!std::isfinite(curvature[i]) || curvature[i]<=0. || !std::isfinite(block.rho[i]) || block.rho[i]<=0.) { failed=true; break; }
				alpha[i]=block.rho[i]/curvature[i];
			}
			if(failed) break;
			block.solution+=block.direction*alpha; block.residual-=applied*alpha;
			ColumnBundle preconditioned=bareBandPrecondition(block.residual,block.k,block.sign);
			diagMatrix rho=diagDot(block.residual,preconditioned), beta(rho.size());
			for(size_t i=0;i<rho.size();i++) beta[i]=rho[i]/block.rho[i];
			block.direction=preconditioned+block.direction*beta; block.rho=std::move(rho);
		}
		mpiWorld->allReduce(failed,MPIUtil::ReduceLOr);
		if(failed) {
			state*=0.;
			logPrintf("DFPT independent-band CG encountered nonpositive curvature; using MINRES for this inner solve.\n");
			return false;
		}
		mpiWorld->allReduce(unfinished,MPIUtil::ReduceLOr);
		if(!unfinished) {
			mpiWorld->allReduce(applications,MPIUtil::ReduceSum);
			logPrintf("DFPT independent-band solve: %d iterations, %d band Hamiltonian applications.\n",iteration,applications);
			return true;
		}
	}
	state*=0.;
	logPrintf("DFPT independent-band solve reached its iteration limit; using MINRES for this inner solve.\n");
	return false;
}

void PerturbationSolver::solveNestedResponse()
{
	complexScalarFieldArray potential(pInfo.dVsclocpq.size());
	nullToZero(potential,e.gInfo); initZero(potential);
	std::vector<complexScalarFieldArray> inputs, outputs, errors;
	auto potentialDot=[](const complexScalarFieldArray& a,const complexScalarFieldArray& b) {
		double result=0.; for(size_t s=0;s<a.size();s++) result+=dot(a[s],b[s]).real(); return result;
	};
	const double sourceNorm=dot(pInfo.dGradTau,pInfo.dGradTau);
	double relative=1.;
	for(int outer=0;outer<pInfo.phononMixIterations;outer++) {
		PerturbationGradient rhs=clone(pInfo.dGradTau);
		if(outer) axpy(1.,screeningSource(potential),rhs);
		// Solve a correction about the previous *physical* response. MINRES
		// stores preconditioned coordinates internally, so passing its previous
		// physical state directly as a warm start would apply Ksqrt twice.
		PerturbationGradient previous=std::move(state), correctionRhs;
		screeningEnabled=false;
		hessian(correctionRhs,previous); axpy(1.,rhs,correctionRhs);
		state=clone(previous); state*=0.;
		SolverParams inner=pInfo.solverParams;
		inner.algorithm=SolverParams::MINRES;
		inner.residualDiffThreshold=0.;
		inner.residualTol=std::max(0.05*pInfo.solverParams.residualTol,std::min(1e-2,0.1*relative));
		if(!solveIndependentBands(correctionRhs,inner)) solve(correctionRhs,inner);
		axpy(1.,previous,state);
		screeningEnabled=true;
		PerturbationGradient residual;
		hessian(residual,state); axpy(1.,pInfo.dGradTau,residual);
		double norm=dot(residual,residual);
		relative=sourceNorm ? sqrt(norm/sourceNorm) : sqrt(norm);
		logPrintf("DFPT nested iteration %d: coupled relative residual %.6le, inner tolerance %.6le.\n",outer+1,relative,inner.residualTol);
		if(!std::isfinite(relative)) die("Nonfinite nested DFPT response residual.\n");
		if(relative<=pInfo.solverParams.residualTol) return;
		complexScalarFieldArray output=clone(pInfo.dVsclocpq), error=output-potential;
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
	
	logPrintf("Variational perturbation solver is starting.\n");
	
	if (pInfo.datom) {
		AtomicMode m = pInfo.datom->mode;
		if (m.sp >= iInfo.species.size() || m.at >= iInfo.species[m.sp]->atpos.size())
			die("Perturbed atom does not exist! Check species and atom number.\n");
		logPrintf("Perturbing species %d atom %d in direction %g %g %g.\n", m.sp, m.at, m.dirCartesian[0], m.dirCartesian[1], m.dirCartesian[2]);
	}
	
	if (pInfo.dVext)
		logPrintf("Perturbing Vexternal.\n");
	
	if(!pInfo.groundStateReady && e.eVars.wfnsFilename.empty() && !e.eVars.skipWfnsInit) {
		logPrintf("No input wavefunctions given. Performing electronic structure calculation first.\n");
		elecFluidMinimize(e);
	}
	
	// The phonon driver has already built the ground-state density, potential
	// and subspace Hamiltonians. Responses only modify dC/Cinc, not eVars.C.
	if(!pInfo.phononGrid.isNonzero() || !pInfo.groundStateReady)
		eVars.elecEnergyAndGrad(e.ener, nullptr, nullptr, true);
	pInfo.groundStateReady = true;
	if(!pInfo.commensurate && pInfo.wfnsFilename.empty() && !pInfo.offsetStatesReady)
	{ computeIncommensurateWfns(); pInfo.offsetStatesReady=true; }
	if(metallic() && !metallicReady) setupMetallicResponse();
	prepareKReduction();

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
			state.X[q+eInfo.nStates].zero();
		}
	}
	
	if(!pInfo.phononGrid.isNonzero()) eVars.elecEnergyAndGrad(e.ener, nullptr, nullptr, true);

	if(!usePhononCache() || !groundCachesReady) {
		updateExcorrCache();
		updateHC();
		groundCachesReady=true;
	}
	updateNonlocalDerivs();
	
	pInfo.dGradTau.init(e);
	pInfo.dGradPsi.init(e);
	
	calcdGradTau();
	
	SolverParams responseParams=pInfo.solverParams;
	// The band-space and plane-wave blocks have different energy scales;
	// leave margin between the preconditioned and verified Euclidean residual.
	if(metallic()) responseParams.residualTol*=0.05;
	if(pInfo.phononNestedSolver && pInfo.phononGrid.isNonzero()) solveNestedResponse();
	else solve(pInfo.dGradTau, responseParams);
	
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
	axpy(1.,pInfo.dGradTau,residual);
	double bnorm = dot(pInfo.dGradTau,pInfo.dGradTau);
	double rnorm = dot(residual,residual);
	pInfo.lastResponseResidual=bnorm ? sqrt(rnorm/bnorm) : sqrt(rnorm);
	logPrintf("DFPT final relative residual: %.6le\n",bnorm ? sqrt(rnorm/bnorm) : sqrt(rnorm));
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
		for(double eps:eVars.Hsub_eigs[k])
			metallicDOS-=eInfo.qnums[k].weight*eInfo.smearPrime(metallicMu,eps);
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
		if (!pInfo.commensurate) precond_inv_kinetic(v.X[q + eInfo.nStates], 1);
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
		if (!pInfo.commensurate) precond_inv_kinetic(v.X[q + eInfo.nStates], 1, true);
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
			pInfo.dC[kmq_k] = v.X[kmq_k] - O(pInfo.Cinc[kmq]*(pInfo.Cinc[kmq]^v.X[kmq_k]));
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
		for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
			if(!activeResponse(q)) { Av.X[q].free(); Av.X[q+eInfo.nStates].free(); continue; }

			ColumnBundle dHC_kpq_k, dHC_kmq_k,
			HdC_kpq_k, HdC_kmq_k,
			dwGradEq_kpq_k, dwGradEq_kmq_k;

			diagMatrix F = eVars.F[q];
			int kpq_k = q;
			int kmq_k = q + e.eInfo.nStates;
			int kpq = q ;
			int kmq = q + e.eInfo.nStates;

			dHC_kpq_k = pInfo.Cinc[kpq].similar();
			dHC_kmq_k = pInfo.Cinc[kmq].similar();
			dHC_kpq_k.zero();
			dHC_kmq_k.zero();
			
			if(screeningEnabled) dHpsi(pInfo.kplusq_vectors[q], dHC_kpq_k, eVars.C[q], pInfo.dVsclocpq);
			applyH(pInfo.kplusq_vectors[q], F, HdC_kpq_k, pInfo.dC[kpq_k]);
			dwGradEq_kpq_k = dHC_kpq_k;
			dwGradEq_kpq_k += HdC_kpq_k;
			dwGradEq_kpq_k -= O(pInfo.Cinc[kpq]*(pInfo.Cinc[kpq]^dHC_kpq_k));
			dwGradEq_kpq_k -= O(pInfo.dC[kpq_k]*eVars.Hsub[q]);

			if(screeningEnabled) dHpsi(pInfo.kminusq_vectors[q], dHC_kmq_k, eVars.C[q], pInfo.dVsclocmq);
			applyH(pInfo.kminusq_vectors[q], F, HdC_kmq_k, pInfo.dC[kmq_k]);
			dwGradEq_kmq_k = dHC_kmq_k;
			dwGradEq_kmq_k += HdC_kmq_k;
			dwGradEq_kmq_k -= O(pInfo.Cinc[kmq]*(pInfo.Cinc[kmq]^dHC_kmq_k));
			dwGradEq_kmq_k -= O(pInfo.dC[kmq_k]*eVars.Hsub[q]);

			Av.X[kpq_k] = (dwGradEq_kpq_k - O(pInfo.Cinc[kpq]*(pInfo.Cinc[kpq]^dwGradEq_kpq_k)))*F*responseWeight(q);
			Av.X[kmq_k] = (dwGradEq_kmq_k - O(pInfo.Cinc[kmq]*(pInfo.Cinc[kmq]^dwGradEq_kmq_k)))*F*responseWeight(q);
		}
	}
	
	watch.stop();
}

void PerturbationSolver::calcdGradTau() {
	if(usePhononCache() && !kReductionActive) {
		auto cached=sourceCache.find(pInfo.datom.get());
		if(cached!=sourceCache.end()) { pInfo.dGradTau=cached->second; sourceCacheHits++; return; }
	}
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
		for(int q=eInfo.qStart; q<eInfo.qStop; q++) {
			if(!activeResponse(q)) { pInfo.dGradTau.X[q].free(); pInfo.dGradTau.X[q+eInfo.nStates].free(); continue; }
			ColumnBundle dHC_kpq_k, dHC_kmq_k,
			dwGradEq_kpq_k, dwGradEq_kmq_k;
			diagMatrix F = eVars.F[q];

			int kpq_k = q;
			int kmq_k = q + e.eInfo.nStates;
			int kpq = q ;
			int kmq = q + e.eInfo.nStates;

			dHC_kpq_k = pInfo.Cinc[kpq].similar();
			dHC_kmq_k = pInfo.Cinc[kmq].similar();
			dHC_kpq_k.zero();
			dHC_kmq_k.zero();

			dHtau(pInfo.kplusq_vectors[q], dHC_kpq_k, eVars.C[q], 1);
			dwGradEq_kpq_k = dHC_kpq_k - O(pInfo.Cinc[kpq]*(pInfo.Cinc[kpq]^dHC_kpq_k));

			dHtau(pInfo.kminusq_vectors[q], dHC_kmq_k, eVars.C[q], -1);
			dwGradEq_kmq_k = dHC_kmq_k - O(pInfo.Cinc[kmq]*(pInfo.Cinc[kmq]^dHC_kmq_k));

			pInfo.dGradTau.X[kpq_k] = dwGradEq_kpq_k*F*responseWeight(q);
			pInfo.dGradTau.X[kmq_k] = dwGradEq_kmq_k*F*responseWeight(q);
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
	sourceCache.emplace(pInfo.datom.get(),pInfo.dGradTau);
	sourceCacheBytes+=size_t(maximumBytes);
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

void PerturbationSolver::computeMetallicGroundStates()
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
	eVars.elecEnergyAndGrad(e.ener,nullptr,nullptr,true);
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
	if(pInfo.phononGrid.isNonzero() && pInfo.phononReuseOffsetStates) {
		for(int sign=0;sign<2;sign++) for(int k=0;k<eInfo.nStates;k++) {
			const vector3<>& target=sign ? pInfo.kminusq_vectors[k].k : pInfo.kplusq_vectors[k].k;
			for(int source=0;source<eInfo.nStates;source++) {
				vector3<> delta=target-eInfo.qnums[source].k;
				for(int d=0;d<3;d++) delta[d]-=round(delta[d]);
				if(delta.length_squared()<1e-16) {reused[k+sign*eInfo.nStates]=source; break;}
			}
		}
		// Broadcast one source at a time, keeping memory bounded and allowing
		// source and destination k points to belong to different MPI ranks.
		for(int source=0;source<eInfo.nStates;source++) {
			if(std::find(reused.begin(),reused.end(),source)==reused.end()) continue;
			ColumnBundle received;
			ColumnBundle& Csource=eInfo.isMine(source) ? eVars.C[source] : received;
			if(!eInfo.isMine(source)) Csource.init(eInfo.nBands,e.basis[source].nbasis,&e.basis[source],&eInfo.qnums[source],isGpuEnabled());
			if(mpiWorld->nProcesses()>1) mpiWorld->bcastData(Csource,eInfo.whose(source));
			for(int sign=0;sign<2;sign++) for(int k=eInfo.qStart;k<eInfo.qStop;k++) {
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
	for(int i=0;i<3;i++) {
		pInfo.IDJn_cached[i]=I(D(J(nXC[0]),i));
		pInfo.sigma_cached+=pInfo.IDJn_cached[i]*pInfo.IDJn_cached[i];
	}

	e.exCorr.getSecondDerivatives(nXC[0], pInfo.e_nn_cached, pInfo.e_sigma_cached, pInfo.e_nsigma_cached, pInfo.e_sigmasigma_cached, 1e-9, &pInfo.sigma_cached);
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

		if (C) {
			dnpq += responseWeight(q) * diagouterI((metallic() ? eye(eInfo.nBands) : eVars.F[q]), (*C)[q], (*dC)[kmq_k], dnpq.size(), &e.gInfo);
			dnpq += responseWeight(q) * diagouterI((metallic() ? eye(eInfo.nBands) : eVars.F[q]), (*dC)[kpq_k], (*C)[q], dnpq.size(), &e.gInfo);
		} else {
			dnpq += responseWeight(q) * diagouterI((metallic() ? eye(eInfo.nBands) : eVars.F[q]), eVars.C[q], (*dC)[kmq_k], dnpq.size(), &e.gInfo);
			dnpq += responseWeight(q) * diagouterI((metallic() ? eye(eInfo.nBands) : eVars.F[q]), (*dC)[kpq_k], eVars.C[q], dnpq.size(), &e.gInfo);
		}
	}

	for(complexScalarField& ns: dnpq)
	{	nullToZero(ns, e.gInfo);
		ns->allReduceData(mpiWorld, MPIUtil::ReduceSum);
	}

	e.symm.symmetrize(dnpq);
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


void PerturbationSolver::applyH(const QuantumNumber& qnum, const diagMatrix& Fq, ColumnBundle& HCq, const ColumnBundle& Cq)
{	static StopWatch watch("applyH"); watch.start();
	assert(Cq); //make sure wavefunction is available for this state
	
	std::vector<matrix> HVdagCq(e.iInfo.species.size());
	std::vector<matrix> VdagCq(e.iInfo.species.size());
	
	HCq.zero();
	
	iInfo.setE_nAug(pInfo.E_nAug_cached);

	HCq += Idag_DiagV_I(Cq, eVars.Vscloc);

	e.iInfo.project(Cq, VdagCq); //update the atomic projections
	e.iInfo.augmentDensitySphericalGrad(qnum, VdagCq, HVdagCq);

	HCq += (-0.5) * L(Cq);

	//Nonlocal pseudopotentials:

	e.iInfo.EnlAndGrad(qnum, Fq, VdagCq, HVdagCq);
	e.iInfo.projectGrad(HVdagCq, Cq, HCq);
	
	watch.stop();
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
	ColumnBundle HCr = HCq.similar(), HCi = HCq.similar();
	HCr = Idag_DiagV_I(Cq, Real(dVscloc), &HCr);
	HCi = Idag_DiagV_I(Cq, Imag(dVscloc), &HCi);
	HCq = HCr + complex(0,1)*HCi;
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

	// Hartree term:
	complexScalarFieldTilde dVsclocTilde;
	
	assert(e.coulombParams.geometry == CoulombParams::Geometry::Periodic);
	dVsclocTilde = -4*M_PI*Linv(O(dnTilde), q);

	if(e.exCorr.needFiniteDifference_dVxc())
		die("Finite-q response requires an analytic LDA/GGA XC kernel.\n");
	complexScalarField dVxc = pInfo.e_nn_cached*dn[0];
	if(pInfo.e_sigma_cached) // LDA has no gradient-density derivatives.
	{
		complexScalarField dsigma;
		nullToZero(dsigma,e.gInfo);
		initZero(dsigma);
		for(int i=0;i<3;i++)
			dsigma += 2*pInfo.IDJn_cached[i]*blochDerivative(dn[0],*q,i);
		dVxc += pInfo.e_nsigma_cached*dsigma;
		for(int i=0;i<3;i++)
		{
			complexScalarField term = pInfo.e_sigma_cached*blochDerivative(dn[0],*q,i)
				+ pInfo.IDJn_cached[i]*(pInfo.e_nsigma_cached*dn[0]+pInfo.e_sigmasigma_cached*dsigma);
			dVxc -= 2*blochDerivative(term,*q,i);
		}
	}
	for(unsigned s=0;s<dVscloc.size();s++)
		dVscloc[s] = JdagOJ(dVxc) + Jdag(O(dVsclocTilde),true);

	e.symm.symmetrize(dVscloc);
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
		
		//Add long-range part to Vlocps and smoothen rhoIon:
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
			dVscloc[s] += -Jdag(O(D(pInfo.datom->Vlocps, m.dirCartesian)), true);
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
			dVscloc[s] += Jdag(O(ionicDerivative(sp->getLocalPotential(qvec,m.at),qvec,m.dirCartesian)),true);
		}

		
		if (pInfo.dVext) {
			if (qsign == 1) dVscloc[s] += JdagOJ(pInfo.dVext->dVextpq[s]);
			else if (qsign == -1) dVscloc[s] += JdagOJ(pInfo.dVext->dVextmq[s]);
		}
		
		if (pInfo.drhoExt) {
			if (qsign == 1) dVscloc[s] += Jdag(O(-4*M_PI*Linv(O(pInfo.drhoExt->drhoExtpq), &qvec)), true);
			else if (qsign == -1) dVscloc[s] += Jdag(O(-4*M_PI*Linv(O(pInfo.drhoExt->drhoExtmq), &qvec)), true);
		}
	}

	e.symm.symmetrize(dVscloc);
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
