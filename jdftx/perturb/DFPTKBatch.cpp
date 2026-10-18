/* Streamed fixed-potential phonon responses. GPL v3 or later; part of JDFTx. */
#include <perturb/PerturbationSolver.h>
#include <perturb/DFPTGroundState.h>
#include <electronic/Everything.h>
#include <core/Operators.h>

std::vector<complex> PerturbationSolver::solvePhononKBatch(
	const std::vector<std::shared_ptr<AtomPerturbation>>& modes)
{
	assert(pInfo.phononKBatchSize>0 && !pInfo.commensurate);
	const auto sourceAtom=pInfo.datom;
	const double weight=eInfo.qnums.front().weight;
	for(const auto& k:eInfo.qnums) if(fabs(k.weight-weight)>1e-12 || weight<=0.)
		die("Streamed DFPT requires uniform full-mesh electronic weights.\n");
	int batches=(eInfo.qStop-eInfo.qStart+pInfo.phononKBatchSize-1)/pInfo.phononKBatchSize;
	mpiWorld->allReduce(batches,MPIUtil::ReduceMax);
	const bool gamma=metallic() && pInfo.qvec.length_squared()<1e-20;
	kReductionActive=false; physicalResponse=false; pInfo.oneSidedResponse=false;
	streamedUnprojected=true;
	updateExcorrCache();
	logPrintf("DFPT streamed k response: %d batches, at most %d k points/rank; %s.\n",
		batches,pInfo.phononKBatchSize,gamma ? "global fixed-number Gamma projection" : "full k integration");
	// Reset only the large per-batch state. XC derivatives and the ground state
	// stay frozen throughout all sweeps of this displacement.
	auto prepare=[&](int batch) {
		releasePhononResponse(); metallicReady=false;
		pInfo.phononKBatchStart=eInfo.qStart+batch*pInfo.phononKBatchSize;
		pInfo.phononKBatchStop=std::min(eInfo.qStop,pInfo.phononKBatchStart+pInfo.phononKBatchSize);
		if(pInfo.phononGroundState) pInfo.phononGroundState->loadBatch(batch);
		pInfo.datom=sourceAtom;
		pInfo.setup(e,eVars);
		computeIncommensurateWfns(); pInfo.offsetStatesReady=true;
		if(metallic()) setupMetallicResponse();
		state.init(e);
		pInfo.initInc(state.X,2*eInfo.nStates,eInfo.nBands,&eInfo);
		for(auto& X:state.X) if(X) X.zero();
		pInfo.dGradTau.init(e); pInfo.dGradPsi.init(e);
		calcdGradTau();
	};
	auto chargeVector=[&](int k,int sign) {
		int index=k+sign*eInfo.nStates;
		matrix t=pInfo.Cinc[index]^O(eVars.C[k]);
		complex* data=t.data(); const complex* W=metallicWeights[index].data();
		for(int i=0;i<t.nRows()*t.nCols();i++) data[i]*=W[i];
		return t;
	};
	auto addCharge=[&](const std::vector<ColumnBundle>& X,complex* charge,double* norm) {
		if(!gamma) return;
		for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<2;sign++) {
			int index=k+sign*eInfo.nStates;
			matrix t=chargeVector(k,sign);
			charge[sign]+=weight*trace(dagger(t)*(pInfo.Cinc[index]^O(X[index])));
			if(norm) norm[sign]+=weight*std::pow(nrm2(t),2);
		}
	};
	auto projectedNorm=[&](double raw,const complex* charge,const double* norm,bool weighted) {
		if(gamma) for(int sign=0;sign<2;sign++) if(norm[sign]>1e-30)
			// charge/norm use the overlap metric O = volume for NC states;
			// PerturbationGradient::dot uses the Euclidean PW coefficients.
			raw-=2.*(weighted ? weight : 1./weight)*charge[sign].norm()/(e.gInfo.detR*norm[sign]);
		return std::max(0.,raw);
	};
	complexScalarFieldArray potential(eVars.n.size()); nullToZero(potential,e.gInfo); initZero(potential);
	std::vector<complexScalarFieldArray> inputs,outputs,errors;
	std::vector<complex> forces(modes.size());
	complex chargeSolution[2]; double chargeNorm[2]={0.,0.};
	double verified=INFINITY;
	for(int outer=0;outer<pInfo.phononMixIterations;outer++) {
		complexScalarFieldArray density(eVars.n.size()); nullToZero(density,e.gInfo); initZero(density);
		complexScalarFieldArray densityCharge[2];
		for(auto& d:densityCharge) { d.resize(eVars.n.size()); nullToZero(d,e.gInfo); initZero(d); }
		complex chargeBare[2]={},chargeRaw[2]={};
		chargeNorm[0]=chargeNorm[1]=0.;
		double sourceNorm=0.,innerNorm=0.;
		// No response wavefunction survives a k batch. All batches see exactly
		// the same input potential; screening is updated only after this sweep.
		for(int batch=0;batch<batches;batch++) {
			prepare(batch);
			PerturbationGradient rhs=clone(sourceGradient());
			sourceNorm+=dot(rhs,rhs);
			// addCharge includes a weight. rhs already contains that weight.
			complex c[2]={}; addCharge(rhs.X,c,chargeNorm);
			for(int sign=0;sign<2;sign++) chargeBare[sign]+=c[sign]/weight;
			axpy(1.,screeningSource(potential),rhs);
			SolverParams inner=pInfo.solverParams; inner.algorithm=SolverParams::MINRES;
			inner.residualDiffThreshold=0.;
			inner.residualTol=std::min(1e-2,0.01*pInfo.solverParams.residualTol
				*sqrt(std::max(1e-300,dot(sourceGradient(),sourceGradient()))/std::max(1e-300,dot(rhs,rhs))));
			screeningEnabled=false;
			if(!pInfo.independentBandCG || !solveIndependentBands(rhs,inner)) solve(rhs,inner);
			PerturbationGradient residual; hessian(residual,state); axpy(1.,rhs,residual);
			innerNorm+=dot(residual,residual);
			addCharge(state.X,chargeRaw,nullptr);
			std::vector<ColumnBundle> physical(2*eInfo.nStates);
			for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<2;sign++) {
				int index=k+sign*eInfo.nStates;
				const ColumnBundle& C=pInfo.Cinc[index];
				if(metallic()) physical[index]=metallicMap(state.X[index],k,sign);
				else { physical[index]=state.X[index]; physical[index]-=C*(C^O(state.X[index])); }
				if(gamma) {
					ColumnBundle tangent=C*chargeVector(k,sign);
					tangent=metallicMap(tangent,k,sign);
					// Keep the two signs separate until their global charge
					// coefficients are known. Norm-conserving scalar states only.
					densityCharge[sign]+=weight*(sign
						? diagouterI(eye(eInfo.nBands),eVars.C[k],tangent,density.size(),&e.gInfo)
						: diagouterI(eye(eInfo.nBands),tangent,eVars.C[k],density.size(),&e.gInfo));
				}
			}
			getdnInc(&physical,nullptr,pInfo.dnpq,pInfo.dnmq);
			density+=pInfo.dnpq;
			ManagedMemoryBase::reportAllocationCache("streamed k batch complete");
		}
		if(gamma) {
			for(int sign=0;sign<2;sign++) {
				mpiWorld->allReduce(chargeBare[sign],MPIUtil::ReduceSum);
				mpiWorld->allReduce(chargeRaw[sign],MPIUtil::ReduceSum);
				mpiWorld->allReduce(chargeNorm[sign],MPIUtil::ReduceSum);
				for(auto& field:densityCharge[sign]) field->allReduceData(mpiWorld,MPIUtil::ReduceSum);
				chargeSolution[sign]=chargeNorm[sign]>1e-30 ? chargeRaw[sign]/chargeNorm[sign] : complex();
				// The minus branch enters density as a conjugated wavefunction.
				for(size_t s=0;s<density.size();s++) {
					complex coefficient=sign ? conj(chargeSolution[sign]) : chargeSolution[sign];
					complex* data=density[s]->data(); const complex* tangent=densityCharge[sign][s]->data();
					for(int i=0;i<e.gInfo.nr;i++) data[i]-=coefficient*tangent[i];
				}
			}
		}
		sourceNorm=projectedNorm(sourceNorm,chargeBare,chargeNorm,true);
		complexScalarFieldArray output(eVars.n.size()); getdVsclocPsi(density,output,&pInfo.qvec);
		auto difference=output-potential;
		// A second sweep evaluates the original coupled residual, bounded by
		// ||inner residual|| + ||screening source(Vout - Vin)||. The Gamma
		// projection is orthogonal for uniform k weights and H0 is identity
		// on the charge tangent, so it cannot increase the inner residual norm.
		double screeningNorm=0.; complex chargeScreen[2]={};
		for(int batch=0;batch<batches;batch++) {
			prepare(batch);
			auto source=screeningSource(difference);
			screeningNorm+=dot(source,source);
			complex c[2]={}; addCharge(source.X,c,nullptr);
			for(int sign=0;sign<2;sign++) chargeScreen[sign]+=c[sign]/weight;
		}
		if(gamma) for(int sign=0;sign<2;sign++) mpiWorld->allReduce(chargeScreen[sign],MPIUtil::ReduceSum);
		screeningNorm=projectedNorm(screeningNorm,chargeScreen,chargeNorm,true);
		verified=(sqrt(std::max(0.,innerNorm))+sqrt(screeningNorm))/sqrt(std::max(1e-300,sourceNorm));
		logPrintf("DFPT streamed iteration %d: coupled relative residual bound %.6le.\n",outer+1,verified);
		if(!std::isfinite(verified)) die("Nonfinite streamed DFPT residual.\n");
		if(verified<=pInfo.solverParams.residualTol) {
			if(gamma) logPrintf("DFPT streamed Gamma projection coefficients: %.12le %+.12lei; %.12le %+.12lei.\n",
				chargeSolution[0].real(),chargeSolution[0].imag(),chargeSolution[1].real(),chargeSolution[1].imag());
			// Repeat only the fixed-potential solve, contracting each physical
			// response before freeing it. This avoids retaining a full k mesh.
			for(int batch=0;batch<batches;batch++) {
				prepare(batch);
				PerturbationGradient rhs=clone(sourceGradient()); axpy(1.,screeningSource(potential),rhs);
				SolverParams inner=pInfo.solverParams; inner.algorithm=SolverParams::MINRES;
				inner.residualDiffThreshold=0.; inner.residualTol=0.001*pInfo.solverParams.residualTol
					*sqrt(std::max(1e-300,dot(sourceGradient(),sourceGradient()))/std::max(1e-300,dot(rhs,rhs)));
				screeningEnabled=false;
				if(!pInfo.independentBandCG || !solveIndependentBands(rhs,inner)) solve(rhs,inner);
				std::vector<ColumnBundle> physical(2*eInfo.nStates);
				for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<2;sign++) {
					int index=k+sign*eInfo.nStates; const ColumnBundle& C=pInfo.Cinc[index];
					ColumnBundle X=state.X[index];
					if(gamma) X-=C*(chargeVector(k,sign)*chargeSolution[sign]);
					if(metallic()) physical[index]=metallicMap(X,k,sign);
					else { physical[index]=X; physical[index]-=C*(C^O(X)); }
				}
				for(size_t b=0;b<modes.size();b++) {
					pInfo.datom=modes[b];
					for(int k=eInfo.qStart;k<eInfo.qStop;k++) if(activeResponse(k)) for(int sign=0;sign<2;sign++) {
						int index=k+sign*eInfo.nStates;
						ColumnBundle derivative=pInfo.Cinc[index].similar(); derivative.zero();
						dHtau(*derivative.qnum,derivative,eVars.C[k],sign ? -1 : 1);
						complex value=weight*dotc(derivative,physical[index]);
						forces[b]+=sign ? conj(value) : value;
					}
				}
			}
			mpiWorld->allReduceData(forces,MPIUtil::ReduceSum);
			releasePhononResponse(); metallicReady=false;
			if(pInfo.phononGroundState) pInfo.phononGroundState->release();
			pInfo.phononKBatchStart=pInfo.phononKBatchStop=-1;
			streamedUnprojected=false; screeningEnabled=true; pInfo.datom=sourceAtom;
			pInfo.dVsclocpq=std::move(output); pInfo.dVsclocmq=conj(pInfo.dVsclocpq);
			pInfo.lastResponseResidual=verified;
			logPrintf("DFPT final relative residual bound: %.6le (streamed k batches).\n",verified);
			return forces;
		}
		inputs.push_back(clone(potential)); outputs.push_back(std::move(output)); errors.push_back(std::move(difference));
		if(int(errors.size())>pInfo.phononMixHistory) { inputs.erase(inputs.begin()); outputs.erase(outputs.begin()); errors.erase(errors.begin()); }
		int count=errors.size(); matrix gram=zeroes(count,count),ones=zeroes(count,1);
		double scale=0.;
		for(int i=0;i<count;i++) {
			ones.set(i,0,1.);
			for(int j=0;j<=i;j++) {
				double value=0.; for(size_t s=0;s<potential.size();s++) value+=dot(errors[i][s],errors[j][s]).real();
				gram.set(i,j,value); gram.set(j,i,value);
			}
			scale=std::max(scale,gram(i,i).real());
		}
		for(int i=0;i<count;i++) gram.set(i,i,gram(i,i)+std::max(1e-30,1e-10*scale));
		matrix coefficients=invApply(gram,ones); double total=0.;
		for(int i=0;i<count;i++) total+=coefficients(i,0).real();
		bool valid=std::isfinite(total) && total>0.;
		for(int i=0;i<count;i++) valid &= std::isfinite(coefficients(i,0).real()) && fabs(coefficients(i,0).real()/total)<20.;
		initZero(potential);
		if(valid) for(int i=0;i<count;i++) {
			double c=coefficients(i,0).real()/total;
			axpy(c*(1.-pInfo.phononMixFraction),inputs[i],potential); axpy(c*pInfo.phononMixFraction,outputs[i],potential);
		} else {
			axpy(1.-pInfo.phononMixFraction,inputs.back(),potential); axpy(pInfo.phononMixFraction,outputs.back(),potential);
			inputs.clear(); outputs.clear(); errors.clear();
		}
	}
	die("Streamed DFPT did not converge; increase phonon-dfpt-mixing iterations.\n");
	return forces;
}
