/* On-demand reconstruction of IBZ ground states. GPL v3 or later; part of JDFTx. */
#include <perturb/DFPTGroundState.h>
#include <electronic/Everything.h>
#include <electronic/ElecMinimizer.h>
#include <electronic/ColumnBundleTransform.h>
#include <core/Operators.h>

DFPTGroundState::DFPTGroundState(Everything& e,std::shared_ptr<Everything> source,
	std::vector<int> owners,std::vector<int> inverts,std::vector<int> operations)
: e(e),source(std::move(source)),owners(std::move(owners)),inverts(std::move(inverts)),operations(std::move(operations)) {}

void DFPTGroundState::prepare(const MPIUtil* parent)
{
	if(ready) return;
	auto& s=*source;
	logPrintf("DFPT IBZ ground-state preparation: %d reduced / %d full-mesh states.\n",s.eInfo.nStates,e.eInfo.nStates);
	logFlush();
	if(s.cntrl.fixed_H) {
		logPrintf("Using saved NSCF states in their fixed converged density; no SCF mesh change.\n");
		logFlush();
		s.eVars.EdensityAndVscloc(s.ener);
	} else elecFluidMinimize(s);
	if(killFlag) die("IBZ ground-state preparation interrupted; no phonon response was started.\n");
	if(!s.cntrl.fixed_H) s.eVars.elecEnergyAndGrad(s.ener,nullptr,nullptr,true);
	// Frozen-density restarts can validate before expensive band refinement.
	auto transferHamiltonian = [&]() {
		if(!(s.gInfo.S==e.gInfo.S) || s.exCorr.getName()!=e.exCorr.getName())
			die("IBZ ground-state storage requires identical FFT grids and functionals.\n");
		// Allocate density on the target GridInfo; source GridInfo remains private.
		for(size_t spin=0;spin<e.eVars.n.size();spin++) {
			e.eVars.n[spin]=ScalarFieldData::alloc(e.gInfo);
			std::copy(s.eVars.n[spin]->data(),s.eVars.n[spin]->data()+e.gInfo.nr,e.eVars.n[spin]->data());
		}
		e.ener=s.ener;
		e.eVars.EdensityAndVscloc(e.ener);
		// The saved IBZ Hamiltonian is projected onto the source crystal
		// symmetries. Apply the same projection on the full-mesh target:
		// its symmetries are disabled only to enumerate all electronic k points.
		// A frozen density may contain small symmetry-breaking SCF residuals.
		s.symm.symmetrize(e.eVars.Vscloc);
		double potentialError=0.,potentialNorm=0.;
		for(size_t spin=0;spin<e.eVars.Vscloc.size();spin++) {
			ScalarFieldTilde difference=J(e.eVars.Vscloc[spin]-s.eVars.Vscloc[spin]);
			ScalarFieldTilde reference=J(s.eVars.Vscloc[spin]);
			// Non-Cartesian crystal rotations need not preserve unused FFT-box
			// corners. Compare only Fourier components connecting the PW basis.
			complex* d=difference->data(); complex* r=reference->data();
			const auto& S=e.gInfo.S;
			for(int x=0;x<S[0];x++) for(int y=0;y<S[1];y++) for(int z=0;z<=S[2]/2;z++) {
				vector3<int> g(x,y,z);
				for(int j=0;j<3;j++) if(2*g[j]>S[j]) g[j]-=S[j];
				if(dot(g,e.gInfo.GGT*g)>8*e.cntrl.Ecut*(1+1e-12)) {
					int index=(x*S[1]+y)*(S[2]/2+1)+z; d[index]=r[index]=0.;
				}
			}
			potentialError+=dot(difference,difference); potentialNorm+=dot(reference,reference);
			std::copy(s.eVars.Vscloc[spin]->data(),s.eVars.Vscloc[spin]->data()+e.gInfo.nr,e.eVars.Vscloc[spin]->data());
		}
		if(!std::isfinite(potentialError) || potentialError>1e-16*std::max(1.,potentialNorm))
			die("IBZ and DFPT ground-state Hamiltonians differ (relative error %.6le); match functional, external potentials and Coulomb settings.\n",
				sqrt(potentialError/std::max(1.,potentialNorm)));
		logPrintf("IBZ/DFPT Hamiltonian relative error: %.6le\n", sqrt(potentialError/std::max(1.,potentialNorm)));
		logFlush();
	};
	if(s.cntrl.fixed_H) transferHamiltonian();
	const bool metallic=e.eInfo.fillingsUpdate==ElecInfo::FillingsHsub;
	if(metallic || s.cntrl.fixed_H) PerturbationSolver(s).computeMetallicGroundStates(!s.cntrl.fixed_H);
	s.eVars.setEigenvectors();
	if(metallic) {
		double Bz=0.,mu=s.eInfo.findMu(s.eVars.Hsub_eigs,s.eInfo.nElectrons,Bz);
		for(int k=s.eInfo.qStart;k<s.eInfo.qStop;k++) {
			s.eVars.Haux_eigs[k]=s.eVars.Hsub_eigs[k];
			s.eVars.F[k]=s.eInfo.smear(mu,s.eVars.Hsub_eigs[k]);
		}
	}
	// Use one band gauge across q groups, including degenerate eigenspaces.
	if(parent) for(int k=0;k<s.eInfo.nStates;k++) {
		int root=s.eInfo.whose(k); parent->bcast(root);
		ColumnBundle remote;
		ColumnBundle& C=s.eInfo.isMine(k) ? s.eVars.C[k] : remote;
		if(!s.eInfo.isMine(k)) C.init(s.eInfo.nBands,s.basis[k].nbasis,&s.basis[k],&s.eInfo.qnums[k],isGpuEnabled());
		diagMatrix eps=s.eInfo.isMine(k) ? s.eVars.Hsub_eigs[k] : diagMatrix(s.eInfo.nBands);
		diagMatrix f=s.eInfo.isMine(k) ? s.eVars.F[k] : diagMatrix(s.eInfo.nBands);
		parent->bcastData(C,root); parent->bcastData(eps,root); parent->bcastData(f,root);
		if(s.eInfo.isMine(k)) {
			s.eVars.Hsub_eigs[k]=eps; s.eVars.F[k]=f;
			if(metallic) s.eVars.Haux_eigs[k]=eps;
			s.iInfo.project(C,s.eVars.VdagC[k]);
		}
	}
	if(!s.cntrl.fixed_H) {
		s.eVars.elecEnergyAndGrad(s.ener,nullptr,nullptr,false);
		transferHamiltonian();
	}
	e.ener=s.ener;
	double tail=0.;
	for(int k=0;k<s.eInfo.nStates;k++) {
		diagMatrix eps=s.eInfo.isMine(k) ? s.eVars.Hsub_eigs[k] : diagMatrix(s.eInfo.nBands);
		diagMatrix f=s.eInfo.isMine(k) ? s.eVars.F[k] : diagMatrix(s.eInfo.nBands);
		mpiWorld->bcastData(eps,s.eInfo.whose(k)); mpiWorld->bcastData(f,s.eInfo.whose(k));
		tail=std::max(tail,f.back());
		for(int target=e.eInfo.qStart;target<e.eInfo.qStop;target++) if(owners[target]==k) {
			e.eVars.F[target]=f; e.eVars.Hsub_eigs[target]=eps;
			if(metallic) e.eVars.Haux_eigs[target]=eps;
		}
	}
	if(metallic && tail>1e-6) die("Metallic DFPT needs more empty bands: highest-band occupation exceeds 1e-6.\n");
	// Persistent wavefunctions stay on the host even in the CUDA build. Each
	// reconstruction transfers only one representative at a time to the GPU.
	size_t bytes=0;
	for(int k=s.eInfo.qStart;k<s.eInfo.qStop;k++) { s.eVars.C[k].data(); bytes+=s.eVars.C[k].nData()*sizeof(complex); }
	s.eVars.VdagC.clear(); s.eVars.Hsub.clear(); s.eVars.Hsub_evecs.clear();
	for(auto& sp:s.iInfo.species) sp->clearProjectorCache();
	e.eVars.isRandom=false; e.eVars.HauxInitialized=metallic;
	e.pertInfo.groundStateReady=true;
	ready=true;
	logPrintf("DFPT IBZ storage ready: %d/%d states; %.3f MiB retained on group-head host; no full-mesh wavefunctions allocated.\n",
		s.eInfo.nStates,e.eInfo.nStates,bytes/(1024.*1024.));
	logPrintf("DFPT active ground-state limit: %d k points per rank.\n",e.pertInfo.phononKBatchSize);
	ManagedMemoryBase::reportAllocationCache("IBZ storage ready");
	logFlush();
}

int DFPTGroundState::batches() const
{
	const auto& ei=e.eInfo;
	int count=(ei.qStop-ei.qStart+e.pertInfo.phononKBatchSize-1)/e.pertInfo.phononKBatchSize;
	mpiWorld->allReduce(count,MPIUtil::ReduceMax);
	return count;
}

void DFPTGroundState::release()
{
	for(int k=e.eInfo.qStart;k<e.eInfo.qStop;k++) {
		e.eVars.C[k].free(); e.eVars.Hsub[k]=matrix();
		for(auto& projection:e.eVars.VdagC[k]) projection=matrix();
	}
}

void DFPTGroundState::loadBatch(int batch)
{
	assert(ready);
	release();
	const auto& ei=e.eInfo;
	int start=ei.qStart+batch*e.pertInfo.phononKBatchSize,stop=std::min(ei.qStop,start+e.pertInfo.phononKBatchSize);
	std::vector<std::pair<int,ColumnBundle*>> requests;
	for(int k=start;k<stop;k++) {
		e.eVars.C[k].init(ei.nBands,e.basis[k].nbasis,&e.basis[k],&ei.qnums[k],isGpuEnabled());
		e.eVars.Hsub[k]=e.eVars.Hsub_eigs[k];
		requests.emplace_back(k,&e.eVars.C[k]);
	}
	reconstruct(requests);
}

void DFPTGroundState::reconstruct(const std::vector<std::pair<int,ColumnBundle*>>& requests)
{
	assert(ready);
	auto& s=*source;
	std::vector<int> needed(s.eInfo.nStates,0);
	for(const auto& r:requests) { assert(r.first>=0 && r.first<e.eInfo.nStates); needed[owners[r.first]]=1; }
	mpiWorld->allReduceData(needed,MPIUtil::ReduceMax);
	for(int k=0;k<s.eInfo.nStates;k++) if(needed[k]) {
		ColumnBundle received(s.eInfo.nBands,s.basis[k].nbasis,&s.basis[k],&s.eInfo.qnums[k],false);
		if(s.eInfo.isMine(k)) std::copy(s.eVars.C[k].data(),s.eVars.C[k].data()+received.nData(),received.data());
		mpiWorld->bcastData(received,s.eInfo.whose(k));
		for(const auto& r:requests) if(owners[r.first]==k) {
			auto& target=*r.second; target.zero();
			ColumnBundleTransform::BasisWrapper wrapper(*target.basis);
			ColumnBundleTransform(s.eInfo.qnums[k].k,s.basis[k],target.qnum->k,wrapper,1,
				s.symm.getMatrices()[operations[r.first]],inverts[r.first]).scatterAxpy(1.,received,target,0,1);
		}
	}
}

void DFPTGroundState::writeWavefunctions(const char* filename)
{
	FILE* fp=mpiWorld->isHead() ? fopen(filename,"wb") : nullptr;
	if(mpiWorld->isHead() && !fp) die_alone("Cannot write streamed DFPT wavefunctions.\n");
	for(int k=0;k<e.eInfo.nStates;k++) {
		ColumnBundle C; std::vector<std::pair<int,ColumnBundle*>> requests;
		if(mpiWorld->isHead()) {
			C.init(e.eInfo.nBands,e.basis[k].nbasis,&e.basis[k],&e.eInfo.qnums[k],isGpuEnabled());
			requests.emplace_back(k,&C);
		}
		reconstruct(requests);
		if(fp && fwrite(C.data(),sizeof(complex),C.nData(),fp)!=C.nData()) die_alone("Error writing streamed DFPT wavefunctions.\n");
	}
	if(fp && fclose(fp)) die_alone("Error closing streamed DFPT wavefunctions.\n");
}
