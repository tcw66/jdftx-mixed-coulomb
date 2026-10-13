/* One-shot KI(2), Colonna et al., JCTC 18, 5435 (2022),
 * doi:10.1021/acs.jctc.2c00161, Eqs. 11, 18, 20, 22 and SI Eqs. 5-6.
 * Distributed under the GNU GPL v3 or later, as part of JDFTx.
 */
#include <wannier/WannierMinimizer.h>
#include <core/CoulombPeriodic.h>
#include <perturb/PerturbationSolver.h>
#include <electronic/ColumnBundleFFT.h>
#include <electronic/KMeshCorrelation.h>
#include <core/Units.h>
#include <core/ScalarFieldIO.h>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

namespace
{
// Ground-state projectors may survive across all orbital sources, but offset
// bases are rebuilt per q. Bound retained values and remove stale basis keys.
class KIProjectorCache
{
	IonInfo& ions;
	std::vector<size_t> savedLimits;
public:
	KIProjectorCache(IonInfo& ions):ions(ions) {
		size_t limit=size_t(1024)*1024*1024/std::max(size_t(1),ions.species.size());
		for(auto& species:ions.species) {
			savedLimits.push_back(species->projectorCacheLimit());
			species->clearProjectorCache(); species->setProjectorCacheLimit(limit);
		}
	}
	void clear() { for(auto& species:ions.species) species->clearProjectorCache(); }
	void clearOffsets(const PerturbationInfo& p) {
		for(auto& species:ions.species) {
			for(const auto& basis:p.kplusq_basis) species->clearProjectorCache(&basis);
			for(const auto& basis:p.kminusq_basis) species->clearProjectorCache(&basis);
		}
	}
	~KIProjectorCache() {
		clear();
		for(size_t i=0;i<ions.species.size();i++) ions.species[i]->setProjectorCacheLimit(savedLimits[i]);
	}
};

// A k+q point on the mesh, with the reciprocal lattice shift needed to retain
// the same Wannier gauge: u_(k+q) = exp(-i Gwrap.r) u_p.
struct MeshPoint { int index; vector3<> wrap; };
MeshPoint findMeshPoint(const std::vector<vector3<>>& mesh, const vector3<>& target)
{
	for(size_t p=0; p<mesh.size(); p++)
	{
		vector3<> wrap, delta=target-mesh[p];
		for(int d=0; d<3; d++) wrap[d]=round(delta[d]);
		if((delta-wrap).length_squared()<1e-16) return {int(p), wrap};
	}
	return {-1, vector3<>()};
}

MeshPoint meshPoint(const std::vector<vector3<>>& mesh, const vector3<>& target)
{
	MeshPoint result=findMeshPoint(mesh,target);
	if(result.index<0) die("wannier-koopmans requires a complete uniform k mesh closed under k+q.\n");
	return result;
}

// Only the scalar <delta n, f_Hxc rho> enters screening. Reuse it when the
// *actual* orbital source densities agree under a crystal operation, possibly
// followed by time reversal, modulo a unit Bloch phase. All unscreened matrix
// elements are still integrated directly; no orbital sewing matrix is assumed.
class ScreeningReuse
{
	struct Operation
	{
		std::shared_ptr<IndexArray> gridMap;
		std::vector<MeshPoint> qMap;
		bool conjugate;
	};
	struct Entry
	{
		int orbital, iq;
		double occupation, norm, bare, response;
		complexScalarField rho;
	};
	const Everything& e;
	bool enabled;
	bool verbose;
	bool diagnostics;
	bool quiet;
	double tolerance;
	std::vector<std::unique_ptr<Operation>> operations;
	std::vector<Entry> entries;
	std::vector<std::vector<int>> byQ;
	size_t cacheBytes=0, peakCacheBytes=0;
	double cacheLimit;
	int solved=0, reused=0;
	double maxError=0.;
	int curvatureRejected=0;
	double maxCurvatureError=0.;

	void addOperation(const SpaceGroupOp& op, bool conjugate, const std::vector<vector3<>>& qmesh)
	{
		const auto& g=e.gInfo;
		const SpaceGroupOp inverse=op.inv();
		auto operation=std::unique_ptr<Operation>(new Operation);
		operation->conjugate=conjugate;
		for(const auto& q: qmesh)
		{
			MeshPoint target=findMeshPoint(qmesh,(conjugate ? -q : q)*inverse.rot);
			if(target.index<0) return; // An anisotropic mesh need not preserve every operation.
			operation->qMap.push_back(target);
		}
		matrix3<int> rotation;
		vector3<int> shift;
		for(int i=0;i<3;i++)
		{
			double offset=inverse.a[i]*g.S[i];
			if(fabs(offset-round(offset))>1e-8) return;
			shift[i]=int(round(offset));
			for(int j=0;j<3;j++)
			{
				double value=double(inverse.rot(i,j))*g.S[i]/g.S[j];
				if(fabs(value-round(value))>1e-8) return;
				rotation(i,j)=int(round(value));
			}
		}
		operation->gridMap=std::make_shared<IndexArray>();
		operation->gridMap->init(g.nr);
		int* map=operation->gridMap->data();
		const double* density=e.eVars.n[0]->data();
		const double* potential=e.eVars.Vscloc[0]->data();
		double densityError=0., potentialError=0., densityNorm=0., potentialNorm=0.;
		vector3<int> x;
		for(x[0]=0;x[0]<g.S[0];x[0]++) for(x[1]=0;x[1]<g.S[1];x[1]++) for(x[2]=0;x[2]<g.S[2];x[2]++)
		{
			vector3<int> y=rotation*x+shift;
			for(int d=0;d<3;d++) y[d]=(y[d]%g.S[d]+g.S[d])%g.S[d];
			int index=(x[0]*g.S[1]+x[1])*g.S[2]+x[2];
			int source=(y[0]*g.S[1]+y[1])*g.S[2]+y[2];
			map[index]=source;
			densityError+=pow(density[index]-density[source],2);
			potentialError+=pow(potential[index]-potential[source],2);
			densityNorm+=pow(density[index],2);
			potentialNorm+=pow(potential[index],2);
		}
		// Geometry symmetry alone is insufficient for an asymmetric frozen state.
		if(densityError>1e-16*densityNorm || potentialError>1e-16*potentialNorm) return;
		operations.push_back(std::move(operation));
	}
public:
	ScreeningReuse(const Everything& e, const std::vector<vector3<>>& qmesh, bool enabled, double tolerance, bool quiet=false,double cacheMiB=256.,bool verbose=false)
		: e(e), enabled(enabled), verbose(verbose), diagnostics(false), quiet(quiet), tolerance(tolerance), byQ(qmesh.size()),cacheLimit(cacheMiB*1024.*1024.)
	{
		if(const char* setting=getenv("JDFTX_KI_SYMMETRY_DIAGNOSTICS")) {
			if(string(setting)!="0" && string(setting)!="1") die("JDFTX_KI_SYMMETRY_DIAGNOSTICS must be 0 or 1.\n");
			diagnostics=!quiet && string(setting)=="1";
		}
		if(enabled && mpiWorld->isHead())
		{
			// Put pure time reversal first: it remains valid for complex MLWFs
			// and requires neither a spatial permutation nor orbital equivalence.
			addOperation(SpaceGroupOp(),true,qmesh);
			for(const auto& op:e.symm.getMatrices())
			{
				size_t before=operations.size();
				addOperation(op,false,qmesh);
				if(operations.size()==before || (op.rot==matrix3<int>(1,1,1) && op.a.length_squared()<1e-20)) continue;
				// Compose each spatial operation with time reversal, sharing the
				// grid permutation so the extra q maps do not double GPU memory.
				auto combined=std::unique_ptr<Operation>(new Operation);
				combined->gridMap=operations.back()->gridMap;
				combined->conjugate=true;
				bool compatible=true;
				for(const auto& q:qmesh)
				{
					MeshPoint target=findMeshPoint(qmesh,-q*op.inv().rot);
					if(target.index<0) { compatible=false; break; }
					combined->qMap.push_back(target);
				}
				if(compatible) operations.push_back(std::move(combined));
			}
		}
		int count=operations.size(); mpiWorld->bcast(count);
		if(!quiet) logPrintf("KI response symmetry: %s, %d compatible operations, density tolerance %.3g.\n",
			enabled ? "enabled" : "disabled",count,tolerance);
	}
	// Keep each connected q star in one response group. Any successful cache
	// lookup maps q through one of these operations, so it never crosses groups.
	std::vector<int> qOwners(int& nGroups,std::vector<int>& order,std::vector<int>& starIndex,const std::vector<bool>& screeningQ={}) const
	{
		std::vector<int> owners(byQ.size());
		if(mpiWorld->isHead())
		{
			std::vector<int> parent(byQ.size());
			std::iota(parent.begin(),parent.end(),0);
			auto root=[&](int q) { while(parent[q]!=q) { parent[q]=parent[parent[q]]; q=parent[q]; } return q; };
			for(const auto& op: operations) for(size_t iq=0;iq<parent.size();iq++)
			{
				int a=root(iq), b=root(op->qMap[iq].index);
				parent[b]=a;
			}
			std::vector<std::vector<int>> stars(byQ.size());
			for(size_t iq=0;iq<parent.size();iq++) stars[root(iq)].push_back(iq);
			stars.erase(std::remove_if(stars.begin(),stars.end(),[](const std::vector<int>& s) { return s.empty(); }),stars.end());
			auto cost=[&](const std::vector<int>& star) {
				int work=star.size();
				if(!screeningQ.empty()) for(int iq:star) if(screeningQ[iq]) work+=byQ.size();
				return work;
			};
			std::stable_sort(stars.begin(),stars.end(),[&](const std::vector<int>& a,const std::vector<int>& b) { return cost(a)>cost(b); });
			nGroups=std::min(nGroups,int(stars.size()));
			if(!screeningQ.empty()) {
				int activeStars=0;
				for(const auto& star:stars) if(std::any_of(star.begin(),star.end(),[&](int iq) { return screeningQ[iq]; })) activeStars++;
				nGroups=std::min(nGroups,activeStars);
			}
			std::vector<int> capacity(nGroups,0), load(nGroups,0);
			for(int rank=0;rank<mpiWorld->nProcesses();rank++) capacity[rank*nGroups/mpiWorld->nProcesses()]++;
			for(const auto& star: stars)
			{
				int group=0;
				for(int g=1;g<nGroups;g++) if(double(load[g])/capacity[g]<double(load[group])/capacity[group]) group=g;
				for(int iq: star) owners[iq]=group;
				load[group]+=cost(star);
			}
			order.clear(); starIndex.resize(byQ.size());
			for(size_t star=0;star<stars.size();star++) for(int iq:stars[star]) {
				order.push_back(iq); starIndex[iq]=star;
			}
		}
		mpiWorld->bcast(nGroups);
		mpiWorld->bcastData(owners);
		mpiWorld->bcastData(order); mpiWorld->bcastData(starIndex);
		return owners;
	}
	void clearStar()
	{
		entries.clear(); for(auto& indices:byQ) indices.clear(); cacheBytes=0;
	}
	bool lookup(const complexScalarField& rho, double occupation, int iq, double bare, double& response, int orbital)
	{
		if(!enabled) return false;
		int match=-1;
		double error=0.;
		if(mpiWorld->isHead())
		{
			double norm=nrm2(rho);
			double closestError=INFINITY,closestNormError=INFINITY,closestCurvatureError=INFINITY;
			int closestOrbital=-1,closestQ=-1;
			for(const auto& op: operations)
			{
				const auto& mapped=op->qMap[iq];
				complexScalarField transformed;
				for(int index:byQ[mapped.index])
				{
					const auto& entry=entries[index];
					if(occupation!=entry.occupation || norm<1e-15 || entry.norm<1e-15) continue;
					double normError=fabs(norm-entry.norm)/std::max(norm,entry.norm);
					if(normError>tolerance && !diagnostics) continue;
					if(!transformed)
					{
						nullToZero(transformed,e.gInfo); initZero(transformed);
						callPref(eblas_gather_zdaxpy)(e.gInfo.nr,1.,op->gridMap->dataPref(),rho->dataPref(),transformed->dataPref(),op->conjugate);
						multiplyBlochPhase(transformed,mapped.wrap);
					}
					complex overlap=dot(entry.rho,transformed);
					double magnitude=abs(overlap);
					if(magnitude<1e-30) continue;
					complexScalarField difference=clone(transformed);
					callPref(eblas_zaxpy)(e.gInfo.nr,-overlap/magnitude,entry.rho->dataPref(),1,difference->dataPref(),1);
					error=nrm2(difference)/std::max(norm,entry.norm);
					if(diagnostics && error<closestError) {
						closestError=error; closestNormError=normError;
						closestOrbital=entry.orbital; closestQ=entry.iq;
						closestCurvatureError=fabs(bare-entry.bare)/std::max(1e-30,std::max(fabs(bare),fabs(entry.bare)));
					}
					if(error<=tolerance && normError<=tolerance) {
						double curvatureError=fabs(bare-entry.bare)/std::max(1e-30,std::max(fabs(bare),fabs(entry.bare)));
						if(curvatureError>tolerance) {
							if(!quiet && curvatureRejected<8) logPrintf("KI symmetry curvature rejection: q %d vs %d, source error %.3g, curvature error %.3g (%.12g vs %.12g).\n",
								iq+1,entry.iq+1,error,curvatureError,bare,entry.bare);
							curvatureRejected++; maxCurvatureError=std::max(maxCurvatureError,curvatureError); continue;
						}
						match=index; break;
					}
				}
				if(match>=0) break;
			}
			if(diagnostics && match<0 && closestOrbital>=0)
				logPrintf("KI symmetry mismatch: orbital %d, q %d; nearest orbital %d, q %d; source %.9g, norm %.9g, curvature %.9g; tolerance %.9g.\n",
					orbital+1,iq+1,closestOrbital+1,closestQ+1,closestError,closestNormError,closestCurvatureError,tolerance);
		}
		// The head decides once so GPU roundoff cannot change collective ordering.
		mpiWorld->bcast(match);
		if(match<0) return false;
		int sourceOrbital=0, sourceQ=0;
		if(mpiWorld->isHead())
		{
			const auto& entry=entries[match]; response=entry.response;
			sourceOrbital=entry.orbital; sourceQ=entry.iq;
			maxError=std::max(maxError,error);
		}
		mpiWorld->bcast(response); mpiWorld->bcast(error);
		mpiWorld->bcast(sourceOrbital); mpiWorld->bcast(sourceQ);
		if(verbose) logPrintf("KI response reuse from orbital %d, q %d (relative density error %.3g).\n",sourceOrbital+1,sourceQ+1,error);
		reused++;
		return true;
	}
	void record(const complexScalarField& rho, int orbital, double occupation, int iq, double bare, double response, bool countSolve=true)
	{
		if(countSolve) solved++;
		if(!enabled || !mpiWorld->isHead()) return;
		size_t bytes=size_t(e.gInfo.nr)*sizeof(complex);
		if(cacheBytes+bytes>cacheLimit) return; // Bounded memory; uncached pairs are solved normally.
		byQ[iq].push_back(entries.size());
		entries.push_back({orbital,iq,occupation,nrm2(rho),bare,response,clone(rho)});
		cacheBytes+=bytes;
		peakCacheBytes=std::max(peakCacheBytes,cacheBytes);
	}
	void report(const MPIUtil* reduction=nullptr, bool contribute=true, bool audit=false) const
	{
		int counts[2]={contribute ? solved : 0, contribute ? reused : 0};
		double error=contribute ? maxError : 0., bytes=contribute ? double(peakCacheBytes) : 0.;
		int rejected=contribute ? curvatureRejected : 0;
		double curvatureError=contribute ? maxCurvatureError : 0.;
		if(reduction)
		{
			reduction->allReduce(counts,2,MPIUtil::ReduceSum);
			reduction->allReduce(error,MPIUtil::ReduceMax);
			reduction->allReduce(bytes,MPIUtil::ReduceSum);
			reduction->allReduce(rejected,MPIUtil::ReduceSum);
			reduction->allReduce(curvatureError,MPIUtil::ReduceMax);
		}
		logPrintf("KI %s: %d/%d; reused %d symmetry-equivalent pairs; maximum density error %.3g; cache %.1f MiB.\n",
			audit ? "symmetry audit pairs" : "response solves",counts[0],counts[0]+counts[1],counts[1],error,bytes/(1024.*1024.));
		logPrintf("KI symmetry curvature checks: %d rejected density-equivalent candidates; maximum relative curvature error %.3g.\n",rejected,curvatureError);
	}
};

// The native response solver uses mpiWorld throughout. Temporarily scope it
// to a response group, including cached real/reciprocal grid loop ranges.
// GPU assignment remains the original world assignment from initialization.
class QResponseGroups
{
	MPIUtil *world=mpiWorld, *group=mpiGroup, *heads=mpiGroupHead;
	FILE *log=globalLog, *originalLog=globalLogOrig;
	std::unique_ptr<MPIUtil> response, localGroup, localHeads;
	FILE* groupLog=nullptr;
	struct GridRange { GridInfo* grid; int rStart,rStop,gStart,gStop; };
	std::vector<GridRange> ranges;
	bool active=false;
public:
	int index=0;
	QResponseGroups(Everything& e, const Wannier& w, int spin, int nGroups)
	{
		if(nGroups==1) return;
		response.reset(new MPIUtil(0,nullptr,MPIUtil::ProcDivision(world,nGroups)));
		index=response->procDivision.iGroup;
		localGroup.reset(new MPIUtil(0,nullptr,MPIUtil::ProcDivision(response.get(),response->nProcesses())));
		localHeads.reset(new MPIUtil(0,nullptr,MPIUtil::ProcDivision(response.get(),0,localGroup->iProcess())));
		mpiWorld=response.get(); mpiGroup=localGroup.get(); mpiGroupHead=localHeads.get();
		active=true;
		if(index && mpiWorld->isHead())
		{
			ostringstream name; name << "koopmansQGroup" << index << ".out";
			groupLog=fopen(w.getFilename(Wannier::FilenameDump,name.str(),&spin).c_str(),"w");
			if(!groupLog) die("Cannot open KI q-group log.\n");
			globalLog=groupLog;
		}
		globalLogOrig=globalLog;
		auto redistribute=[&](GridInfo& grid)
		{
			ranges.push_back({&grid,grid.irStart,grid.irStop,grid.iGstart,grid.iGstop});
			TaskDivision(grid.nr,mpiWorld).myRange(grid.irStart,grid.irStop);
			TaskDivision(grid.nG,mpiWorld).myRange(grid.iGstart,grid.iGstop);
		};
		redistribute(e.gInfo);
		if(e.gInfoWfns && e.gInfoWfns.get()!=&e.gInfo) redistribute(*e.gInfoWfns);
		logPrintf("KI q group %d/%d: %d MPI ranks; k parallelism within this group.\n",index+1,nGroups,mpiWorld->nProcesses());
	}
	void restore()
	{
		if(!active) return;
		for(const auto& r: ranges)
		{
			r.grid->irStart=r.rStart; r.grid->irStop=r.rStop;
			r.grid->iGstart=r.gStart; r.grid->iGstop=r.gStop;
		}
		mpiWorld=world; mpiGroup=group; mpiGroupHead=heads; globalLog=log; globalLogOrig=originalLog;
		if(groupLog) { fclose(groupLog); groupLog=nullptr; }
		active=false;
	}
	~QResponseGroups() { restore(); }
};

// The Wannier executable owns mutable Everything. Restore its response state
// after spectral postprocessing so other Wannier outputs remain independent.
struct ResponseState
{
	Everything& e;
	PerturbationInfo saved;
	ResponseState(Everything& e) : e(e), saved(std::move(e.pertInfo)) { e.pertInfo=PerturbationInfo(); }
	~ResponseState() { e.pertInfo=std::move(saved); }
};
// Preserve the original vector buffers: loaded wavefunctions contain pointers
// into the reduced basis and quantum-number arrays. Restore their addresses,
// as well as the original MPI distribution, before exporting Wannier matrices.
struct FullMeshState
{
	Everything& e;
	ElecInfo info;
	std::vector<QuantumNumber> qnums;
	std::vector<Basis> basis;
	std::vector<ColumnBundle> C;
	std::vector<diagMatrix> F, eigs;
	std::vector<matrix> H, vectors;
	std::vector<std::vector<matrix>> projections;
	bool active=true;
	FullMeshState(Everything& e) : e(e), info(e.eInfo),
		qnums(std::move(e.eInfo.qnums)), basis(std::move(e.basis)),
		C(std::move(e.eVars.C)), F(std::move(e.eVars.F)), eigs(std::move(e.eVars.Hsub_eigs)),
		H(std::move(e.eVars.Hsub)), vectors(std::move(e.eVars.Hsub_evecs)),
		projections(std::move(e.eVars.VdagC)) {}
	void restore()
	{
		if(!active) return;
		e.eVars.C=std::move(C); e.eVars.F=std::move(F);
		e.eVars.Hsub_eigs=std::move(eigs); e.eVars.Hsub=std::move(H);
		e.eVars.Hsub_evecs=std::move(vectors); e.eVars.VdagC=std::move(projections);
		e.basis=std::move(basis);
		e.eInfo=info;
		e.eInfo.qnums=std::move(qnums);
		active=false;
	}
	~FullMeshState() { restore(); }
};
FILE* openOutput(const string& name)
{
	FILE* fp=fopen(name.c_str(), "w");
	if(!fp) die_alone("Could not write '%s'.\n", name.c_str());
	return fp;
}
}

void WannierMinimizer::saveMLWF_KI_densities(int iSpin)
{
	if(e.eInfo.spinType!=SpinNone) die("wannier-koopmans-save-densities requires spintype no-spin.\n");
	Everything& system=const_cast<Everything&>(e);
	std::vector<ScalarField> densities(nCenters);
	for(auto& density:densities) { nullToZero(density,e.gInfo); initZero(density); }
	Cother.resize(e.eInfo.nStates);
	for(const auto& ke:kMesh) {
		int state=ke.point.iReduced, owner=e.eInfo.whose(state);
		ColumnBundle received;
		ColumnBundle& source=e.eInfo.isMine(state) ? system.eVars.C[state] : received;
		if(!e.eInfo.isMine(state)) source.init(nBands,e.basis[state].nbasis,&e.basis[state],&e.eInfo.qnums[state],isGpuEnabled());
		mpiWorld->bcastData(source,owner);
		if(!e.eInfo.isMine(state)) Cother[state]=source;
		ColumnBundle localized=getWfns(ke.point,iSpin)*ke.U;
		for(int n=0;n<nCenters;n++) {
			auto u=changeGrid(I(localized.getColumn(n,0)),e.gInfo);
			densities[n]+=ke.point.weight*Real(conj(u)*u);
		}
		if(!e.eInfo.isMine(state)) Cother[state].free();
	}
	Cother.clear();
	if(mpiWorld->isHead()) {
		FILE* output=openOutput(wannier.getFilename(Wannier::FilenameDump,"koopmansOrbitalDensities",&iSpin));
		for(const auto& density:densities) saveRawBinary(density,output);
		fclose(output);
		FILE* format=openOutput(wannier.getFilename(Wannier::FilenameDump,"koopmansDensityFormat",&iSpin));
		fprintf(format,"float64 %d %d %d %d %d\n",nCenters,e.gInfo.nr,e.gInfo.S[0],e.gInfo.S[1],e.gInfo.S[2]);
		for(int i=0;i<3;i++) for(int j=0;j<3;j++) fprintf(format,"%.15g%c",e.gInfo.R(i,j),j==2 ? '\n' : ' ');
		fclose(format);
	}
}

void WannierMinimizer::saveMLWF_KI(int iSpin, const matrix& phase)
{
	const char* auditSetting=getenv("JDFTX_KI_SYMMETRY_AUDIT");
	if(auditSetting && string(auditSetting)!="0" && string(auditSetting)!="1") die("JDFTX_KI_SYMMETRY_AUDIT must be 0 or 1.\n");
	const bool symmetryAudit=auditSetting && string(auditSetting)=="1";
	logPrintf("\nComputing primitive-cell Koopmans KI(2) Hamiltonian.\n");
	Everything& system=const_cast<Everything&>(e);
	KIProjectorCache projectorCache(system.iInfo);
	ResponseState restore(system);
	PerturbationInfo& p=system.pertInfo;
	p.commensurate=false; // also at Gamma: use the complex finite-q interface
	p.occupiedProjectorSupplied=true;
	p.fullMeshResponse=true;
	p.kiVerbose=wannier.koopmansVerbose;
	p.optimizeFrozenResponse=wannier.koopmansOptimizeResponse;
	p.frozenOrbitalCacheMiB=wannier.koopmansOrbitalCacheMiB;
	p.nestedResponse=wannier.koopmansNestedResponse;
	p.independentBandCG=wannier.koopmansIndependentBandCG;
	p.phononMixFraction=wannier.koopmansMixFraction;
	p.phononMixHistory=wannier.koopmansMixHistory;
	p.phononMixIterations=wannier.koopmansMixIterations;
	p.solverParams.nIterations=wannier.koopmansIterations;
	p.solverParams.residualTol=wannier.koopmansTolerance;
	p.solverParams.residualDiffThreshold=0.;
	p.solverParams.fpLog=globalLog;
	p.solverParams.linePrefix="KoopmansResponse: ";
	p.solverParams.recomputeResidual=true;
	p.solverParams.logIterations=wannier.koopmansVerbose;

	// Reconstruct the frozen DFT density/potential. This does not rotate C.
	if(wannier.koopmansReferenceDensity.empty())
		system.eVars.elecEnergyAndGrad(system.ener, nullptr, nullptr, true);
	else {
		logPrintf("KI reference density: '%s'; retain SCF density for NSCF response states.\n",wannier.koopmansReferenceDensity.c_str());
		system.eVars.n.resize(1); nullToZero(system.eVars.n[0],e.gInfo);
		loadRawBinary(system.eVars.n[0],wannier.koopmansReferenceDensity.c_str());
		double electrons=integral(system.eVars.n[0]);
		if(!std::isfinite(electrons) || fabs(electrons-e.eInfo.nElectrons)>1e-6*std::max(1.,e.eInfo.nElectrons))
			die("KI reference density has an inconsistent electron count.\n");
		Energies referenceEnergy=system.ener;
		referenceEnergy.E["KE"]=0.; referenceEnergy.E["Enl"]=0.;
		system.eVars.EdensityAndVscloc(referenceEnergy);
		for(int state=e.eInfo.qStart;state<e.eInfo.qStop;state++) {
			ColumnBundle HC;
			system.eVars.applyHamiltonian(state,system.eVars.F[state],HC,referenceEnergy,true);
		}
	}
	const int nk=kMesh.size();
	std::vector<vector3<>> mesh;
	for(const auto& ke: kMesh) mesh.push_back(ke.point.k);
	bool hasGamma=false;
	for(vector3<> k: mesh)
	{
		for(int d=0; d<3; d++) k[d]-=round(k[d]);
		hasGamma |= k.length_squared()<1e-16;
	}
	if(!hasGamma) die("wannier-koopmans requires a Gamma-centered uniform k mesh.\n");
	const vector3<int> folding=e.eInfo.kFoldingCount();
	if(nk!=folding[0]*folding[1]*folding[2])
		die("wannier-koopmans requires a complete regular kpoint-folding mesh.\n");
	for(const auto& ke: kMesh) if(fabs(ke.point.weight-1./nk)>1e-10)
		die("wannier-koopmans requires equal weights on the complete k mesh.\n");
	std::vector<vector3<>> qmesh;
	for(int iq=0; iq<nk; iq++) {
		vector3<> q=mesh[iq]-mesh[0];
		for(int d=0; d<3; d++) q[d]-=floor(q[d]+0.5);
		qmesh.push_back(q);
	}
	vector3<int> screeningMesh=wannier.koopmansScreeningMesh.isNonzero() ? wannier.koopmansScreeningMesh : folding;
	const vector3<int> requestedScreeningMesh=screeningMesh;
	int nqScreen=screeningMesh[0]*screeningMesh[1]*screeningMesh[2];
	bool coarseScreening=nqScreen!=nk;
	double screeningWeight=double(nk)/nqScreen;
	const bool adaptiveScreening=coarseScreening && wannier.koopmansScreeningEnergyTolerance>0. && !symmetryAudit;
	const bool importedScreening=!wannier.koopmansAlphaFilename.empty();
	vector3<int> alphaSourceMesh=screeningMesh;
	diagMatrix importedAlpha(nCenters);
	std::vector<double> importedBare(nCenters), importedRelaxed(nCenters);
	std::vector<bool> screeningQ(nk,true);
	if(coarseScreening) for(int iq=0;iq<nk;iq++) for(int d=0;d<3;d++) {
		double index=qmesh[iq][d]*screeningMesh[d];
		if(fabs(index-round(index))>1e-8) screeningQ[iq]=false;
	}
	if(int(std::count(screeningQ.begin(),screeningQ.end(),true))!=nqScreen)
		die("KI screening q mesh is not a subset of the Wannier mesh.\n");
	if(importedScreening) std::fill(screeningQ.begin(),screeningQ.end(),false);
	logPrintf("KI screening q mesh: %d %d %d (%d/%d q points); full k integration and bare Hamiltonian retained.\n",
		screeningMesh[0],screeningMesh[1],screeningMesh[2],nqScreen,nk);
	if(adaptiveScreening) logPrintf("KI adaptive screening: tolerance %.6g eV; require two successive Hamiltonian changes below tolerance; reuse completed responses.\n",
		wannier.koopmansScreeningEnergyTolerance);

	// Gather the canonical states once, one k point at a time. Source columns
	// are retained for building exact occupied projectors at k +/- q.
	Cother.resize(e.eInfo.nStates);
	std::vector<ColumnBundle> canonical(nk);
	std::vector<std::vector<complexScalarField>> orbital(nk);
	std::vector<matrix> base(nk), correction(nk), fullH(nk);
	std::vector<diagMatrix> fullF(nk), fullEigs(nk);
	diagMatrix occupations(nCenters, -1.);
	double evMax=-INFINITY, ecMin=INFINITY;
	for(int ik=0; ik<nk; ik++)
	{
		const auto& ke=kMesh[ik];
		int state=ke.point.iReduced;
		int owner=e.eInfo.whose(state);
		ColumnBundle received;
		ColumnBundle& source=e.eInfo.isMine(state) ? system.eVars.C[state] : received;
		if(!e.eInfo.isMine(state)) source.init(nBands, e.basis[state].nbasis, &e.basis[state], &e.eInfo.qnums[state], isGpuEnabled());
		mpiWorld->bcastData(source, owner);
		// getWfns handles the common-basis transform, including k wrapping.
		if(!e.eInfo.isMine(state)) Cother[state]=source;
		canonical[ik]=getWfns(ke.point, iSpin);
		ColumnBundle localized=canonical[ik]*ke.U;
		for(int n=0; n<nCenters; n++) orbital[ik].push_back(changeGrid(I(localized.getColumn(n,0)), e.gInfo));
		if(!e.eInfo.isMine(state)) Cother[state].free();

		diagMatrix filling(nBands), eigs(nBands);
		matrix hsub=zeroes(nBands,nBands);
		if(e.eInfo.isMine(state)) { filling=e.eVars.F[state]; eigs=e.eVars.Hsub_eigs[state]; hsub=e.eVars.Hsub[state]; }
		mpiWorld->bcastData(filling,owner);
		mpiWorld->bcastData(eigs,owner);
		mpiWorld->bcastData(hsub,owner);
		for(int b=0; b<nBands; b++)
		{
			if(fabs(filling[b]-round(filling[b]))>1e-8 || filling[b]<-1e-8 || filling[b]>1.+1e-8)
				die("wannier-koopmans requires integer band occupations.\n");
			if(filling[b]>0.5) evMax=std::max(evMax,eigs[b]);
			else ecMin=std::min(ecMin,eigs[b]);
		}
		matrix fWannier=dagger(ke.U)*filling*ke.U;
		for(int n=0; n<nCenters; n++)
		{
			double f=fWannier(n,n).real();
			if(!std::isfinite(f) || fabs(f-round(f))>1e-7)
				die("KI Wannier rotations mix occupied and empty states; localize the manifolds separately.\n");
			if(occupations[n]<0.) occupations[n]=round(f);
			if(fabs(f-occupations[n])>1e-7)
				die("KI Wannier occupations change across the k mesh.\n");
		}
		if(ke.point.invert<0) hsub=conj(hsub);
		fullF[ik]=filling; fullEigs[ik]=eigs; fullH[ik]=hsub;
		base[ik]=dagger(ke.U)*hsub*ke.U;
		correction[ik]=zeroes(nCenters,nCenters);
	}
	Cother.clear();
	if(importedScreening) {
		std::ifstream input(wannier.koopmansAlphaFilename.c_str());
		if(!input) die("Cannot open imported KI screening table.\n");
		bool functionalOK=false,spinOK=false,epsilonOK=false,meshOK=false;
		std::vector<bool> seen(nCenters,false);
		std::string line;
		while(std::getline(input,line)) {
			std::istringstream fields(line);
			std::string first; if(!(fields>>first)) continue;
			if(first=="#") {
				std::string key,value; fields>>key;
				if(key=="functional") { fields>>value; functionalOK=value=="KI_second_order"; }
				if(key=="spin_treatment") {
					fields>>value;
					spinOK=value==(wannier.koopmansSpinResponse ? "single_spin_charge_and_magnetic" : "restricted_charge_only");
				}
				if(key=="epsilon_infinity") {
					double epsilon; if(fields>>epsilon) epsilonOK=std::isfinite(epsilon) && fabs(epsilon-wannier.koopmansEpsilon)<1e-12;
				}
				if(key=="screening_q_mesh") {
					vector3<int> full; std::string separator,label;
					if(fields>>alphaSourceMesh[0]>>alphaSourceMesh[1]>>alphaSourceMesh[2]>>separator>>label>>full[0]>>full[1]>>full[2])
						meshOK=separator==";" && label=="full_k_mesh" && full==alphaSourceMesh && full[0]>0 && full[1]>0 && full[2]>0;
				}
				continue;
			}
			std::istringstream row(line);
			int n; double occupation,bareValue,relaxed,alpha,image;
			if(!(row>>n>>occupation>>bareValue>>relaxed>>alpha>>image) || n<0 || n>=nCenters || seen[n])
				die("Invalid, duplicate or out-of-range row in imported KI screening table.\n");
			if(occupation!=occupations[n] || !std::isfinite(bareValue) || fabs(bareValue)<1e-12 || !std::isfinite(relaxed)
				|| !std::isfinite(alpha) || !std::isfinite(image) || fabs(relaxed-alpha*bareValue)>1e-10*(1.+fabs(relaxed)))
				die("Imported KI screening has inconsistent occupation, curvature or alpha.\n");
			seen[n]=true; importedAlpha[n]=alpha; importedBare[n]=bareValue; importedRelaxed[n]=relaxed;
		}
		if(!functionalOK || !spinOK || !epsilonOK || !meshOK || std::count(seen.begin(),seen.end(),true)!=nCenters)
			die("Imported KI screening requires complete orbital rows and matching functional, spin, epsilon and full coarse k/q mesh metadata.\n");
		logPrintf("KI imported screening: '%s'; source k/q mesh %d %d %d; no screening response solves.\n",
			wannier.koopmansAlphaFilename.c_str(),alphaSourceMesh[0],alphaSourceMesh[1],alphaSourceMesh[2]);
	}
	if(!std::isfinite(evMax) || !std::isfinite(ecMin) || !(ecMin>evMax+1e-8))
		die("wannier-koopmans requires a finite DFT band gap; store at least one empty band to check it.\n");

	// Each response column is the variation of an occupied ground-state orbital,
	// including when the perturbing density belongs to an empty Wannier orbital.
	// The variation still spans the full plane-wave complement of the occupied
	// projector: no finite empty-band approximation is introduced. Retain all
	// stored bands in canonical and base for the Wannier orbitals and spectra.
	std::vector<std::vector<int>> responseBands(nk);
	for(int ik=0; ik<nk; ik++)
		for(int b=0; b<nBands; b++)
			if(!wannier.koopmansOccupiedResponse || fullF[ik][b]>0.5)
				responseBands[ik].push_back(b);
	const int nResponse=responseBands[0].size();
	for(const auto& bands: responseBands)
		if(int(bands.size())!=nResponse || !nResponse)
			die("KI occupied response requires the same nonzero occupied band count at every k point.\n");
	// Lowest canonical empty states accelerate the near-gap Sternheimer modes.
	// Keep the source and offset spaces together within a per-rank budget.
	int coarseCount=8;
	double coarseBudgetMiB=256.;
	if(const char* value=getenv("JDFTX_KI_EMPTY_PRECOND_BANDS")) {
		char* end=nullptr; long count=strtol(value,&end,10);
		if(end==value || *end || count<0 || count>64) die("JDFTX_KI_EMPTY_PRECOND_BANDS must be an integer from 0 to 64.\n");
		coarseCount=count;
	}
	if(const char* value=getenv("JDFTX_KI_EMPTY_PRECOND_MIB")) {
		char* end=nullptr; coarseBudgetMiB=strtod(value,&end);
		if(end==value || *end || !std::isfinite(coarseBudgetMiB) || coarseBudgetMiB<0. || coarseBudgetMiB>65536.)
			die("JDFTX_KI_EMPTY_PRECOND_MIB must be a finite number from 0 to 65536 MiB.\n");
	}
	coarseCount=std::min(coarseCount,nBands-nResponse);
	if(!wannier.koopmansOptimizeResponse || !wannier.koopmansOccupiedResponse || !p.nestedResponse
		|| !p.independentBandCG || !coarseBudgetMiB) coarseCount=0;
	size_t nativeBytesPerColumn=0;
	for(int ik=0;ik<nk;ik++) nativeBytesPerColumn+=e.basis[kMesh[ik].point.iReduced].nbasis*sizeof(complex);
	// Three complete meshes bound one retained source and both offset signs,
	// including inverse-denominator storage, regardless of MPI partitioning.
	const size_t coarseBytesPerColumn=3*nativeBytesPerColumn+2*size_t(nk)*nResponse*sizeof(complex);
	coarseCount=std::min(size_t(coarseCount),size_t(coarseBudgetMiB*1024.*1024.)/std::max(size_t(1),coarseBytesPerColumn));
	std::vector<ColumnBundle> coarseCanonical(nk);
	if(coarseCount) for(int ik=0;ik<nk;ik++) {
		std::vector<int> empty;
		for(int b=0;b<nBands;b++) if(fullF[ik][b]<0.5) empty.push_back(b);
		matrix hEmpty(empty.size(),empty.size()); complex* h=hEmpty.data();
		const matrix& hFull=fullH[ik]; const complex* original=hFull.data();
		for(size_t j=0;j<empty.size();j++) for(size_t i=0;i<empty.size();i++)
			h[hEmpty.index(i,j)]=original[hFull.index(empty[i],empty[j])];
		matrix vectors; diagMatrix energies; hEmpty.diagonalize(vectors,energies);
		matrix rotation(nBands,coarseCount); complex* r=rotation.data();
		std::fill(r,r+rotation.nData(),complex(0.));
		const complex* v=vectors.data();
		for(int col=0;col<coarseCount;col++) for(size_t row=0;row<empty.size();row++)
			r[rotation.index(empty[row],col)]=v[vectors.index(row,col)];
		coarseCanonical[ik]=canonical[ik]*rotation;
	}
	if(wannier.koopmansOccupiedResponse)
		for(int ik=0; ik<nk; ik++)
		{
			diagMatrix filling(nResponse), eigs(nResponse);
			matrix hsub=zeroes(nResponse,nResponse);
			for(int b=0; b<nResponse; b++)
			{
				const int source=responseBands[ik][b];
				filling[b]=fullF[ik][source]; eigs[b]=fullEigs[ik][source];
				for(int c=0; c<nResponse; c++) hsub.set(b,c,fullH[ik](source,responseBands[ik][c]));
			}
			fullF[ik]=std::move(filling); fullEigs[ik]=std::move(eigs); fullH[ik]=std::move(hsub);
		}
	// The base Wannier Hamiltonian and orbitals above already contain the full
	// stored-band contribution. Only occupied columns are needed from now on.
	if(wannier.koopmansOptimizeResponse && wannier.koopmansOccupiedResponse)
		for(int ik=0;ik<nk;ik++) {
			ColumnBundle compact=canonical[ik].similar(nResponse);
			for(int b=0;b<nResponse;b++) compact.copyColumnFrom(b,canonical[ik],responseBands[ik][b]);
			// MLWF orbitals and the base Hamiltonian have already been formed in
			// their original gauge. An occupied-only unitary preserves the frozen
			// projector/density and lets independent-band CG use actual eigenvalues.
			const char* canonicalSetting=getenv("JDFTX_KI_CANONICAL_RESPONSE");
			if(p.nestedResponse && p.independentBandCG && (!canonicalSetting || string(canonicalSetting)!="0")) {
				matrix vectors; diagMatrix values;
				fullH[ik].diagonalize(vectors,values);
				compact=compact*vectors; fullH[ik]=matrix(values); fullEigs[ik]=values;
			}
			canonical[ik]=std::move(compact);
		}
	const bool compactCanonical=wannier.koopmansOptimizeResponse && wannier.koopmansOccupiedResponse;
	if(compactCanonical && p.nestedResponse && p.independentBandCG)
		logPrintf("KI occupied response gauge: %s.\n",getenv("JDFTX_KI_CANONICAL_RESPONSE") && string(getenv("JDFTX_KI_CANONICAL_RESPONSE"))=="0"
			? "original" : "Hamiltonian eigenvectors (MLWF gauge retained for outputs)");
	std::vector<int> emptyCenters;
	for(int n=0;n<nCenters;n++) if(!occupations[n]) emptyCenters.push_back(n);
	std::vector<matrix> conductionOrbitals;
	ManagedArray<int> emptyRows(emptyCenters);
	std::unique_ptr<KMeshCorrelation> correlation;
	if(wannier.koopmansOptimizeResponse)
		correlation.reset(new KMeshCorrelation(e.gInfo,folding,mesh,qmesh,orbital,wannier.koopmansContractionCacheMiB));
	const bool fftCorrelation=correlation && correlation->enabled();
	if(wannier.koopmansOptimizeResponse && !fftCorrelation && !emptyCenters.empty()) {
		conductionOrbitals.resize(nk);
		size_t packedBytes=0; int packedPoints=0;
		const double budget=wannier.koopmansContractionCacheMiB*1024.*1024.;
		for(int ik=0;ik<nk;ik++) {
			size_t required=size_t(e.gInfo.nr)*emptyCenters.size()*sizeof(complex);
			if(double(packedBytes)+required>budget) continue;
			matrix& packed=conductionOrbitals[ik]; packed.init(e.gInfo.nr,emptyCenters.size(),isGpuEnabled());
			for(size_t m=0;m<emptyCenters.size();m++)
				callPref(eblas_copy)(packed.dataPref()+m*e.gInfo.nr,orbital[ik][emptyCenters[m]]->dataPref(),e.gInfo.nr);
			packedBytes+=required; packedPoints++;
		}
		logPrintf("KI conduction contraction cache: %d/%d k points, %.1f MiB (budget %.1f MiB/rank).\n",
			packedPoints,nk,packedBytes/(1024.*1024.),wannier.koopmansContractionCacheMiB);
	}

	// Individual orbital perturbations need the full integration mesh even
	// when the underlying SCF calculation uses space-group/time reversal reduction.
	const int reducedStates=system.eInfo.nStates;
	int nQGroups=wannier.koopmansQGroups;
	std::vector<int> qOwner(nk,0);
	std::vector<int> qOrder(nk),qStar(nk);
	std::iota(qOrder.begin(),qOrder.end(),0);
	std::iota(qStar.begin(),qStar.end(),0);
	if(nQGroups>1 || wannier.koopmansReduceSymmetry)
	{
		ScreeningReuse planner(e,qmesh,wannier.koopmansReduceSymmetry,wannier.koopmansSymmetryTolerance,true);
		qOwner=planner.qOwners(nQGroups,qOrder,qStar,coarseScreening && !adaptiveScreening ? screeningQ : std::vector<bool>());
		if(!wannier.koopmansOptimizeResponse) std::iota(qOrder.begin(),qOrder.end(),0);
	}
	logPrintf("KI q parallelism: %d response groups requested, %d active, %d MPI ranks total.\n",
		wannier.koopmansQGroups,nQGroups,mpiWorld->nProcesses());
	for(int group=0;group<nQGroups;group++)
		logPrintf("KI q assignment: group %d owns %d/%d q points.\n",group+1,int(std::count(qOwner.begin(),qOwner.end(),group)),nk);
	projectorCache.clear();
	FullMeshState fullState(system);
	MPIUtil* responseWorld=mpiWorld;
	QResponseGroups responseGroups(system,wannier,iSpin,nQGroups);
	p.solverParams.fpLog=globalLog;
	ElecInfo& info=system.eInfo;
	info.nBands=nResponse;
	info.qnums.resize(nk);
	info.nStates=nk;
	info.qDivision.init(nk,mpiWorld);
	info.qDivision.myRange(info.qStart,info.qStop);
	system.basis.resize(nk);
	system.eVars.C.resize(nk);
	system.eVars.F=fullF;
	system.eVars.Hsub=fullH;
	system.eVars.Hsub_eigs=fullEigs;
	system.eVars.Hsub_evecs.resize(nk);
	system.eVars.VdagC.resize(nk,std::vector<matrix>(e.iInfo.species.size()));
	const GridInfo& waveGrid=e.gInfoWfns ? *e.gInfoWfns : e.gInfo;
	std::vector<ColumnBundle> coarseSource(nk);
	size_t coarseSourceBytes=0;
	FILE* basisLog=globalLog;
	if(!wannier.koopmansVerbose) logSuspend();
	for(int ik=0; ik<nk; ik++)
	{
		info.qnums[ik]=kMesh[ik].point;
		info.qnums[ik].weight=info.spinWeight/double(nk);
		system.basis[ik].setup(waveGrid,e.iInfo,e.cntrl.Ecut,mesh[ik]);
		if(coarseCount) {
			auto& C=coarseSource[ik];
			C.init(coarseCount,system.basis[ik].nbasis,&system.basis[ik],&info.qnums[ik],isGpuEnabled());
			for(int b=0;b<coarseCount;b++) C.setColumn(b,0,coarseCanonical[ik].getColumn(b,0));
			coarseSourceBytes+=C.nData()*sizeof(complex);
			coarseCanonical[ik].free();
		}
		if(info.isMine(ik))
		{
			ColumnBundle& C=system.eVars.C[ik];
			C.init(nResponse,system.basis[ik].nbasis,&system.basis[ik],&info.qnums[ik],isGpuEnabled());
			for(int b=0; b<nResponse; b++) C.setColumn(b,0,canonical[ik].getColumn(compactCanonical ? b : responseBands[ik][b],0));
		}
	}
	globalLog=basisLog;
	coarseCanonical.clear();
	// Native k-dependent bases at equivalent k points differ by an integer
	// reciprocal shift. Retain remote occupied columns once so an offset
	// projector can usually be copied without an inverse/forward FFT pair.
	std::vector<ColumnBundle> offsetSource(nk);
	if(wannier.koopmansOptimizeResponse && wannier.koopmansOccupiedResponse) for(int ik=0;ik<nk;ik++) if(!info.isMine(ik)) {
		auto& C=offsetSource[ik];
		C.init(nResponse,system.basis[ik].nbasis,&system.basis[ik],&info.qnums[ik],isGpuEnabled());
		for(int b=0;b<nResponse;b++) C.setColumn(b,0,canonical[ik].getColumn(compactCanonical ? b : responseBands[ik][b],0));
	}
	p.checkSupportedFeatures(system,info);
	logPrintf("KI screening uses %d full-mesh k points reconstructed from %d SCF states.\n",nk,reducedStates);
	logPrintf("KI response columns: %d/%d stored bands (%s); full plane-wave response space retained.\n",
		nResponse,nBands,wannier.koopmansOccupiedResponse ? "occupied only" : "all stored bands");

	// Point-charge Ewald regularization, Eq. 24. Only its Gamma constant
	// differs from the Hartree kernel used by the Sternheimer solver.
	// Add D to the bare curvature and D/epsilonInfinity to the screened one.
	CoulombParams regularParams=e.coulombParams;
	regularParams.exchangeRegularization=CoulombParams::ProbeChargeEwald;
	regularParams.omegaSet.clear();
	regularParams.omegaSet.insert(0.);
	std::shared_ptr<CoulombPeriodic> regularCoulomb;
	if(wannier.koopmansEpsilon) regularCoulomb=std::make_shared<CoulombPeriodic>(e.gInfo,regularParams);
	else logPrintf("KI uses a neutralizing background; converge the q mesh for charged-orbital image interactions.\n");
	std::shared_ptr<CoulombPeriodic> screeningCoulomb=regularCoulomb;
	auto updateScreeningCoulomb=[&]() {
		screeningCoulomb=regularCoulomb;
		if(!regularCoulomb || !coarseScreening) return;
		// The Gamma singularity depends on the q-mesh supercell, not just
		// its quadrature weight. Retain the fine kernel for the Hamiltonian
		// and construct the coarse one independently for screening.
		std::vector<vector3<>> screeningPoints;
		for(int iq=0;iq<nk;iq++) if(screeningQ[iq]) screeningPoints.push_back(qmesh[iq]);
		regularParams.supercell=std::make_shared<Supercell>(e.gInfo,screeningPoints,
			std::vector<SpaceGroupOp>(1),std::vector<int>(1,1));
		screeningCoulomb=std::make_shared<CoulombPeriodic>(e.gInfo,regularParams);
	};
	updateScreeningCoulomb();

	std::vector<double> bare(nCenters,0.), screeningBare(nCenters,0.), relaxation(nCenters,0.), singular(nCenters,0.);
	std::vector<double> screeningSingular(nCenters,0.);
	std::vector<double> bareQ(size_t(nk)*nCenters,0.), responseQ(bareQ.size(),0.), imageQ(bareQ.size(),0.);
	std::vector<double> screeningImageQ(bareQ.size(),0.);
	ScreeningReuse reuse(e,qmesh,wannier.koopmansReduceSymmetry,wannier.koopmansSymmetryTolerance,false,
		wannier.koopmansOptimizeResponse ? wannier.koopmansSymmetryCacheMiB : 256.,wannier.koopmansVerbose);
	// Only the offset bases/projectors and source depend on q. The frozen
	// XC kernel and solver references remain valid for this whole group.
	p.kiSpinResponse=wannier.koopmansSpinResponse;
	p.magneticResponse=false;
	PerturbationSolver solver(system);
	if(wannier.koopmansOptimizeResponse) solver.updateExcorrCache();
	logPrintf("KI spin treatment: %s. Orbital sources integrate to one electron; Bloch response includes both spins.\n",
		p.kiSpinResponse ? "single-spin orbital (charge and magnetic response)" : "restricted charge only (legacy)");
	auto orbitalPotentials=[&](const complexScalarField& rho,vector3<>& q) {
		complexScalarFieldArray result(p.kiSpinResponse ? 2 : 1),density(1,rho);
		for(size_t channel=0;channel<result.size();channel++) {
			p.magneticResponse=(channel==1);
			complexScalarFieldArray potential(1);
			solver.getdVsclocPsi(density,potential,&q);
			result[channel]=potential[0];
		}
		p.magneticResponse=false;
		return result;
	};
	logPrintf("KI response optimization: %s (frozen XC kernel, direct complex potentials, reused GGA gradients).\n",
		wannier.koopmansOptimizeResponse ? "enabled" : "disabled");
	logPrintf("KI screening solver: %s%s.\n",p.nestedResponse ? "nested" : "coupled MINRES",
		p.nestedResponse ? (p.independentBandCG ? " with independent-band CG / MINRES fallback" : " with MINRES inner solves") : "");
	// Share source construction with the planning pass so its reuse decisions
	// use exactly the same density, curvature, ordering and cache budget.
	auto orbitalSource=[&](int n,int iq,const std::vector<MeshPoint>& plus) {
		if(fftCorrelation) return correlation->source(n,iq);
		complexScalarField rho;
		nullToZero(rho,e.gInfo); initZero(rho);
		for(int ik=0;ik<nk;ik++) {
			complexScalarField shifted=clone(orbital[plus[ik].index][n]);
			multiplyBlochPhase(shifted,-plus[ik].wrap);
			rho+=(1./nk)*conj(orbital[ik][n])*shifted;
		}
		return rho;
	};
	int plannedQCount=0,plannedPairCount=0;
	logPrintf("KI planning independent q points (orbital-source symmetry checks only).\n"); logFlush();
	if(wannier.koopmansReduceSymmetry && !adaptiveScreening) {
		if(!wannier.koopmansOptimizeResponse) solver.updateExcorrCache();
		ScreeningReuse plan(e,qmesh,true,wannier.koopmansSymmetryTolerance,true,
			wannier.koopmansOptimizeResponse ? wannier.koopmansSymmetryCacheMiB : 256.);
		int lastPlannedStar=-1;
		for(int iq:qOrder) {
			if(qOwner[iq]!=responseGroups.index || !screeningQ[iq]) continue;
			if(wannier.koopmansOptimizeResponse && qStar[iq]!=lastPlannedStar) {
				plan.clearStar(); lastPlannedStar=qStar[iq];
			}
			p.qvec=qmesh[iq];
			std::vector<MeshPoint> plus(nk);
			if(!fftCorrelation) for(int ik=0;ik<nk;ik++) plus[ik]=meshPoint(mesh,mesh[ik]+qmesh[iq]);
			bool independent=false;
			for(int n=0;n<nCenters;n++) {
				auto rho=orbitalSource(n,iq,plus);
				auto potential=orbitalPotentials(rho,qmesh[iq]);
				double bareTerm=0.,response=0.;
				for(const auto& vChannel:potential) bareTerm+=dot(rho,vChannel).real()/nk;
				if(!plan.lookup(rho,occupations[n],iq,bareTerm,response,n)) {
					independent=true; plannedPairCount++;
					plan.record(rho,n,occupations[n],iq,bareTerm,0.);
				}
			}
			if(independent) plannedQCount++;
		}
	} else {
		for(int iq:qOrder) if(qOwner[iq]==responseGroups.index && screeningQ[iq]) plannedQCount++;
		plannedPairCount=plannedQCount*nCenters;
	}
	int workload[2]={mpiWorld->isHead() ? plannedQCount : 0,mpiWorld->isHead() ? plannedPairCount : 0};
	responseWorld->allReduce(workload,2,MPIUtil::ReduceSum);
	logPrintf("KI response workload: %d independent q points across %d groups; %d independent orbital-q pairs, %d sector solves.\n",
		workload[0],nQGroups,workload[1],workload[1]*(p.kiSpinResponse ? 2 : 1));
	logPrintf("KI group %d/%d: %d independent q points assigned.\n",responseGroups.index+1,nQGroups,plannedQCount); logFlush();
	int computedQCount=0,lastComputedQ=-1;
	auto printScreeningProgress=[&](int iq,int orbital,const char* activity,double elapsedSeconds=-1.) {
		logPrintf("KI group %d/%d: %s q %d/%d, mesh q %d/%d [%.8g %.8g %.8g], orbital %d/%d: %s",
			responseGroups.index+1,nQGroups,symmetryAudit ? "audit" : "computed",
			computedQCount,plannedQCount,iq+1,nk,qmesh[iq][0],qmesh[iq][1],qmesh[iq][2],orbital+1,nCenters,activity);
		if(elapsedSeconds>=0.) logPrintf(" completed in %.2f s.\n",elapsedSeconds);
		else logPrintf(".\n");
		logFlush(); // Flush both start and completion messages promptly.
	};
	// Bare conduction corrections depend only on frozen orbital sources. Form
	// one orbital's owned-q potentials, convolve them across the mesh, and
	// release those fields before moving to the next orbital.
	if(fftCorrelation && !symmetryAudit && !wannier.koopmansScreeningOnly) for(int n:emptyCenters) {
		std::vector<complexScalarField> potentials(nk);
		for(int iq=0;iq<nk;iq++) if(qOwner[iq]==responseGroups.index) {
			complexScalarField rho=correlation->source(n,iq);
			auto potential=orbitalPotentials(rho,qmesh[iq]);
			potentials[iq]=(1./e.gInfo.dV)*potential[0];
			if(p.kiSpinResponse) potentials[iq]+=(1./e.gInfo.dV)*potential[1];
			if(regularCoulomb && qmesh[iq].length_squared()<1e-16) {
				auto rhoG=J(rho);
				potentials[iq]+=I((*regularCoulomb)(rhoG,qmesh[iq],0.)-e.coulomb->applyBloch(rhoG,qmesh[iq]));
			}
		}
		correlation->accumulateConduction(n,potentials,emptyCenters,correction);
		logPrintf("KI bare conduction convolution: orbital %d/%d complete.\n",n+1,nCenters); logFlush();
	}
	// A coarse ratio is only a mesh estimate. Keep every completed scalar
	// response as the mesh grows; never interpolate it or clip its sign.
	std::vector<bool> responseDone(nk,false);
	struct ScreeningStage {
		vector3<int> mesh;
		std::vector<double> bare,relaxed;
		diagMatrix alpha;
		double change;
	};
	std::vector<ScreeningStage> screeningStages;
	matrix convergenceCorrection;
	std::vector<double> convergenceBare;
	int smallChanges=0;
	for(int screeningPass=0;;screeningPass++) {
	const bool firstPass=screeningPass==0;
	if(!firstPass) {
		std::fill(screeningBare.begin(),screeningBare.end(),0.);
		std::fill(relaxation.begin(),relaxation.end(),0.);
		std::fill(screeningSingular.begin(),screeningSingular.end(),0.);
		computedQCount=0; lastComputedQ=-1;
		plannedQCount=0;
		for(int iq:qOrder) if(qOwner[iq]==responseGroups.index && screeningQ[iq] && !responseDone[iq]) plannedQCount++;
		logPrintf("KI screening refinement: mesh %d %d %d (%d/%d q points); at most %d new q points in group %d.\n",
			screeningMesh[0],screeningMesh[1],screeningMesh[2],nqScreen,nk,plannedQCount,responseGroups.index+1);
	}
	if(adaptiveScreening && firstPass)
		logPrintf("KI adaptive workload counts are upper bounds before orbital-source symmetry reuse.\n");
	reuse.clearStar();
	int lastStar=-1;
	for(int iq:qOrder)
	{
		if(qOwner[iq]!=responseGroups.index) continue;
		if(symmetryAudit && !screeningQ[iq]) continue;
		if(!firstPass && !screeningQ[iq]) continue;
		if(wannier.koopmansOptimizeResponse && wannier.koopmansReduceSymmetry && qStar[iq]!=lastStar) {
			reuse.clearStar(); lastStar=qStar[iq];
		}
		vector3<> q=qmesh[iq];
		p.qvec=q;
		bool qPrepared=false;
		auto prepareQ=[&]() {
		if(qPrepared) return;
		solver.clearBandCoarseSpace();
		projectorCache.clearOffsets(p);
		p.setupkpoints(system,e.eInfo);
		p.initInc(p.Cinc,2*e.eInfo.nStates,nResponse,&e.eInfo);
		p.initInc(p.dC,2*e.eInfo.nStates,nResponse,&e.eInfo);
		p.dVsclocpq.resize(1); p.dVsclocmq.resize(1);
		p.dnpq.resize(1); p.dnmq.resize(1);
		nullToZero(p.dnpq,e.gInfo); nullToZero(p.dnmq,e.gInfo);
		// The response projector includes occupied states only, even in the
		// optional full-column comparison path. Select the mapped k point's
		// occupied orbitals rather than assuming they are contiguous columns.
		for(int state=e.eInfo.qStart; state<e.eInfo.qStop; state++)
		for(int sign=0; sign<(p.oneSidedResponse ? 1 : 2); sign++)
		{
			int index=state+sign*e.eInfo.nStates;
			const vector3<>& target=sign ? p.kminusq_vectors[state].k : p.kplusq_vectors[state].k;
			MeshPoint mapped=meshPoint(mesh,target);
			p.Cinc[index].zero();
			if(wannier.koopmansOptimizeResponse && wannier.koopmansOccupiedResponse) {
				const auto& source=info.isMine(mapped.index) ? system.eVars.C[mapped.index] : offsetSource[mapped.index];
				const auto& targetBasis=*p.Cinc[index].basis;
				bool shiftedOrdering=source.basis->nbasis==targetBasis.nbasis;
				const auto* sourceG=source.basis->iGarr.data(); const auto* targetG=targetBasis.iGarr.data();
				vector3<int> wrap;
				for(int d=0;d<3;d++) wrap[d]=int(round(mapped.wrap[d]));
				for(size_t i=0;shiftedOrdering && i<targetBasis.nbasis;i++) shiftedOrdering=(sourceG[i]-wrap)==targetG[i];
				if(shiftedOrdering) {
					callPref(eblas_copy)(p.Cinc[index].dataPref(),source.dataPref(),source.nData());
					continue;
				}
			}
			for(int b=0; b<nResponse; b++) if(fullF[mapped.index][b]>0.5)
			{
				complexScalarField u=I(canonical[mapped.index].getColumn(compactCanonical ? b : responseBands[mapped.index][b],0));
				multiplyBlochPhase(u,-mapped.wrap);
				p.Cinc[index].setColumn(b,0,J(u));
			}
		}
		p.offsetStatesReady=true;
		p.groundStateReady=true;
		if(coarseCount) {
			size_t coarseBytes=coarseSourceBytes;
			for(int state=info.qStart;state<info.qStop;state++) for(int sign=0;sign<(p.oneSidedResponse ? 1 : 2);sign++)
				coarseBytes+=size_t(coarseCount)*(p.Cinc[state+sign*info.nStates].colLength()+info.nBands)*sizeof(complex);
			bool fits=double(coarseBytes)<=coarseBudgetMiB*1024.*1024.;
			mpiWorld->allReduce(fits,MPIUtil::ReduceLAnd);
			if(fits) {
				std::vector<ColumnBundle> coarse(2*info.nStates);
				for(int state=info.qStart;state<info.qStop;state++) for(int sign=0;sign<(p.oneSidedResponse ? 1 : 2);sign++) {
					int index=state+sign*info.nStates;
					const auto& projector=p.Cinc[index];
					coarse[index]=projector.similar(coarseCount);
					MeshPoint mapped=meshPoint(mesh,projector.qnum->k);
					const auto& source=coarseSource[mapped.index];
					vector3<int> wrap; for(int d=0;d<3;d++) wrap[d]=int(round(mapped.wrap[d]));
					bool shifted=source.basis->nbasis==projector.basis->nbasis;
					const auto* sourceG=source.basis->iGarr.data(); const auto* targetG=projector.basis->iGarr.data();
					for(size_t i=0;shifted && i<projector.basis->nbasis;i++) shifted=(sourceG[i]-wrap)==targetG[i];
					if(shifted) {
						callPref(eblas_copy)(coarse[index].dataPref(),source.dataPref(),source.nData());
						continue;
					}
					for(int b=0;b<coarseCount;b++) {
						auto u=I(source.getColumn(b,0));
						multiplyBlochPhase(u,-mapped.wrap);
						coarse[index].setColumn(b,0,J(changeGrid(u,*projector.basis->gInfo)));
					}
				}
				solver.setBandCoarseSpace(std::move(coarse));
				logPrintf("KI empty-state preconditioner: %d columns, %.1f MiB/rank (budget %.1f MiB/rank).\n",
					coarseCount,coarseBytes/(1024.*1024.),coarseBudgetMiB);
			} else logPrintf("KI empty-state preconditioner: memory budget insufficient; kinetic fallback.\n");
		}
		if(!wannier.koopmansOptimizeResponse) solver.updateExcorrCache();
		p.dVext=std::make_shared<VextPerturbation>(system);
		qPrepared=true;
		};
		// Symmetry-reused scalar responses need no offset wavefunctions/bases.
		if(!wannier.koopmansOptimizeResponse && screeningQ[iq]) prepareQ();
		std::vector<MeshPoint> plus(nk);
		if(!fftCorrelation) for(int ik=0; ik<nk; ik++) plus[ik]=meshPoint(mesh,mesh[ik]+q);
		for(int n=0; n<nCenters; n++)
		{
			if(wannier.koopmansVerbose) {
				logPrintf("KI orbital %d/%d, q %d/%d [%.8g %.8g %.8g]\n",n+1,nCenters,iq+1,nk,q[0],q[1],q[2]); logFlush();
			}
			complexScalarField rho=orbitalSource(n,iq,plus);
			auto potential=orbitalPotentials(rho,q);
			// getdVsclocPsi returns a quadrature-weighted potential, JdagOJ V.
			// A spin-up orbital has charge and magnetization source rho. Its
			// self curvature contracts K_c+K_m; only K_c includes Hartree.
			double bareTerm=0.;
			for(const auto& vChannel:potential) bareTerm+=dot(rho,vChannel).real()/nk;
			if(firstPass) {
				bare[n]+=bareTerm;
				bareQ[size_t(iq)*nCenters+n]=nk*bareTerm;
			}
			if(screeningQ[iq]) screeningBare[n]+=screeningWeight*bareTerm;
			complexScalarField v=(1./e.gInfo.dV)*potential[0];
			if(p.kiSpinResponse) v+=(1./e.gInfo.dV)*potential[1];
			double response=0.;
			if(screeningQ[iq] && responseDone[iq]) {
				response=responseQ[size_t(iq)*nCenters+n]/nk;
				// Seed the current star cache without counting a second solve.
				reuse.record(rho,n,occupations[n],iq,bareTerm,response,false);
			}
			else if(screeningQ[iq] && !reuse.lookup(rho,occupations[n],iq,bareTerm,response,n))
			{
				// Count a q only when an orbital actually needs an independent
				// response. Reused q points do not advance this group-local index;
				// other orbitals and spin sectors at the same q keep its index.
				if(iq!=lastComputedQ) { lastComputedQ=iq; computedQCount++; }
				if(symmetryAudit) printScreeningProgress(iq,n,"independent pair (audit)");
				if(!symmetryAudit) prepareQ();
				for(size_t channel=0;channel<potential.size();channel++) {
					if(symmetryAudit) continue;
					p.magneticResponse=(channel==1);
					printScreeningProgress(iq,n,p.magneticResponse ? "magnetic response" : "charge response");
					p.dVext->dVextpq=complexScalarFieldArray(1,(1./e.gInfo.dV)*potential[channel]);
					p.dVext->dVextmq=conj(p.dVext->dVextpq);
					const double responseStart=clock_us();
					solver.solvePerturbation();
					if(!std::isfinite(p.lastResponseResidual) || p.lastResponseResidual>10*wannier.koopmansTolerance)
						die("Koopmans screening response did not converge (%s sector). Increase nIterations or improve the DFT state.\n",
							p.magneticResponse ? "magnetic" : "charge");
					response+=dot(p.dnpq[0],potential[channel]).real()/nk;
					printScreeningProgress(iq,n,p.magneticResponse ? "magnetic response" : "charge response",
						1e-6*(clock_us()-responseStart));
				}
				p.magneticResponse=false;
				reuse.record(rho,n,occupations[n],iq,bareTerm,response);
			}
			relaxation[n]+=screeningWeight*response;
			responseQ[size_t(iq)*nCenters+n]=nk*response;
			complexScalarField vBare=v;
			if(regularCoulomb && q.length_squared()<1e-16)
			{
				complexScalarFieldTilde rhoG=J(rho);
				complexScalarField delta=I((*regularCoulomb)(rhoG,q,0.)-e.coulomb->applyBloch(rhoG,q));
				double image=e.gInfo.dV*dot(rho,delta).real();
				if(firstPass) {
					singular[n]+=image/nk;
					imageQ[size_t(iq)*nCenters+n]=image;
				}
				complexScalarField screeningDelta=coarseScreening
					? I((*screeningCoulomb)(rhoG,q,0.)-e.coulomb->applyBloch(rhoG,q)) : delta;
				double screeningImage=e.gInfo.dV*dot(rho,screeningDelta).real();
				screeningSingular[n]+=screeningImage/nqScreen;
				screeningImageQ[size_t(iq)*nCenters+n]=screeningImage;
				vBare+=delta;
			}
			// SI Eq. 6: <V_(n,q)|conj(u_(i,k)) u_(n,k+q)> / Nq.
			// Occupied columns have only the constant -Delta_n/2 contribution.
			if(firstPass && !symmetryAudit && !wannier.koopmansScreeningOnly && !occupations[n] && !fftCorrelation) for(int ik=0; ik<nk; ik++)
			{
				complexScalarField shifted=clone(orbital[plus[ik].index][n]);
				multiplyBlochPhase(shifted,-plus[ik].wrap);
				if(wannier.koopmansOptimizeResponse && conductionOrbitals[ik]) {
					complexScalarField weighted=conj(vBare)*shifted;
					matrix values(emptyCenters.size(),1,isGpuEnabled());
					const matrix& packed=conductionOrbitals[ik];
					callPref(eblas_zgemm)(CblasConjTrans,CblasNoTrans,emptyCenters.size(),1,e.gInfo.nr,
						e.gInfo.dV/nk,packed.dataPref(),e.gInfo.nr,weighted->dataPref(),e.gInfo.nr,0.,values.dataPref(),emptyCenters.size());
					#ifdef GPU_ENABLED
					matrixColumnAccumulate_gpu(emptyCenters.size(),emptyRows.dataGpu(),nCenters,n,values.dataGpu(),correction[ik].dataGpu());
					#else
					for(size_t m=0;m<emptyCenters.size();m++) correction[ik].data()[correction[ik].index(emptyCenters[m],n)]+=values.data()[m];
					#endif
					continue;
				}
				for(int m=0; m<nCenters; m++) if(!occupations[m])
				{
					complex value=e.gInfo.dV*dot(vBare,conj(orbital[ik][m])*shifted)/nk;
					correction[ik].set(m,n,correction[ik](m,n)+value);
				}
			}
		}
		p.dVext.reset();
		if(screeningQ[iq]) responseDone[iq]=true;
	}
	if(!adaptiveScreening) break;

	// Collect replicas only from group heads. The original communicator is
	// retained while the Sternheimer solver uses group-local communicators.
	auto collect=[&](std::vector<double> values) {
		if(nQGroups>1) {
			if(!mpiWorld->isHead()) std::fill(values.begin(),values.end(),0.);
			responseWorld->allReduceData(values,MPIUtil::ReduceSum);
		}
		return values;
	};
	if(firstPass) {
		convergenceBare=bare;
		for(int n=0;n<nCenters;n++) convergenceBare[n]+=singular[n];
		convergenceBare=collect(convergenceBare);
		convergenceCorrection=zeroes(nCenters*nCenters,nk);
		if(nQGroups==1 || mpiWorld->isHead()) for(int ik=0;ik<nk;ik++)
			callPref(eblas_copy)(convergenceCorrection.dataPref()+convergenceCorrection.index(0,ik),correction[ik].dataPref(),correction[ik].nData());
		if(nQGroups>1) responseWorld->allReduceData(convergenceCorrection,MPIUtil::ReduceSum);
		// Complete exactly the same upper-triangle convention as the final
		// Hamiltonian, including its full-mesh diagonal -bare/2 term.
		complex* entries=convergenceCorrection.data();
		for(int ik=0;ik<nk;ik++) for(int n=0;n<nCenters;n++)
			entries[convergenceCorrection.index(n+n*nCenters,ik)]-=0.5*convergenceBare[n];
	}
	auto stageBare=screeningBare, stageRelaxed=screeningBare;
	for(int n=0;n<nCenters;n++) {
		stageBare[n]+=screeningSingular[n];
		stageRelaxed[n]+=relaxation[n]+(wannier.koopmansEpsilon ? screeningSingular[n]/wannier.koopmansEpsilon : 0.);
	}
	stageBare=collect(stageBare); stageRelaxed=collect(stageRelaxed);
	diagMatrix stageAlpha(nCenters);
	bool acceptable=true;
	for(int n=0;n<nCenters;n++) {
		stageAlpha[n]=std::isfinite(stageBare[n]) && fabs(stageBare[n])>=1e-12 ? stageRelaxed[n]/stageBare[n] : NAN;
		// Negative coefficients remain available for diagnosis at the full
		// mesh; they never justify accepting a coarse approximation.
		acceptable=acceptable && std::isfinite(stageAlpha[n]) && stageAlpha[n]>=0.;
		logPrintf("KI screening stage %d orbital %d: bare=%.12g relaxed=%.12g alpha=%.12g.\n",
			screeningPass+1,n,stageBare[n],stageRelaxed[n],stageAlpha[n]);
	}
	double change=INFINITY;
	if(!screeningStages.empty()) {
		// ||Delta H||_2 <= ||Delta H||_F bounds the change in each native-k
		// eigenvalue between these two meshes. It does NOT bound the residual
		// integration error or arbitrary-k Wannier interpolation error.
		change=0.;
		const auto& previous=screeningStages.back().alpha;
		const complex* entries=convergenceCorrection.data();
		for(int ik=0;ik<nk;ik++) {
			double normSquared=0.;
			for(int n=0;n<nCenters;n++) for(int m=0;m<=n;m++) {
				complex value=entries[convergenceCorrection.index(m+n*nCenters,ik)]*(stageAlpha[n]-previous[n]);
				double elementNorm=value.real()*value.real()+(m==n ? 0. : value.imag()*value.imag());
				normSquared+=(m==n ? 1. : 2.)*elementNorm;
			}
			if(!std::isfinite(normSquared)) { change=INFINITY; break; }
			change=std::max(change,sqrt(normSquared));
		}
	}
	smallChanges=acceptable && change<=wannier.koopmansScreeningEnergyTolerance*eV ? smallChanges+1 : 0;
	screeningStages.push_back({screeningMesh,stageBare,stageRelaxed,stageAlpha,change});
	logPrintf("KI screening stage %d: mesh %d %d %d, Hamiltonian change bound %.9g eV, %d/2 successive small changes.\n",
		screeningPass+1,screeningMesh[0],screeningMesh[1],screeningMesh[2],change/eV,smallChanges);
	bool stop=!coarseScreening || smallChanges>=2;
	responseWorld->bcast(stop); // One decision for all solver communicators.
	if(stop) {
		logPrintf("KI adaptive screening finished: %s.\n",coarseScreening ? "mesh convergence estimate satisfied" : "full electronic q mesh reached");
		break;
	}
	// Grow along nested divisors, preserving all previously sampled q's.
	// This also handles anisotropic and non-power-of-two electronic meshes.
	for(int d=0;d<3;d++) {
		int ratio=folding[d]/screeningMesh[d];
		if(ratio==1) continue;
		int factor=2;
		while(ratio%factor) factor++;
		screeningMesh[d]*=factor;
	}
	nqScreen=screeningMesh[0]*screeningMesh[1]*screeningMesh[2];
	coarseScreening=nqScreen!=nk; screeningWeight=double(nk)/nqScreen;
	std::fill(screeningQ.begin(),screeningQ.end(),true);
	for(int iq=0;iq<nk;iq++) for(int d=0;d<3;d++) {
		double index=qmesh[iq][d]*screeningMesh[d];
		if(fabs(index-round(index))>1e-8) screeningQ[iq]=false;
	}
	updateScreeningCoulomb();
	} // screening refinement passes

	logPrintf("KI group %d/%d: %s q points total: %d.\n",
		responseGroups.index+1,nQGroups,symmetryAudit ? "audit" : "computed",computedQCount);
	logFlush();
	const bool groupHead=mpiWorld->isHead();
	solver.clearBandCoarseSpace();
	coarseSource.clear();
	projectorCache.clear();
	fullState.restore();
	responseGroups.restore();
	p.solverParams.fpLog=globalLog;
	if(nQGroups>1)
	{
		// Each group's result is replicated within its communicator. Only its
		// head contributes to the world sum, including the Gamma correction.
		if(!groupHead)
		{
			std::fill(bare.begin(),bare.end(),0.);
			std::fill(screeningBare.begin(),screeningBare.end(),0.);
			std::fill(relaxation.begin(),relaxation.end(),0.);
			std::fill(singular.begin(),singular.end(),0.);
			std::fill(bareQ.begin(),bareQ.end(),0.);
			std::fill(responseQ.begin(),responseQ.end(),0.);
			std::fill(imageQ.begin(),imageQ.end(),0.);
			std::fill(screeningSingular.begin(),screeningSingular.end(),0.);
			std::fill(screeningImageQ.begin(),screeningImageQ.end(),0.);
		}
		mpiWorld->allReduceData(bare,MPIUtil::ReduceSum);
		mpiWorld->allReduceData(screeningBare,MPIUtil::ReduceSum);
		mpiWorld->allReduceData(relaxation,MPIUtil::ReduceSum);
		mpiWorld->allReduceData(singular,MPIUtil::ReduceSum);
		mpiWorld->allReduceData(bareQ,MPIUtil::ReduceSum);
		mpiWorld->allReduceData(responseQ,MPIUtil::ReduceSum);
		mpiWorld->allReduceData(imageQ,MPIUtil::ReduceSum);
		mpiWorld->allReduceData(screeningSingular,MPIUtil::ReduceSum);
		mpiWorld->allReduceData(screeningImageQ,MPIUtil::ReduceSum);
		matrix packed=zeroes(nCenters*nCenters,nk);
		if(groupHead) for(int ik=0;ik<nk;ik++)
			callPref(eblas_copy)(packed.dataPref()+packed.index(0,ik),correction[ik].dataPref(),correction[ik].nData());
		mpiWorld->allReduceData(packed,MPIUtil::ReduceSum);
		for(int ik=0;ik<nk;ik++)
			callPref(eblas_copy)(correction[ik].dataPref(),packed.dataPref()+packed.index(0,ik),correction[ik].nData());
	}
	reuse.report(nQGroups>1 ? mpiWorld : nullptr,nQGroups>1 ? groupHead : true,symmetryAudit);
	if(symmetryAudit) {
		for(int n=0;n<nCenters;n++)
			logPrintf("KI audited screening curvature %d: bare=%.15g (%d q points; no response).\n",
				n,screeningBare[n]+screeningSingular[n],nqScreen);
		logPrintf("KI symmetry audit complete: no response solves or KI spectral outputs were produced.\n");
		return;
	}
	if(mpiWorld->isHead()) {
		FILE* resolved=openOutput(wannier.getFilename(Wannier::FilenameDump,"koopmansResponse",&iSpin));
		fprintf(resolved,"# iq orbital qx qy qz bare_Ha response_Ha image_Ha screened screening_image_Ha; unweighted q contributions\n");
		for(int iq=0;iq<nk;iq++) for(int n=0;n<nCenters;n++) {
			size_t index=size_t(iq)*nCenters+n;
			fprintf(resolved,"%d %d %.15g %.15g %.15g %.15g %.15g %.15g %d %.15g\n",iq,n,qmesh[iq][0],qmesh[iq][1],qmesh[iq][2],bareQ[index],responseQ[index],imageQ[index],screeningQ[iq] ? 1 : 0,screeningImageQ[index]);
		}
		fclose(resolved);
	}

	FILE* screening=mpiWorld->isHead() ? openOutput(wannier.getFilename(Wannier::FilenameDump,"koopmansScreening",&iSpin)) : nullptr;
	if(screening) {
		fprintf(screening,"# functional KI_second_order\n");
		fprintf(screening,"# epsilon_infinity %.15g; electrostatics %s\n",wannier.koopmansEpsilon,
			wannier.koopmansEpsilon ? "probe_charge_ewald" : "neutralizing_background");
		fprintf(screening,"# spin_treatment %s\n",wannier.koopmansSpinResponse ? "single_spin_charge_and_magnetic" : "restricted_charge_only");
		const auto& reportedMesh=importedScreening ? alphaSourceMesh : screeningMesh;
		fprintf(screening,"# screening_q_mesh %d %d %d; full_k_mesh %d %d %d\n",reportedMesh[0],reportedMesh[1],reportedMesh[2],folding[0],folding[1],folding[2]);
		fprintf(screening,"# screening_mode %s; requested_q_mesh %d %d %d; energy_tolerance_eV %.15g\n",
			importedScreening ? "imported" : (adaptiveScreening ? "adaptive" : (coarseScreening ? "fixed" : "full")),
			requestedScreeningMesh[0],requestedScreeningMesh[1],requestedScreeningMesh[2],wannier.koopmansScreeningEnergyTolerance);
		if(importedScreening) fprintf(screening,"# alpha_source %s\n",wannier.koopmansAlphaFilename.c_str());
		if(!wannier.koopmansReferenceDensity.empty()) fprintf(screening,"# reference_density %s\n",wannier.koopmansReferenceDensity.c_str());
		for(size_t stage=0;stage<screeningStages.size();stage++) {
			const auto& result=screeningStages[stage];
			fprintf(screening,"# screening_stage %zu %d %d %d change_bound_eV %.15g\n",stage+1,result.mesh[0],result.mesh[1],result.mesh[2],result.change/eV);
			for(int n=0;n<nCenters;n++) fprintf(screening,"# screening_stage_curvature %zu %d %.15g %.15g %.15g\n",
				stage+1,n,result.bare[n],result.relaxed[n],result.alpha[n]);
		}
		fprintf(screening,"# orbital occupation bare_curvature_Ha relaxed_curvature_Ha alpha image_correction_Ha\n");
	}
	diagMatrix alpha(nCenters);
	const auto& alphaMesh=importedScreening ? alphaSourceMesh : screeningMesh;
	for(int n=0; n<nCenters; n++)
	{
		double screeningRelaxed=screeningBare[n]+relaxation[n]+(wannier.koopmansEpsilon ? screeningSingular[n]/wannier.koopmansEpsilon : 0.);
		screeningBare[n]+=screeningSingular[n];
		if(importedScreening) { screeningBare[n]=importedBare[n]; screeningRelaxed=importedRelaxed[n]; }
		bare[n]+=singular[n];
		if(!std::isfinite(bare[n]) || fabs(bare[n])<1e-12 || !std::isfinite(screeningBare[n]) || fabs(screeningBare[n])<1e-12 || !std::isfinite(screeningRelaxed))
			die("Invalid or vanishing KI orbital curvature.\n");
		alpha[n]=importedScreening ? importedAlpha[n] : screeningRelaxed/screeningBare[n]; // Eq. 18; do not silently clamp alpha.
		if(alpha[n]<0.) logPrintf("WARNING: KI orbital %d has negative screening alpha %.9g on the %d %d %d q mesh; check electrostatic and mesh convergence.\n",
			n,alpha[n],alphaMesh[0],alphaMesh[1],alphaMesh[2]);
		double relaxed=alpha[n]*bare[n];
		if(coarseScreening || importedScreening) {
			int sourceCount=importedScreening ? alphaSourceMesh[0]*alphaSourceMesh[1]*alphaSourceMesh[2] : nqScreen;
			logPrintf("KI screening curvature %d: bare=%.12g relaxed=%.12g (%d source q points).\n",n,screeningBare[n],screeningRelaxed,sourceCount);
			if(screening) fprintf(screening,"# screening_curvature %d %.15g %.15g\n",n,screeningBare[n],screeningRelaxed);
		}
		logPrintf("KI orbital %d: f=%.0f bare=%.12g relaxed=%.12g alpha=%.12g\n",n,occupations[n],bare[n],relaxed,alpha[n]);
		if(screening) fprintf(screening,"%d %.0f %.15g %.15g %.15g %.15g\n",n,occupations[n],bare[n],relaxed,alpha[n],singular[n]);
		for(int ik=0; ik<nk; ik++) correction[ik].set(n,n,correction[ik](n,n)-0.5*bare[n]);
	}
	if(screening) fclose(screening);
	if(wannier.koopmansScreeningOnly) {
		logPrintf("KI screening-only calculation complete; no KI spectral Hamiltonian written.\n");
		return;
	}

	matrix hTilde=zeroes(nCenters*nCenters,phase.nRows()), deltaTilde=zeroes(nCenters*nCenters,phase.nRows());
	FILE* bands=mpiWorld->isHead() ? openOutput(wannier.getFilename(Wannier::FilenameDump,"koopmansBands",&iSpin)) : nullptr;
	if(bands) fprintf(bands,"# kx ky kz followed by nCenters KI eigenvalues in Hartree\n");
	int local=0;
	double antiHermitian=0.;
	for(int ik=0; ik<nk; ik++)
	{
		matrix delta=correction[ik]*alpha; // screen each orbital's column
		antiHermitian=std::max(antiHermitian,nrm2(delta-dagger(delta)));
		// Match the reference kcw.x convention: screen the upper triangle,
		// then complete the Hermitian matrix. Report the unsymmetrized norm
		// because approximate MLWFs need not satisfy the Pederson condition.
		for(int n=0; n<nCenters; n++)
		{
			delta.set(n,n,delta(n,n).real());
			for(int m=0; m<n; m++) delta.set(n,m,delta(m,n).conj());
		}
		matrix h=base[ik]+delta;
		if(isMine_q(ik,iSpin))
		{
			callPref(eblas_copy)(hTilde.dataPref()+hTilde.index(0,local),h.dataPref(),h.nData());
			callPref(eblas_copy)(deltaTilde.dataPref()+deltaTilde.index(0,local),delta.dataPref(),delta.nData());
			local++;
		}
		if(bands)
		{
			matrix vectors; diagMatrix values; h.diagonalize(vectors,values);
			fprintf(bands,"%.15g %.15g %.15g",mesh[ik][0],mesh[ik][1],mesh[ik][2]);
			for(double value: values) fprintf(bands," %.15g",value);
			fprintf(bands,"\n");
		}
	}
	if(bands) fclose(bands);
	logPrintf("KI maximum unsymmetrized anti-Hermitian norm: %.6g Hartree\n",antiHermitian);
	// Spinless MLWFs can still be complex (especially disentangled conduction
	// orbitals). Never discard a physically significant imaginary hopping.
	matrix hRealSpace=hTilde*phase, deltaRealSpace=deltaTilde*phase;
	mpiWorld->allReduceData(hRealSpace,MPIUtil::ReduceSum);
	mpiWorld->allReduceData(deltaRealSpace,MPIUtil::ReduceSum);
	auto imaginaryNorm=[](const matrix& m)
	{
		return callPref(eblas_dnrm2)(m.nData(),((const double*)m.dataPref())+1,2);
	};
	bool kiReal=realPartOnly
		&& imaginaryNorm(hRealSpace)<=1e-12+1e-10*nrm2(hRealSpace)
		&& imaginaryNorm(deltaRealSpace)<=1e-12+1e-10*nrm2(deltaRealSpace);
	if(mpiWorld->isHead())
	{
		FILE* format=openOutput(wannier.getFilename(Wannier::FilenameDump,"koopmansFormat",&iSpin));
		fprintf(format,"%s %d %d\n",kiReal ? "real64" : "complex128",phase.nCols(),nCenters);
		fclose(format);
	}
	logPrintf("KI real-space Hamiltonian format: %s\n",kiReal ? "real64" : "complex128");
	dumpWannierized(hTilde,phase,"mlwfHki",kiReal,iSpin);
	dumpWannierized(deltaTilde,phase,"mlwfKI",kiReal,iSpin);
}
