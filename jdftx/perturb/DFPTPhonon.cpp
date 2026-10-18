/* Primitive-cell DFPT phonons and validation. GPL v3 or later; part of JDFTx. */
#include <perturb/DFPTPhonon.h>
#include <perturb/DFPTGroundState.h>
#include <perturb/PerturbationSolver.h>
#include <electronic/Everything.h>
#include <electronic/ElecMinimizer.h>
#include <electronic/ColumnBundleTransform.h>
#include <electronic/ColumnBundleFFT.h>
#include <core/Operators.h>
#include <tuple>
#include <commands/parser.h>
#include <commands/command.h>
#include <core/LatticeUtils.h>
#include <core/Units.h>
#include <gsl/gsl_sf_erf.h>
#include <perturb/DFPTCheckpoint.h>

namespace {
struct QSymmetry {
	SpaceGroupOp op;
	matrix3<> cartesian;
	std::vector<int> atomMap;
	std::vector<vector3<int>> cellShift;
};

int qGridIndex(const vector3<>& q,const vector3<int>& grid)
{
	vector3<int> index;
	for(int d=0;d<3;d++) {
		double scaled=q[d]*grid[d];
		int nearest=int(round(scaled));
		if(fabs(scaled-nearest)>1e-8) return -1;
		index[d]=(nearest%grid[d]+grid[d])%grid[d];
	}
	return (index[0]*grid[1]+index[1])*grid[2]+index[2];
}

bool sameReciprocalPoint(vector3<> a,const vector3<>& b)
{
	a-=b;
	for(int d=0;d<3;d++) a[d]-=round(a[d]);
	return a.length_squared()<1e-16;
}

// Use the native phonon / Wannier ordering: Cartesian mode, target k,
// source k, then a column-major band matrix. Only the Gamma-centered subset
// commensurate with the phonon supercell is stored.
struct ElectronPhononMesh {
	std::vector<vector3<>> kpoints;
	std::vector<int> indices;
	ElectronPhononMesh(Everything& e,const vector3<int>& grid)
	{
		if(!e.coulombParams.supercell) e.updateSupercell(true);
		for(const auto& k:e.coulombParams.supercell->kmesh) if(qGridIndex(k,grid)>=0) {
			int source=-1;
			for(int s=0;s<e.eInfo.nStates;s++)
				if(sameReciprocalPoint(k,e.eInfo.qnums[s].k)) {source=s; break;}
			if(source<0) die("Electron-phonon export requires the full electronic k mesh (symmetries none).\n");
			kpoints.push_back(k); indices.push_back(source);
		}
		if(int(indices.size())!=grid[0]*grid[1]*grid[2])
			die("Electron-phonon export requires a Gamma-centered electronic k mesh containing the phonon q grid. Use kpoint-folding dimensions divisible by phonon-dfpt dimensions.\n");
	}
};

// Independent q groups must export the same electronic gauge. The first
// group's eigenstates define that gauge, including degenerate band rotations.
void synchronizeElectronPhononStates(Everything& e,const MPIUtil* parent)
{
	if(!parent) return;
	const ElecInfo& ei=e.eInfo;
	for(int k=0;k<ei.nStates;k++) {
		int root=ei.whose(k); parent->bcast(root);
		ColumnBundle received;
		ColumnBundle& C=ei.isMine(k) ? e.eVars.C[k] : received;
		if(!ei.isMine(k)) C.init(ei.nBands,e.basis[k].nbasis,&e.basis[k],&ei.qnums[k],isGpuEnabled());
		diagMatrix energies=ei.isMine(k) ? e.eVars.Hsub_eigs[k] : diagMatrix(ei.nBands);
		diagMatrix fillings=ei.isMine(k) ? e.eVars.F[k] : diagMatrix(ei.nBands);
		parent->bcastData(C,root);
		parent->bcastData(energies,root);
		parent->bcastData(fillings,root);
		if(ei.isMine(k)) {
			e.eVars.Hsub_eigs[k]=energies;
			if(ei.fillingsUpdate==ElecInfo::FillingsHsub) e.eVars.Haux_eigs[k]=energies;
			e.eVars.F[k]=fillings;
			e.iInfo.project(C,e.eVars.VdagC[k]);
		}
	}
	e.eVars.elecEnergyAndGrad(e.ener,nullptr,nullptr,true);
	for(int k=ei.qStart;k<ei.qStop;k++) {
		e.eVars.Hsub[k]=e.eVars.Hsub_eigs[k];
		e.eVars.Hsub_evecs[k]=eye(ei.nBands);
	}
	logPrintf("Electron-phonon export: synchronized ground-state band gauge across q groups.\n");
}

// These caches live only between response solving and export of one q orbit.
struct ElectronPhononExportCache {
	struct Projections { matrix value; std::array<matrix,3> derivative; };
	Everything& e;
	std::vector<ManagedArray<complex>> wavefunctions;
	std::map<std::tuple<int,int,int>,Projections> projections;
	ElectronPhononExportCache(Everything& e,const ElectronPhononMesh& mesh) : e(e),wavefunctions(mesh.indices.size()) {
		size_t used=0,limit=size_t(e.pertInfo.phononEphCacheMiB*1024.*1024.);
		if(e.pertInfo.phononOptimize) for(size_t i=0;i<mesh.indices.size();i++) {
			int k=mesh.indices[i]; if(!e.eInfo.isMine(k) || !e.eVars.C[k]) continue;
			size_t bytes=size_t(e.eInfo.nBands)*e.basis[k].gInfo->nr*sizeof(complex);
			if(bytes>limit-used) continue;
			wavefunctions[i]=realSpaceColumns(e.eVars.C[k]); used+=bytes;
		}
		logPrintf("Electron-phonon real-space cache: %.3f MiB on group head (limit %.3f MiB/rank).\n",used/(1024.*1024.),e.pertInfo.phononEphCacheMiB);
		ManagedMemoryBase::reportAllocationCache("export cache ready");
	}
	const Projections& source(int species,int atom,int k) {
		auto key=std::make_tuple(species,atom,k); auto found=projections.find(key);
		if(found==projections.end()) {
			Projections entry; const ColumnBundle& C=e.eVars.C[k];
			ColumnBundle P=e.iInfo.species[species]->getVatom(C,atom);
			entry.value=P^C;
			for(int direction=0;direction<3;direction++) {
				vector3<> unit; unit[direction]=1.; entry.derivative[direction]=(-D(P,unit))^C;
			}
			found=projections.emplace(key,std::move(entry)).first;
		}
		return found->second;
	}
};

class ElectronPhononWriter {
	Everything& e;
	PerturbationSolver& solver;
	const ElectronPhononMesh& mesh;
	ElectronPhononExportCache& cache;
	bool reciprocity;
	std::vector<Basis> targetBases;
	std::vector<QuantumNumber> targetQnums;
	std::vector<ColumnBundle> targets;
	std::vector<int> targetIndices;
	FILE* fp=nullptr;
public:
	ElectronPhononWriter(Everything& e,PerturbationSolver& solver,const ElectronPhononMesh& mesh,
		ElectronPhononExportCache& cache,bool reciprocity)
	: e(e),solver(solver),mesh(mesh),cache(cache),reciprocity(reciprocity),targetBases(mesh.indices.size()),targetQnums(mesh.indices.size()),
		targets(mesh.indices.size()),targetIndices(mesh.indices.size(),-1)
	{
		const ElecInfo& ei=e.eInfo;
		const PerturbationInfo& p=e.pertInfo;
		for(size_t i=0;i<mesh.indices.size();i++) {
			int k=mesh.indices[i];
			targetQnums[i]=ei.qnums[k]; targetQnums[i].k+=p.qvec;
			for(size_t j=0;j<mesh.indices.size();j++)
				if(sameReciprocalPoint(targetQnums[i].k,mesh.kpoints[j])) {targetIndices[i]=j; break;}
			if(targetIndices[i]<0) die("Missing commensurate electron-phonon target k point.\n");
			if(ei.isMine(k) && (!p.phononGroundState || e.eVars.C[k])) {
				targetBases[i].setup(e.gInfoWfns ? *e.gInfoWfns : e.gInfo,e.iInfo,e.cntrl.Ecut,targetQnums[i].k);
				targets[i].init(ei.nBands,targetBases[i].nbasis,&targetBases[i],&targetQnums[i],isGpuEnabled());
				targets[i].zero();
			}
		}
		if(p.phononGroundState) {
			std::vector<std::pair<int,ColumnBundle*>> requests;
			for(size_t i=0;i<mesh.indices.size();i++) if(targets[i])
				requests.emplace_back(mesh.indices[targetIndices[i]],&targets[i]);
			p.phononGroundState->reconstruct(requests);
		} else for(int s:mesh.indices) {
			ColumnBundle received;
			ColumnBundle& source=ei.isMine(s) ? e.eVars.C[s] : received;
			if(!ei.isMine(s)) source.init(ei.nBands,e.basis[s].nbasis,&e.basis[s],&ei.qnums[s],isGpuEnabled());
			if(mpiWorld->nProcesses()>1) mpiWorld->bcastData(source,ei.whose(s));
			for(size_t i=0;i<mesh.indices.size();i++) {
				int k=mesh.indices[i];
				if(!ei.isMine(k) || mesh.indices[targetIndices[i]]!=s) continue;
				ColumnBundleTransform::BasisWrapper wrapper(*targets[i].basis);
				ColumnBundleTransform(ei.qnums[s].k,e.basis[s],targets[i].qnum->k,wrapper,
					1,SpaceGroupOp(),1).scatterAxpy(1.,source,targets[i],0,1);
			}
		}
		// Each q group owns disjoint (target k, source k) blocks in this file.
		if(mpiWorld->isHead()) {
			string filename=e.dump.getFilename("phononHsub");
			fp=fopen(filename.c_str(),"r+b");
			if(!fp) die_alone("Cannot open electron-phonon output '%s'.\n",filename.c_str());
		}
	}
	void writeDisplacement(int mode,double massScale,const complexScalarFieldArray& potential)
	{
		const ElecInfo& ei=e.eInfo;
		size_t nk=mesh.indices.size(),blockSize=size_t(ei.nBands)*ei.nBands;
		std::vector<complex> payload(nk*blockSize,complex());
		std::vector<int> processed(nk,0);
		complexScalarFieldArray total;
		if(e.pertInfo.phononOptimize) {
			total.resize(potential.size());
			solver.getdVsclocTau(total,1); total=total+potential;
		}
		int contractions=0;
		for(size_t i=0;i<nk;i++) {
			int k=mesh.indices[i];
			if(reciprocity && size_t(targetIndices[i])<i) continue;
			if(!ei.isMine(k) || !targets[i]) continue;
			processed[i]=1;
			matrix derivative;
			if(e.pertInfo.phononOptimize) {
				ColumnBundle local=cache.wavefunctions[i].nData()
					? Idag_DiagV_I_cached(e.eVars.C[k],total,cache.wavefunctions[i],targets[i])
					: Idag_DiagV_I(e.eVars.C[k],total,&targets[i]);
				derivative=targets[i]^local;
				const AtomicMode& atom=e.pertInfo.datom->mode; auto species=e.iInfo.species[atom.sp];
				if(species->MnlAll.nRows()) {
					const auto& source=cache.source(atom.sp,atom.at,k);
					ColumnBundle P=species->getVatom(targets[i],atom.at), dP=-D(P,atom.dirCartesian);
					matrix sourceDerivative=zeroes(source.value.nRows(),source.value.nCols());
					for(int d=0;d<3;d++) if(atom.dirCartesian[d]) sourceDerivative+=atom.dirCartesian[d]*source.derivative[d];
					derivative+=(targets[i]^dP)*(species->MnlAll*source.value)
						+(targets[i]^P)*(species->MnlAll*sourceDerivative);
				}
			} else {
				ColumnBundle bare=targets[i].similar(),induced=targets[i].similar();
				solver.dHtau(*targets[i].qnum,bare,e.eVars.C[k],1);
				solver.dHpsi(*targets[i].qnum,induced,e.eVars.C[k],potential);
				bare+=induced; derivative=targets[i]^bare;
			}
			derivative*=massScale/nk;
			std::copy(derivative.data(),derivative.data()+blockSize,payload.begin()+i*blockSize);
			contractions++;
		}
		if(mpiWorld->nProcesses()>1) mpiWorld->reduceData(payload,MPIUtil::ReduceSum,0);
		mpiWorld->allReduceData(processed,MPIUtil::ReduceMax);
		mpiWorld->allReduce(contractions,MPIUtil::ReduceSum);
		if(fp) {
			std::vector<complex> reversed(blockSize);
			auto write=[&](size_t target,size_t source,const complex* data) {
				off_t offset=off_t(((size_t(mode)*nk+target)*nk+source)*blockSize*sizeof(complex));
				if(fseeko(fp,offset,SEEK_SET) || fwrite(data,sizeof(complex),blockSize,fp)!=blockSize)
					die_alone("Error writing electron-phonon derivatives.\n");
			};
			for(size_t i=0;i<nk;i++) {
				size_t target=targetIndices[i]; if(!processed[i]) continue;
				const complex* data=payload.data()+i*blockSize;
				write(target,i,data);
				if(reciprocity && target!=i) {
					for(int row=0;row<ei.nBands;row++) for(int col=0;col<ei.nBands;col++)
						reversed[row+ei.nBands*col]=data[col+ei.nBands*row].conj();
					write(i,target,reversed.data());
				}
			}
		}
		logPrintf("Electron-phonon export q [%g %g %g], column %d: %d/%zu contractions%s.\n",
			e.pertInfo.qvec[0],e.pertInfo.qvec[1],e.pertInfo.qvec[2],mode+1,contractions,nk,reciprocity ? " (reciprocity)" : "");
	}

	~ElectronPhononWriter()
	{ if(fp && fclose(fp)) die_alone("Error closing electron-phonon derivatives.\n"); }
};

std::vector<QSymmetry> qSymmetries(Everything& e,const vector3<int>& grid,
	const std::vector<Atom>& atoms,const std::vector<vector3<>>& qpoints)
{
	Symmetries detector;
	detector.mode=e.pertInfo.phononUseSymmetry ? SymmetriesAutomatic : SymmetriesNone;
	detector.setup(e,false); // independent object: preserve the ground-state geometry and k mesh
	std::vector<QSymmetry> result;
	const auto& operations=detector.getMatrices();
	const auto& atomMap=detector.getAtomMap();
	const vector3<int>& S=e.gInfo.S;
	// Only Fourier components which can connect wavefunctions within the
	// plane-wave cutoff enter the Hamiltonian. The unused corners of the FFT
	// box need not obey non-Cartesian rotations of a primitive lattice.
	ScalarFieldTilde filtered=J(e.eVars.Vscloc[0]);
	complex* coefficients=filtered->data();
	vector3<int> iG;
	for(int i=0;i<S[0];i++) for(int j=0;j<S[1];j++) for(int k=0;k<=S[2]/2;k++) {
		iG=vector3<int>(i,j,k);
		for(int d=0;d<3;d++) if(2*iG[d]>S[d]) iG[d]-=S[d];
		if(dot(iG,e.gInfo.GGT*iG)>8*e.cntrl.Ecut*(1+1e-12))
			coefficients[(i*S[1]+j)*(S[2]/2+1)+k]=0.;
	}
	ScalarField potentialField=I(filtered);
	const double* potential=potentialField->data();
	double potentialScale=1.;
	for(int i=0;i<e.gInfo.nr;i++) potentialScale=std::max(potentialScale,fabs(potential[i]));
	for(size_t iop=0;iop<operations.size();iop++) {
		const SpaceGroupOp& op=operations[iop];
		bool compatible=true;
		for(const vector3<>& q:qpoints)
			if(qGridIndex(q*op.inv().rot,grid)<0) {compatible=false; break;}
		if(!compatible) continue;
		for(const QuantumNumber& k:e.eInfo.qnums) {
			vector3<> rotated=k.k*op.inv().rot;
			bool found=false;
			for(const QuantumNumber& other:e.eInfo.qnums)
				if(sameReciprocalPoint(rotated,other.k) && fabs(k.weight-other.weight)<1e-12) {found=true; break;}
			if(!found) {compatible=false; break;}
		}
		if(!compatible) continue;
		// Require an exact permutation of FFT samples, including translations.
		matrix3<int> meshRotation;
		vector3<int> meshShift;
		for(int i=0;i<3;i++) {
			double shift=op.a[i]*S[i];
			meshShift[i]=int(round(shift));
			compatible &= fabs(shift-meshShift[i])<1e-8;
			for(int j=0;j<3;j++) {
				double entry=double(op.rot(i,j))*S[i]/S[j];
				meshRotation(i,j)=int(round(entry));
				compatible &= fabs(entry-meshRotation(i,j))<1e-8;
			}
		}
		if(!compatible) continue;
		// Exclude operations broken by external potentials or the electronic state.
		vector3<int> x;
		for(x[0]=0;x[0]<S[0] && compatible;x[0]++)
		for(x[1]=0;x[1]<S[1] && compatible;x[1]++)
		for(x[2]=0;x[2]<S[2];x[2]++) {
			vector3<int> y=meshRotation*x+meshShift;
			for(int d=0;d<3;d++) y[d]=(y[d]%S[d]+S[d])%S[d];
			int ix=(x[0]*S[1]+x[1])*S[2]+x[2],iy=(y[0]*S[1]+y[1])*S[2]+y[2];
			if(fabs(potential[ix]-potential[iy])>1e-6*potentialScale) {compatible=false; break;}
		}
		if(!compatible) continue;
		QSymmetry sym;
		sym.op=op; sym.cartesian=e.gInfo.R*op.rot*inv(e.gInfo.R);
		int offset=0;
		for(size_t species=0;species<e.iInfo.species.size();species++) {
			auto sp=e.iInfo.species[species];
			for(size_t a=0;a<sp->atpos.size();a++) {
				int target=offset+atomMap[species][a][iop];
				sym.atomMap.push_back(target);
				vector3<> shift=op.rot*sp->atpos[a]+op.a-atoms[target].pos;
				vector3<int> cell;
				for(int d=0;d<3;d++) {
					cell[d]=int(round(shift[d]));
					compatible &= fabs(shift[d]-cell[d])<1e-7;
				}
				sym.cellShift.push_back(cell);
			}
			offset+=sp->atpos.size();
		}
		if(compatible) result.push_back(sym);
	}
	if(result.empty()) die("No compatible DFPT q symmetry, including identity; check the ground-state potential.\n");
	logPrintf("DFPT q symmetries: %zu/%zu compatible space-group operations%s.\n",result.size(),operations.size(),
		e.pertInfo.phononUseSymmetry ? "" : " (reduction disabled)");
	return result;
}

matrix transformForceMatrix(const matrix& F,const QSymmetry& sym,const vector3<>& targetQ)
{
	return transformDFPTForceMatrix(F,sym.cartesian,targetQ,sym.atomMap,sym.cellShift);
}

struct DisplacementPlan {
	struct Mapping { matrix T; bool conjugate; };
	std::vector<Mapping> mappings;
	std::vector<int> columns;
	matrix gram;
};

DisplacementPlan displacementPlan(const std::vector<QSymmetry>& symmetries,
	const vector3<>& q,int nm,int nSigns,bool enabled)
{
	DisplacementPlan plan;
	if(enabled) for(const auto& sym:symmetries) for(int sign=0;sign<nSigns;sign++) {
		if(!sameReciprocalPoint((sign ? -q : q)*sym.op.inv().rot,q)) continue;
		matrix T(nm,nm); std::fill(T.begin(),T.end(),complex());
		for(size_t s=0;s<sym.atomMap.size();s++) {
			complex phase=cis(-2*M_PI*dot(q,sym.cellShift[s]));
			for(int a=0;a<3;a++) for(int b=0;b<3;b++)
				T.set(3*sym.atomMap[s]+a,3*s+b,phase*sym.cartesian(a,b));
		}
		plan.mappings.push_back({std::move(T),bool(sign)});
	}
	if(plan.mappings.empty()) plan.mappings.push_back({eye(nm),false});
	plan.gram=matrix(nm,nm); std::fill(plan.gram.begin(),plan.gram.end(),complex());
	int rank=0;
	for(int a=0;a<nm && rank<nm;a++) {
		matrix trial=plan.gram;
		complex* data=trial.data();
		for(const auto& mapping:plan.mappings) {
			const complex* v=mapping.T.data()+nm*a;
			for(int j=0;j<nm;j++) for(int i=0;i<nm;i++) data[i+nm*j]+=v[i]*v[j].conj();
		}
		matrix vectors; diagMatrix values;
		trial.diagonalize(vectors,values);
		int next=0;
		for(double value:values) if(value>1e-8*values.back()) next++;
		if(next>rank) {plan.columns.push_back(a); plan.gram=std::move(trial); rank=next;}
	}
	if(rank!=nm) die("DFPT displacement symmetry orbit does not span the displacement space.\n");
	return plan;
}

matrix reconstructDisplacements(const matrix& columns,const DisplacementPlan& plan)
{
	int nm=columns.nRows();
	matrix numerator(nm,nm); std::fill(numerator.begin(),numerator.end(),complex());
	complex* data=numerator.data();
	for(int a:plan.columns) for(const auto& mapping:plan.mappings) {
		const complex* T=mapping.T.data();
		const complex* source=columns.data()+nm*a;
		std::vector<complex> response(nm,complex());
		for(int i=0;i<nm;i++) for(int j=0;j<nm;j++)
			response[i]+=T[i+nm*j]*(mapping.conjugate ? source[j].conj() : source[j]);
		for(int j=0;j<nm;j++) for(int i=0;i<nm;i++)
			data[i+nm*j]+=response[i]*T[j+nm*a].conj();
	}
	// M T e_a = T M e_a (or T conj(M e_a) for antiunitary operations).
	// The symmetry images of the computed columns span the full displacement space.
	return numerator*inv(plan.gram);
}

// Reconstruct only the induced local operator. Bare local/nonlocal ionic
// derivatives are evaluated analytically at each target q and displacement.
// Contracting in the actual target eigenstates avoids electronic sewing-matrix
// assumptions, including arbitrary phases and rotations of degenerate bands.
std::vector<complexScalarFieldArray> reconstructElectronPhononPotentials(
	const GridInfo& g,const std::vector<QSymmetry>& symmetries,
	const vector3<>& sourceQ,const vector3<>& targetQ,int nSigns,int nm,
	const std::vector<int>& columns,const std::vector<complexScalarFieldArray>& sources)
{
	assert(!sources.empty() && sources.size()==columns.size());
	std::vector<complexScalarFieldArray> numerator(nm),result(nm);
	for(auto& fields:numerator) {
		fields.resize(sources[0].size());
		for(auto& field:fields) { field=complexScalarFieldData::alloc(g,isGpuEnabled()); initZero(field); }
	}
	matrix gram=zeroes(nm,nm);
	for(const auto& sym:symmetries) for(int sign=0;sign<nSigns;sign++) {
		vector3<> signedQ=sign ? -sourceQ : sourceQ;
		if(!sameReciprocalPoint(signedQ*sym.op.inv().rot,targetQ)) continue;
		// Use unwrapped real-space coordinates in the Bloch phase. This includes
		// fractional translations and reciprocal wrapping at q-grid boundaries.
		std::vector<int> indices(g.nr);
		std::vector<complex> phases(g.nr);
		vector3<int> x;
		for(x[0]=0;x[0]<g.S[0];x[0]++) for(x[1]=0;x[1]<g.S[1];x[1]++) for(x[2]=0;x[2]<g.S[2];x[2]++) {
			vector3<> position;
			for(int d=0;d<3;d++) position[d]=double(x[d])/g.S[d];
			vector3<> image=sym.op.rot*position+sym.op.a;
			vector3<int> y;
			for(int d=0;d<3;d++) { int sample=int(round(image[d]*g.S[d])); y[d]=(sample%g.S[d]+g.S[d])%g.S[d]; }
			int i=(x[0]*g.S[1]+x[1])*g.S[2]+x[2];
			indices[i]=(y[0]*g.S[1]+y[1])*g.S[2]+y[2];
			phases[i]=cis(2*M_PI*(dot(signedQ,position)-dot(targetQ,image)));
		}
		#ifdef GPU_ENABLED
		ManagedArray<int> permutation(indices.data(),indices.size());
		ManagedArray<complex> phaseArray(phases.data(),phases.size());
		const int* deviceIndices=permutation.dataGpu();
		const complex* devicePhases=phaseArray.dataGpu();
		#endif
		for(size_t ia=0;ia<columns.size();ia++) {
			int a=columns[ia],atom=a/3;
			std::vector<complex> displacement(nm,complex());
			complex phase=cis(-2*M_PI*dot(targetQ,sym.cellShift[atom]));
			for(int b=0;b<3;b++) displacement[3*sym.atomMap[atom]+b]=phase*sym.cartesian(b,a%3);
			for(int b=0;b<nm;b++) for(int c=0;c<nm;c++)
				gram.set(b,c,gram(b,c)+displacement[b]*displacement[c].conj());
			for(size_t spin=0;spin<sources[ia].size();spin++) {
				const complex* input=sources[ia][spin]->dataPref();
				for(int b=0;b<nm;b++) if(displacement[b].norm()>1e-24) {
					complex* output=numerator[b][spin]->dataPref();
					complex weight=displacement[b].conj();
					#ifdef GPU_ENABLED
					fieldSymmetryAccumulate_gpu(g.nr,deviceIndices,devicePhases,input,output,weight,bool(sign));
					#else
					for(int i=0;i<g.nr;i++) output[indices[i]]+=weight*phases[i]*(sign ? input[i].conj() : input[i]);
					#endif
				}
			}
		}
	}
	matrix vectors; diagMatrix values; gram.diagonalize(vectors,values);
	if(values.front()<=1e-8*values.back()) die("Electron-phonon symmetry images do not span target displacements.\n");
	matrix inverse=inv(gram);
	for(int b=0;b<nm;b++) {
		result[b].resize(sources[0].size());
		for(size_t spin=0;spin<result[b].size();spin++) {
			result[b][spin]=complexScalarFieldData::alloc(g,isGpuEnabled()); initZero(result[b][spin]);
			complex* output=result[b][spin]->dataPref();
			for(int c=0;c<nm;c++) {
				complex weight=inverse(c,b);
				if(weight.norm()<1e-24) continue;
				const complex* input=numerator[c][spin]->dataPref();
				callPref(eblas_zaxpy)(g.nr,weight,input,1,output,1);
			}
		}
	}
	return result;
}

// Hessian of erfc(eta*r)/r. Gaussian separation cancels against reciprocal sum.
matrix3<> screenedHessian(const vector3<>& r, double eta)
{
	double d=r.length(), a=eta*d, c=erfc(a), b=2*eta/sqrt(M_PI)*exp(-a*a);
	double tangential=-c/(d*d*d)-b/(d*d);
	double radial=3*c/(d*d*d)+3*b/(d*d)+2*eta*eta*b;
	return tangential*matrix3<>(1,1,1)+(radial/(d*d))*outer(r,r);
}

matrix ionicOffDiagonal(const matrix3<>& R, const std::vector<Atom>& atoms, const vector3<>& q)
{
	const matrix3<> G=2*M_PI*inv(R);
	double sigma=1.;
	for(int j=0;j<3;j++) sigma *= R.column(j).length()/G.row(j).length();
	sigma=pow(sigma/std::max(size_t(1),atoms.size()),1./6.);
	double eta=sqrt(0.5)/sigma, volume=fabs(det(R));
	vector3<int> Nr,Ng;
	for(int j=0;j<3;j++) {
		Nr[j]=2+ceil(10*sigma*G.row(j).length()/(2*M_PI));
		Ng[j]=2+ceil(10*R.column(j).length()/(2*M_PI*sigma)+fabs(q[j]));
	}
	int n=atoms.size();
	// The Ewald sum accumulates individual elements on the CPU. Keeping this
	// tiny matrix there avoids a CUDA transfer for every term in the sum.
	matrix F(3*n,3*n);
	std::fill(F.begin(),F.end(),complex());
	for(int s=0;s<n;s++) for(int t=0;t<n;t++) {
		vector3<> delta=atoms[t].pos-atoms[s].pos;
		// Reduce delta for stable bounds; retain the associated cell-phase gauge.
		vector3<int> shift;
		for(int j=0;j<3;j++) {shift[j]=int(floor(delta[j]+0.5)); delta[j]-=shift[j];}
		complex gauge=cis(-2*M_PI*dot(q,shift));
		vector3<int> cell;
		for(cell[0]=-Nr[0];cell[0]<=Nr[0];cell[0]++)
		for(cell[1]=-Nr[1];cell[1]<=Nr[1];cell[1]++)
		for(cell[2]=-Nr[2];cell[2]<=Nr[2];cell[2]++) {
			vector3<> r=R*(cell+delta);
			if(r.length_squared()<1e-24) continue;
			matrix3<> H=screenedHessian(r,eta);
			complex phase=-atoms[s].Z*atoms[t].Z*gauge*cis(2*M_PI*dot(q,cell));
			for(int a=0;a<3;a++) for(int b=0;b<3;b++)
				F.set(3*s+a,3*t+b,F(3*s+a,3*t+b)+phase*H(a,b));
		}
		vector3<int> ig;
		for(ig[0]=-Ng[0];ig[0]<=Ng[0];ig[0]++)
		for(ig[1]=-Ng[1];ig[1]<=Ng[1];ig[1]++)
		for(ig[2]=-Ng[2];ig[2]<=Ng[2];ig[2]++) {
			vector3<> k=ig+q, K=k*G;
			double ksq=K.length_squared();
			if(ksq<1e-24) continue; // zero macroscopic field at Gamma
			complex phase=4*M_PI/volume*atoms[s].Z*atoms[t].Z/ksq
				*exp(-0.5*sigma*sigma*ksq)*cis(-2*M_PI*dot(k,atoms[t].pos-atoms[s].pos));
			for(int a=0;a<3;a++) for(int b=0;b<3;b++)
				F.set(3*s+a,3*t+b,F(3*s+a,3*t+b)+phase*K[a]*K[b]);
		}
	}
	return F;
}

// Hessian of the 2D Ewald sum, using the same Gaussian separation and
// zero-planar-wavevector convention as EwaldSlab::energyAndGrad. Only the
// two periodic lattice directions are summed; z displacements remain free.
matrix slabIonicOffDiagonal(const matrix3<>& R, const std::vector<Atom>& atoms,
	const vector3<>& q, const CoulombParams& cp)
{
	int dir=cp.iDir, n=atoms.size();
	if(fabs(q[dir])>1e-12) die("Slab ionic response requires q in the periodic plane.\n");
	const matrix3<> G=2*M_PI*inv(R);
	double L=R.column(dir).length(), area=fabs(det(R))/L, sigma=1.;
	vector3<> normal=R.column(dir)/L;
	for(int j=0;j<3;j++) if(j!=dir) sigma*=R.column(j).length()/G.row(j).length();
	sigma=pow(5.*sigma,0.25);
	double eta=sqrt(0.5)/sigma;
	vector3<int> Nr,Ng;
	for(int j=0;j<3;j++) {
		Nr[j]=j==dir ? 0 : 2+ceil(10*sigma*G.row(j).length()/(2*M_PI));
		Ng[j]=j==dir ? 0 : 2+ceil(10*R.column(j).length()/(2*M_PI*sigma)+fabs(q[j]));
	}
	matrix F(3*n,3*n); std::fill(F.begin(),F.end(),complex());
	for(int s=0;s<n;s++) for(int t=0;t<n;t++) {
		vector3<> delta=atoms[t].pos-atoms[s].pos;
		vector3<int> shift;
		for(int j=0;j<3;j++) {shift[j]=int(floor(delta[j]+0.5)); delta[j]-=shift[j];}
		double z=L*delta[dir];
		if(fabs(z)>=0.5*L-cp.ionMargin)
			die("Slab DFPT atomic separation lies within the Coulomb ion margin; increase the vacuum spacing.\n");
		double charge=atoms[s].Z*atoms[t].Z;
		complex gauge=cis(-2*M_PI*dot(q,shift));
		vector3<int> cell;
		for(cell[0]=-Nr[0];cell[0]<=Nr[0];cell[0]++)
		for(cell[1]=-Nr[1];cell[1]<=Nr[1];cell[1]++)
		for(cell[2]=-Nr[2];cell[2]<=Nr[2];cell[2]++) {
			vector3<> r=R*(cell+delta);
			if(r.length_squared()<1e-24) continue;
			matrix3<> H=screenedHessian(r,eta);
			complex phase=-charge*gauge*cis(2*M_PI*dot(q,cell));
			for(int a=0;a<3;a++) for(int b=0;b<3;b++)
				F.set(3*s+a,3*t+b,F(3*s+a,3*t+b)+phase*H(a,b));
		}
		vector3<int> ig;
		for(ig[0]=-Ng[0];ig[0]<=Ng[0];ig[0]++)
		for(ig[1]=-Ng[1];ig[1]<=Ng[1];ig[1]++)
		for(ig[2]=-Ng[2];ig[2]<=Ng[2];ig[2]++) {
			vector3<> k=ig+q, K=k*G;
			double g=K.length(), zTerm, zPrime;
			if(g>1e-14) {
				// log(erfc) avoids overflow in exp(+/-g*z)*erfc at large vacuum.
				double plus=exp(g*z+gsl_sf_log_erfc(eta*(sigma*sigma*g+z)));
				double minus=exp(-g*z+gsl_sf_log_erfc(eta*(sigma*sigma*g-z)));
				zTerm=0.5*(plus+minus)/g;
				zPrime=0.5*(plus-minus);
			} else {
				zTerm=-z*erf(eta*z)-exp(-eta*eta*z*z)/(eta*sqrt(M_PI));
				zPrime=-erf(eta*z);
			}
			double zSecond=g*g*zTerm-2*eta/sqrt(M_PI)*exp(-0.5*sigma*sigma*g*g-eta*eta*z*z);
			complex phase=(2*M_PI/area)*charge*gauge*cis(-2*M_PI*dot(k,delta));
			for(int a=0;a<3;a++) for(int b=0;b<3;b++) {
				complex value=K[a]*K[b]*zTerm-normal[a]*normal[b]*zSecond
				+complex(0.,zPrime*(K[a]*normal[b]+normal[a]*K[b]));
				F.set(3*s+a,3*t+b,F(3*s+a,3*t+b)+phase*value);
			}
		}
	}
	return F;
}

// Explicit second ionic-potential derivative at fixed electronic state.
double bareElectronicElement(Everything& e, const AtomicMode& a, const AtomicMode& b,bool includeLocal=true)
{
	if(a.sp!=b.sp || a.at!=b.at) return 0.;
	auto sp=e.iInfo.species[a.sp];
	double result=0.;
	if(includeLocal) {
		ScalarFieldTilde V,rho,ball,core,tau;
		nullToZero(V,e.gInfo); nullToZero(rho,e.gInfo);
		sp->updateLocal(V,rho,ball,core,tau,a.at);
		// Differentiate the point-charge source before applying embedded Coulomb.
		ScalarFieldTilde second=D(D(V,a.dirCartesian),b.dirCartesian)
			+(*e.coulomb)(D(D(rho,a.dirCartesian),b.dirCartesian),Coulomb::PointChargeRight);
		result=dot(J(e.eVars.get_nTot()),O(second));
	}
	double nl=0.;
	for(int k=e.eInfo.qStart;k<e.eInfo.qStop;k++) {
		if(!e.eVars.C[k]) continue;
		const ColumnBundle& C=e.eVars.C[k];
		ColumnBundle P=sp->getVatom(C,a.at);
		if(!P.nCols()) continue;
		matrix Pc=P^C, Pa=(-D(P,a.dirCartesian))^C, Pb=(-D(P,b.dirCartesian))^C;
		matrix Pab=D(D(P,a.dirCartesian),b.dirCartesian)^C;
		const diagMatrix& f=e.eVars.F[k];
		nl+=e.eInfo.qnums[k].weight*real(trace(sp->MnlAll*(
			Pab*f*dagger(Pc)+Pc*f*dagger(Pab)+Pa*f*dagger(Pb)+Pb*f*dagger(Pa))));
	}
	mpiWorld->allReduce(nl,MPIUtil::ReduceSum);
	return result+nl;
}
}

matrix transformDFPTForceMatrix(const matrix& F,const matrix3<>& rotation,const vector3<>& targetQ,
	const std::vector<int>& atomMap,const std::vector<vector3<int>>& cellShift)
{
	int nm=F.nRows();
	// These small matrices are assembled and written on the host. Thousands
	// of tiny CUDA BLAS calls would cost more than their arithmetic.
	matrix result(nm,nm);
	const complex* source=F.data();
	complex* target=result.data();
	for(size_t s=0;s<atomMap.size();s++) for(size_t t=0;t<atomMap.size();t++) {
		complex phase=cis(-2*M_PI*dot(targetQ,cellShift[s]-cellShift[t]));
		for(int a=0;a<3;a++) for(int b=0;b<3;b++) {
			complex value=0.;
			for(int i=0;i<3;i++) for(int j=0;j<3;j++)
				value+=rotation(a,i)*source[(3*s+i)+nm*(3*t+j)]*rotation(b,j);
			target[(3*atomMap[s]+a)+nm*(3*atomMap[t]+b)]=phase*value;
		}
	}
	return result;
}

matrix ionicForceMatrix(const matrix3<>& R, const std::vector<Atom>& atoms, const vector3<>& q)
{
	matrix F=ionicOffDiagonal(R,atoms,q), F0=ionicOffDiagonal(R,atoms,vector3<>());
	for(size_t s=0;s<atoms.size();s++) for(int a=0;a<3;a++) for(int b=0;b<3;b++) {
		complex rowSum=0.;
		for(size_t t=0;t<atoms.size();t++) rowSum+=F0(3*s+a,3*t+b);
		F.set(3*s+a,3*s+b,F(3*s+a,3*s+b)-rowSum);
	}
	return F;
}

matrix ionicForceMatrix(const matrix3<>& R, const std::vector<Atom>& atoms,
	const vector3<>& q, const CoulombParams& cp)
{
	if(cp.geometry==CoulombParams::Periodic) return ionicForceMatrix(R,atoms,q);
	if(cp.geometry!=CoulombParams::Slab)
		die("DFPT ionic response requires periodic or slab Coulomb.\n");
	if(cp.embed) {
		// Match Coulomb::energyAndGrad: center each ion in the original slab
		// before converting to the doubled normal lattice vector. In-plane
		// cell coordinates remain unchanged to preserve the Bloch gauge.
		int dir=cp.iDir;
		double L=R.column(dir).length();
		matrix3<> doubled=R; doubled.set_col(dir,2.*R.column(dir));
		auto centered=atoms;
		for(Atom& atom:centered) {
			double z=atom.pos[dir]-cp.embedCenter[dir]; z-=floor(z+0.5);
			if(fabs(z)*L>=0.5*L-cp.ionMargin)
				die("Slab DFPT atom lies within the ion margin of the embedding boundary.\n");
			atom.pos[dir]=0.5*z;
		}
		CoulombParams nonEmbedded=cp; nonEmbedded.embed=false;
		return ionicForceMatrix(doubled,centered,q,nonEmbedded);
	}
	matrix F=slabIonicOffDiagonal(R,atoms,q,cp), F0=slabIonicOffDiagonal(R,atoms,vector3<>(),cp);
	for(size_t s=0;s<atoms.size();s++) for(int a=0;a<3;a++) for(int b=0;b<3;b++) {
		complex rowSum=0.;
		for(size_t t=0;t<atoms.size();t++) rowSum+=F0(3*s+a,3*t+b);
		F.set(3*s+a,3*s+b,F(3*s+a,3*s+b)-rowSum);
	}
	return F;
}

void loadSymmetryReducedDFPTState(Everything& e,bool dryRun)
{
	logPrintf("\nLoading symmetry-reduced DFPT ground state from %s.\n",e.pertInfo.phononInitialInput.c_str());
	// Parsing the source input sets this process-global tolerance. Retain its
	// value during source setup, then restore the target calculation's value.
	struct RestoreTolerance {
		double threshold=symmThreshold, squared=symmThresholdSq;
		~RestoreTolerance() {symmThreshold=threshold; symmThresholdSq=squared;}
	} restoreTolerance;
	auto sourceOwner=std::make_shared<Everything>();
	Everything& source=*sourceOwner;
	parse(readInputFile(e.pertInfo.phononInitialInput),source,false);
	if(source.pertInfo.phononGrid.isNonzero() || !source.pertInfo.phononInitialInput.empty()
		|| source.pertInfo.solverParams.nIterations || (source.cntrl.fixed_H && !e.pertInfo.phononIBZStorage) || source.vibrations
		|| source.ionicMinParams.nIterations || source.latticeMinParams.nIterations || source.ionicDynParams.nSteps)
		die("phonon-dfpt-initial-state requires a ground-state input (fixed-density NSCF is supported with IBZ storage).\n");
	if(source.cntrl.fixed_H && source.eVars.nFilenamePattern.empty())
		die("IBZ NSCF restart requires fix-electron-density, including the converged density files.\n");
	if(source.eInfo.spinType!=SpinNone || e.eInfo.spinType!=SpinNone
		|| source.cntrl.basisKdep!=BasisKpointDep || source.exCorr.exxFactor())
		die("Symmetry-reduced DFPT restart requires nonmagnetic scalar states in a k-dependent basis.\n");
	if(source.coulombParams.geometry!=e.coulombParams.geometry || source.coulombParams.embed!=e.coulombParams.embed
		|| (e.coulombParams.geometry==CoulombParams::Slab && source.coulombParams.iDir!=e.coulombParams.iDir))
		die("Symmetry-reduced DFPT restart must match the Coulomb geometry and slab direction.\n");
	if(e.coulombParams.embed && (!sameReciprocalPoint(source.coulombParams.embedCenter,e.coulombParams.embedCenter)
		|| fabs(source.coulombParams.ionMargin-e.coulombParams.ionMargin)>1e-12))
		die("Symmetry-reduced embedded DFPT restart must match the embedding center, ion margin and FFT grid.\n");
	if(nrm2(source.gInfo.R-e.gInfo.R)>1e-8 || fabs(source.cntrl.Ecut-e.cntrl.Ecut)>1e-10
		|| source.iInfo.species.size()!=e.iInfo.species.size())
		die("Symmetry-reduced DFPT restart must match the lattice, cutoff and species.\n");
	for(size_t s=0;s<e.iInfo.species.size();s++) {
		const auto& a=*e.iInfo.species[s]; const auto& b=*source.iInfo.species[s];
		if(a.potfilename!=b.potfilename || a.atpos.size()!=b.atpos.size())
			die("Symmetry-reduced DFPT restart must use identical pseudopotentials and atom ordering.\n");
		for(size_t at=0;at<a.atpos.size() && !e.pertInfo.phononIBZStorage;at++)
			if(!sameReciprocalPoint(a.atpos[at],b.atpos[at]))
				die("Symmetry-reduced DFPT restart has different atomic positions.\n");
	}
	string pattern=e.pertInfo.phononInitialState;
	auto stateFile=[&](const char* variable) {string name=pattern; name.replace(name.find("$VAR"),4,variable); return name;};
	source.eVars.wfnsFilename=stateFile("wfns");
	source.eInfo.initialFillingsFilename=stateFile("fillings");
	if(!dryRun && (!isReadable(source.eVars.wfnsFilename) || !isReadable(source.eInfo.initialFillingsFilename)))
		die("Symmetry-reduced DFPT restart requires both wavefunctions and fillings.\n");
	if(dryRun && !isReadable(source.eInfo.initialFillingsFilename)) source.eInfo.initialFillingsFilename.clear();
	source.eVars.eigsFilename=source.cntrl.fixed_H ? stateFile("eigenvals") : string();
	source.eVars.readConversion.reset();
	source.dump.clear(); source.eVars.skipWfnsInit=dryRun;
	source.setup();
	if(e.pertInfo.phononIBZStorage) e.pertInfo.checkSupportedFeatures(source,source.eInfo);
	if(e.coulombParams.embed && !(source.gInfo.S==e.gInfo.S))
		die("Symmetry-reduced embedded DFPT restart must match the FFT grid.\n");
	// Automatic symmetry setup may make a near-symmetric source geometry
	// exactly symmetric. Expansion must use the actual checkpoint geometry.
	if(nrm2(source.gInfo.R-e.gInfo.R)>1e-8)
		die("The symmetry-adjusted SCF lattice differs from DFPT; use the final SCF geometry.\n");
	for(size_t s=0;s<e.iInfo.species.size();s++) for(size_t at=0;at<e.iInfo.species[s]->atpos.size();at++)
		if(!sameReciprocalPoint(e.iInfo.species[s]->atpos[at],source.iInfo.species[s]->atpos[at]))
			die("The symmetry-adjusted SCF positions differ from DFPT; use the final SCF geometry.\n");
	if(source.eInfo.nBands!=e.eInfo.nBands
		|| source.eInfo.fillingsUpdate!=e.eInfo.fillingsUpdate
		|| fabs(source.eInfo.nElectrons-e.eInfo.nElectrons)>1e-8)
		die("Symmetry-reduced DFPT restart must match bands, fillings mode and electron count.\n");
	bool metallic=e.eInfo.fillingsUpdate==ElecInfo::FillingsHsub;
	if(metallic && (source.eInfo.smearingType!=e.eInfo.smearingType
		|| fabs(source.eInfo.smearingWidth-e.eInfo.smearingWidth)>1e-12
		|| !std::isnan(source.eInfo.mu)))
		die("Symmetry-reduced metallic DFPT restart must match smearing at fixed electron number.\n");
	bool occupied=true;
	for(int k=source.eInfo.qStart;k<source.eInfo.qStop;k++)
		for(double filling:source.eVars.F[k]) occupied &= fabs(filling-1.)<1e-12;
	mpiWorld->allReduce(occupied,MPIUtil::ReduceLAnd);
	if(!metallic && !occupied) die("Symmetry-reduced insulating DFPT restart requires fully occupied bands.\n");
	// Rebuild the map for the target's actual full mesh, then match its reduced
	// representatives to the checkpoint ordering (including their weights).
	auto reduced=source.symm.reduceKmesh(e.eInfo.qnums);
	if(reduced.size()!=source.eInfo.qnums.size())
		die("Symmetry-reduced DFPT restart has an incompatible electronic k mesh.\n");
	std::vector<int> reducedToSource(reduced.size(),-1);
	for(size_t i=0;i<reduced.size();i++) {
		for(int k=0;k<source.eInfo.nStates;k++)
			if(sameReciprocalPoint(reduced[i].k,source.eInfo.qnums[k].k)
				&& fabs(reduced[i].weight-source.eInfo.qnums[k].weight)<1e-10) {reducedToSource[i]=k; break;}
		if(reducedToSource[i]<0) die("Symmetry-reduced DFPT restart has different k points or weights.\n");
	}
	if(source.symm.mode==SymmetriesNone)
		die("Use initial-state for a full-mesh checkpoint; phonon-dfpt-initial-state expects a symmetry-enabled SCF input.\n");
	if(e.pertInfo.phononIBZStorage) for(const auto& op:source.symm.getMatrices())
		for(const auto& k:e.eInfo.qnums) {
			bool found=false;
			for(const auto& other:e.eInfo.qnums)
				if(sameReciprocalPoint(k.k*op.rot,other.k) && fabs(k.weight-other.weight)<1e-12) { found=true; break; }
			if(!found) die("IBZ storage requires the full electronic mesh to preserve the source crystal symmetries. Use a compatible k mesh or a smaller source symmetry group.\n");
		}
	logPrintf("DFPT ground-state expansion: %d reduced states -> %d full-mesh states.\n",source.eInfo.nStates,e.eInfo.nStates);
	if(dryRun) return;
	std::vector<int> owners(e.eInfo.nStates),inverts(e.eInfo.nStates),operations(e.eInfo.nStates);
	for(int k=0;k<e.eInfo.nStates;k++) {
		size_t index;
		source.symm.getKpointMap(k,index,inverts[k],operations[k]);
		owners[k]=reducedToSource[index];
	}
	if(e.pertInfo.phononIBZStorage) {
		e.pertInfo.phononGroundState=std::make_shared<DFPTGroundState>(e,std::move(sourceOwner),
			std::move(owners),std::move(inverts),std::move(operations));
		logPrintf("Deferring full-mesh ground-state reconstruction to active DFPT k batches.\n");
		return;
	}
	init(e.eVars.C,e.eInfo.nStates,e.eInfo.nBands,&e.basis[0],&e.eInfo);
	for(int s=0;s<source.eInfo.nStates;s++) {
		ColumnBundle received;
		ColumnBundle& Csource=source.eInfo.isMine(s) ? source.eVars.C[s] : received;
		diagMatrix fillings;
		if(source.eInfo.isMine(s)) {fillings=source.eVars.F[s];}
		else {Csource.init(source.eInfo.nBands,source.basis[s].nbasis,&source.basis[s],&source.eInfo.qnums[s],isGpuEnabled()); fillings.resize(source.eInfo.nBands);}
		if(mpiWorld->nProcesses()>1) {
			mpiWorld->bcastData(Csource,source.eInfo.whose(s));
			mpiWorld->bcastData(fillings,source.eInfo.whose(s));
		}
		for(int k=e.eInfo.qStart;k<e.eInfo.qStop;k++) if(owners[k]==s) {
			ColumnBundle& target=e.eVars.C[k]; target.zero();
			ColumnBundleTransform::BasisWrapper wrapper(e.basis[k]);
			ColumnBundleTransform(source.eInfo.qnums[s].k,source.basis[s],e.eInfo.qnums[k].k,wrapper,
				1,source.symm.getMatrices()[operations[k]],inverts[k]).scatterAxpy(1.,Csource,target,0,1);
			target=target*orthoMatrix(target^O(target));
			e.eVars.F[k]=fillings;
			e.iInfo.project(target,e.eVars.VdagC[k]);
		}
	}
	e.eVars.isRandom=false; e.eVars.skipWfnsInit=false;
}

std::vector<int> balanceDFPTQGroups(const std::vector<double>& costs,const std::vector<int>& capacities)
{
	if(capacities.empty()) die("DFPT q scheduling requires at least one group.\n");
	for(int capacity:capacities) if(capacity<=0) die("DFPT q group capacity must be positive.\n");
	std::vector<size_t> order(costs.size());
	for(size_t i=0;i<costs.size();i++) {
		if(!std::isfinite(costs[i]) || costs[i]<=0.) die("DFPT q cost must be finite and positive.\n");
		order[i]=i;
	}
	std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b) { return costs[a]>costs[b]; });
	std::vector<double> work(capacities.size(),0.);
	std::vector<int> groups(costs.size(),0);
	for(size_t i:order) {
		size_t best=0;
		for(size_t group=1;group<capacities.size();group++)
			if((work[group]+costs[i])/capacities[group] < (work[best]+costs[i])/capacities[best]) best=group;
		groups[i]=best;
		work[best]+=costs[i];
	}
	return groups;
}

void calculateDFPTPhonons(Everything& e,const MPIUtil* parent)
{
	PerturbationInfo& p=e.pertInfo;
	int nGroups=parent ? p.phononQGroups : 1;
	int groupIndex=parent ? mpiWorld->procDivision.iGroup : 0;
	bool outputHead=parent ? parent->isHead() : mpiWorld->isHead();
	bool batch=p.phononQBatch!=0;
	bool exportEph=p.phononElectronPhonon && !batch;
	bool occupied=true;
	for(int k=e.eInfo.qStart;k<e.eInfo.qStop;k++)
		for(double f:e.eVars.F[k]) occupied &= fabs(f-1.)<1e-12;
	mpiWorld->allReduce(occupied,MPIUtil::ReduceLAnd);
	bool metallic=e.eInfo.fillingsUpdate==ElecInfo::FillingsHsub;
	if((!metallic && !occupied) || e.eInfo.spinType!=SpinNone)
		die("phonon-dfpt requires nonmagnetic scalar bands: fully occupied or using electronic smearing.\n");
	const vector3<int> grid=p.phononGrid;
	std::unique_ptr<ElectronPhononMesh> ephMesh;
	if(p.phononElectronPhonon) ephMesh.reset(new ElectronPhononMesh(e,grid));
	int nq=grid[0]*grid[1]*grid[2];
	std::vector<Atom> atoms;
	std::vector<std::shared_ptr<AtomPerturbation>> modes;
	std::vector<vector3<>> positions;
	diagMatrix massScale;
	for(unsigned s=0;s<e.iInfo.species.size();s++) {
		auto sp=e.iInfo.species[s];
		if(sp->isUltrasoft() || sp->isRelativistic() || sp->hasCoreDensity())
			die("phonon-dfpt requires nonrelativistic norm-conserving potentials without nonlinear core corrections.\n");
		for(unsigned at=0;at<sp->atpos.size();at++) {
			atoms.emplace_back(sp->Z,sp->atpos[at]); positions.push_back(sp->atpos[at]);
			for(int a=0;a<3;a++) {
				modes.push_back(std::make_shared<AtomPerturbation>(s,at,a,e));
				massScale.push_back(1./sqrt(sp->mass*amu));
			}
		}
	}
	if(modes.empty()) die("phonon-dfpt requires at least one atom.\n");
	if(p.phononGroundState) p.phononGroundState->prepare(parent);
	else {
		elecFluidMinimize(e);
		p.groundStateReady=true;
		e.eVars.elecEnergyAndGrad(e.ener,nullptr,nullptr,true);
		if(metallic) {
			PerturbationSolver(e).computeMetallicGroundStates();
			e.eVars.setEigenvectors();
			double Bz=0.;
			double mu=e.eInfo.findMu(e.eVars.Hsub_eigs,e.eInfo.nElectrons,Bz);
			double tail=0.;
			for(int k=e.eInfo.qStart;k<e.eInfo.qStop;k++) {
				// elecEnergyAndGrad computes fillings from Haux_eigs, including
				// in SCF mode. Keep it consistent with the refined eigenbasis.
				e.eVars.Haux_eigs[k]=e.eVars.Hsub_eigs[k];
				e.eVars.F[k]=e.eInfo.smear(mu,e.eVars.Hsub_eigs[k]);
				tail=std::max(tail,e.eVars.F[k].back());
			}
			mpiWorld->allReduce(tail,MPIUtil::ReduceMax);
			if(tail>1e-6) die("Metallic DFPT needs more empty bands: highest-band occupation exceeds 1e-6. Increase elec-n-bands.\n");
			e.eVars.elecEnergyAndGrad(e.ener,nullptr,nullptr,true);
			logPrintf("Metallic DFPT: smearing density-matrix response, fixed electron number, highest-band occupation %.6le.\n",tail);
		}
		if(p.phononElectronPhonon) {
			e.eVars.setEigenvectors();
			logPrintf("Electron-phonon export: reconstructing screened potentials under q/displacement symmetries; band contractions use the full electronic mesh.\n");
		}
		// Fixed-screening equations are independent by source eigenstate. Occupied
		// insulating subspaces may otherwise retain an arbitrary unitary gauge.
		if((p.phononNestedSolver || p.phononKBatchSize) && !metallic) e.eVars.setEigenvectors();
		if(exportEph) synchronizeElectronPhononStates(e,parent);
	}

	int nm=modes.size();
	matrix bare=zeroes(nm,nm);
	int groundBatches=p.phononGroundState ? p.phononGroundState->batches() : 1;
	for(int batch=0;batch<groundBatches;batch++) {
		if(p.phononGroundState) p.phononGroundState->loadBatch(batch);
		for(int a=0;a<nm;a++) for(int b=0;b<nm;b++)
			bare.set(a,b,bare(a,b)+bareElectronicElement(e,modes[a]->mode,modes[b]->mode,batch==0));
	}
	if(p.phononGroundState) p.phononGroundState->release();
	std::vector<vector3<>> qpoints;
	for(int i=0;i<grid[0];i++) for(int j=0;j<grid[1];j++) for(int k=0;k<grid[2];k++) {
		vector3<> q(double(i)/grid[0],double(j)/grid[1],double(k)/grid[2]);
		for(int d=0;d<3;d++) if(q[d]>0.5) q[d]-=1.;
		qpoints.push_back(q);
	}
	auto symmetries=qSymmetries(e,grid,atoms,qpoints);
	bool timeReversal=p.phononUseSymmetry;
	for(const QuantumNumber& k:e.eInfo.qnums) {
		bool found=false;
		for(const QuantumNumber& other:e.eInfo.qnums)
			if(sameReciprocalPoint(-k.k,other.k) && fabs(k.weight-other.weight)<1e-12) {found=true; break;}
		if(!found) {timeReversal=false; break;}
	}
	int nSigns=timeReversal ? 2 : 1;
	if(timeReversal) logPrintf("DFPT q reduction also uses time reversal.\n");
	std::vector<int> representatives(nq,-1),counts(nq,0);
	std::vector<int> independent;
	for(int iq=0;iq<nq;iq++) {
		if(representatives[iq]>=0) continue;
		independent.push_back(iq);
		for(const auto& sym:symmetries) for(int sign=0;sign<nSigns;sign++) {
			int target=qGridIndex((sign ? -qpoints[iq] : qpoints[iq])*sym.op.inv().rot,grid);
			if(representatives[target]>=0 && representatives[target]!=iq)
				die("DFPT q symmetry operations do not form disjoint orbits; tighten the symmetry tolerance or disable q reduction.\n");
			representatives[target]=iq;
		}
	}
	std::vector<DisplacementPlan> displacementPlans;
	std::vector<int> responseOffsets(1,0);
	for(int iq:independent) {
		displacementPlans.push_back(displacementPlan(symmetries,qpoints[iq],nm,nSigns,p.phononReduceDisplacements));
		responseOffsets.push_back(responseOffsets.back()+displacementPlans.back().columns.size());
		logPrintf("DFPT displacement symmetry reduction at q %d: %zu/%d independent directions.\n",
			iq+1,displacementPlans.back().columns.size(),nm);
	}
	logPrintf("DFPT q symmetry reduction: %zu/%d representatives (%d displacement solves).\n",independent.size(),nq,responseOffsets.back());
	std::vector<int> groupCapacities(nGroups,0);
	if(mpiWorld->isHead()) groupCapacities[groupIndex]=mpiWorld->nProcesses();
	if(parent) parent->allReduceData(groupCapacities,MPIUtil::ReduceSum);
	else groupCapacities[0]=mpiWorld->nProcesses();
	// Extra ranks beyond the k-point count cannot add k parallelism.
	for(int& capacity:groupCapacities) capacity=std::min(capacity,e.eInfo.nStates);
	std::vector<double> costs(independent.size());
	for(size_t i=0;i<independent.size();i++) {
		int offMesh=0;
		const vector3<>& q=qpoints[independent[i]];
		if(nGroups>1 && p.phononBalanceQGroups)
		for(const auto& k:e.eInfo.qnums) for(int sign:{-1,1}) {
			bool onMesh=false;
			if(p.phononReuseOffsetStates) for(const auto& other:e.eInfo.qnums)
				if(sameReciprocalPoint(k.k+sign*q,other.k)) { onMesh=true; break; }
			if(!onMesh) offMesh++;
		}
		// A relative heuristic, not a timing prediction: one setup unit,
		// one unit per response, and two units for a fully off-mesh offset solve.
		costs[i]=1.+displacementPlans[i].columns.size()+double(offMesh)/e.eInfo.nStates;
		if(p.phononElectronPhonon) {
			int orbit=std::count(representatives.begin(),representatives.end(),independent[i]);
			// Export is one operator application per mode/k, while a response
			// uses many Hamiltonian applications. Keep this heuristic modest.
			costs[i]+=0.25*orbit*nm*double(ephMesh->indices.size())/e.eInfo.nStates;
		}
	}
	std::vector<int> qGroups(independent.size(),-1),selected;
	std::vector<double> selectedCosts;
	for(size_t i=0;i<independent.size();i++)
		if(!batch || int(i%p.phononQBatches)==p.phononQBatch-1) {
			selected.push_back(i); selectedCosts.push_back(costs[i]);
		}
	std::vector<int> selectedGroups(selected.size());
	if(p.phononBalanceQGroups) selectedGroups=balanceDFPTQGroups(selectedCosts,groupCapacities);
	else for(size_t i=0;i<selected.size();i++) selectedGroups[i]=i%nGroups;
	for(size_t i=0;i<selected.size();i++) qGroups[selected[i]]=selectedGroups[i];
	if(nGroups>1) {
		for(int group=0;group<nGroups;group++) {
			int representatives=0,solves=0; double work=0.;
			for(size_t i=0;i<independent.size();i++) if(qGroups[i]==group) {
				representatives++; solves+=displacementPlans[i].columns.size(); work+=costs[i];
			}
			logPrintf("DFPT q group %d/%d assignment: %d representatives, %d responses, estimated work %.3f / capacity %d.\n",
				group+1,nGroups,representatives,solves,work,groupCapacities[group]);
		}
	}
	// Plans form part of the on-disk compatibility check as well as the MPI check.
	std::vector<int> signature{nSigns,int(symmetries.size()),int(independent.size()),nm};
	signature.insert(signature.end(),independent.begin(),independent.end());
	signature.insert(signature.end(),responseOffsets.begin(),responseOffsets.end());
	for(const auto& plan:displacementPlans) signature.insert(signature.end(),plan.columns.begin(),plan.columns.end());
	for(const auto& sym:symmetries) {
		for(int i=0;i<3;i++) for(int j=0;j<3;j++) signature.push_back(sym.op.rot(i,j));
		signature.insert(signature.end(),sym.atomMap.begin(),sym.atomMap.end());
		for(const auto& shift:sym.cellShift) for(int d=0;d<3;d++) signature.push_back(shift[d]);
	}
	DFPTCheckpoint checkpoint(e,parent,signature);
	if(parent) {
		// Plans must agree before independent work: otherwise even the length
		// of the residual gather could differ between q communicators.
		signature.insert(signature.end(),qGroups.begin(),qGroups.end());
		int length=signature.size(); parent->bcast(length);
		std::vector<int> reference(length);
		if(parent->isHead()) reference=signature;
		parent->bcastData(reference);
		bool consistent=reference==signature;
		parent->allReduce(consistent,MPIUtil::ReduceLAnd);
		if(!consistent) die("DFPT symmetry plans differ between q groups; converge the ground state more tightly or disable q symmetry.\n");
	}
	if(p.phononMerge) checkpoint.requireComplete(independent,parent);
	if(exportEph) {
		// Write the actual eigenstates used in every matrix element, rather
		// than relying on the gauge of the input SCF checkpoint.
		if(groupIndex==0) {
			if(p.phononGroundState) p.phononGroundState->writeWavefunctions(e.dump.getFilename("wfns").c_str());
			else e.eInfo.write(e.eVars.C,e.dump.getFilename("wfns").c_str());
			e.eInfo.write(e.eVars.Hsub_eigs,e.dump.getFilename("eigenvals").c_str());
			std::vector<diagMatrix> externalFillings=e.eVars.F;
			for(int k=e.eInfo.qStart;k<e.eInfo.qStop;k++) externalFillings[k]*=2.;
			e.eInfo.write(externalFillings,e.dump.getFilename("fillings").c_str());
		}
		if(outputHead) {
			string filename=e.dump.getFilename("phononHsub");
			FILE* fp=fopen(filename.c_str(),"wb");
			if(!fp || fclose(fp)) die_alone("Cannot create electron-phonon output '%s'.\n",filename.c_str());
			filename=e.dump.getFilename("phononKpts");
			fp=fopen(filename.c_str(),"w");
			if(!fp) die_alone("Cannot write commensurate electron-phonon k points.\n");
			fprintf(fp,"#K-pts commensure with phonon supercell:\n");
			for(const auto& k:ephMesh->kpoints) fprintf(fp,"%+.7f %+.7f %+.7f\n",k[0],k[1],k[2]);
			if(fclose(fp)) die_alone("Error closing electron-phonon k points.\n");
			logPrintf("Exporting native phononHsub: %d commensurate k points, %zu Cartesian modes, %d bands.\n",nq,modes.size(),e.eInfo.nBands);
		}
		bool ready=true; (parent ? parent : mpiWorld)->bcast(ready);
	}

	std::vector<matrix> Fq(nq);
	for(int iq=0;iq<nq;iq++) {
		Fq[iq]=matrix(nm,nm);
		std::fill(Fq[iq].begin(),Fq[iq].end(),complex());
	}
	std::vector<double> residuals(responseOffsets.back(),0.);
	for(size_t iRepresentative=0;iRepresentative<independent.size();iRepresentative++) {
		if(qGroups[iRepresentative]!=groupIndex) continue;
		if(batch && int(iRepresentative%p.phononQBatches)!=p.phononQBatch-1) continue;
		int iq=independent[iRepresentative];
		vector3<> q=qpoints[iq];
		p.qvec=q; p.commensurate=false;
		std::vector<complexScalarFieldArray> inducedPotentials;
		const auto& plan=displacementPlans[iRepresentative];
		matrix F(nm,nm);
		std::vector<double> qResiduals;
		bool loaded=checkpoint.load(iq,plan.columns,F,qResiduals,inducedPotentials);
		if(!loaded && p.phononMerge) die("Missing DFPT checkpoint for representative q %d; complete all batches before merging.\n",iq+1);
		if(!loaded) {
			if(!p.phononKBatchSize) p.setup(e,e.eVars);
			PerturbationSolver solver(e);
			std::vector<SpaceGroupOp> responseOperations;
			for(const auto& sym:symmetries) responseOperations.push_back(sym.op);
			solver.setResponseSymmetries(responseOperations);
			F=bare+ionicForceMatrix(e.gInfo.R,atoms,q,e.coulombParams);
			qResiduals.resize(plan.columns.size());
			for(size_t ia=0;ia<plan.columns.size();ia++) {
				int a=plan.columns[ia];
				logPrintf("DFPT phonon q %d/%d (representative %zu/%zu, group %d/%d), displacement %zu/%zu (Cartesian column %d/%d)\n",iq+1,nq,iRepresentative+1,independent.size(),groupIndex+1,nGroups,ia+1,plan.columns.size(),a+1,nm);
				p.datom=modes[a];
				std::vector<complex> streamedForces;
				if(p.phononKBatchSize) streamedForces=solver.solvePhononKBatch(modes);
				else solver.solvePerturbation();
				if(p.phononElectronPhonon) {
					inducedPotentials.push_back(clone(p.dVsclocpq));
					// Keep completed columns in host memory while subsequent GPU
					// responses run. Export/checkpoint I/O can move them back as needed.
					if(!p.phononCheckpoint.empty())
						for(auto& field:inducedPotentials.back()) field->data();
				}
				qResiduals[ia]=p.lastResponseResidual;
				// Freeze this response: calcdGradTau below does not change dC.
				for(int b=0;b<nm;b++) {
					p.datom=modes[b];
					complex response=p.phononKBatchSize ? streamedForces[b] : solver.forceResponse();
					F.set(b,a,F(b,a)+response);
				}
			}
			solver.reportPhononCache();
			F=reconstructDisplacements(F,plan);
			double herm=nrm2(F-dagger(F))/std::max(1e-30,nrm2(F));
			logPrintf("DFPT force matrix Hermiticity relative error: %.6le\n",herm);
			if(!std::isfinite(herm) || herm>1e-3)
				die("DFPT force matrix is not Hermitian; converge the ground and offset states more tightly.\n");
			F=dagger_symmetrize(F);
			checkpoint.save(iq,plan.columns,F,qResiduals,inducedPotentials);
			// Export/next-q setup can allocate a new set of states. Completed
			// checkpoints no longer need these response vectors on the GPU.
			if(exportEph || !p.phononCheckpoint.empty()) solver.releasePhononResponse();
		}
		std::copy(qResiduals.begin(),qResiduals.end(),residuals.begin()+responseOffsets[iRepresentative]);
		if(batch) continue;
		matrix reversed(nm,nm);
		const complex* original=F.data();
		complex* conjugated=reversed.data();
		for(int a=0;a<nm*nm;a++) conjugated[a]=original[a].conj();
		// Average all equivalent mappings, including little-group operations.
		// The cell shifts supply the phases for nonsymmorphic atom permutations.
		for(const auto& sym:symmetries) for(int sign=0;sign<nSigns;sign++) {
			int target=qGridIndex((sign ? -q : q)*sym.op.inv().rot,grid);
			matrix mapped=transformForceMatrix(sign ? reversed : F,sym,qpoints[target]);
			complex* sum=Fq[target].data();
			const complex* add=mapped.data();
			for(int i=0;i<nm*nm;i++) sum[i]+=add[i];
			counts[target]++;
		}
		if(exportEph) {
			ManagedMemoryBase::reportAllocationCache("response released");
			std::unique_ptr<ElectronPhononExportCache> retainedExportCache;
			if(!p.phononGroundState) retainedExportCache.reset(new ElectronPhononExportCache(e,*ephMesh));
			// The complete orbit belongs to this q group, so output blocks remain
			// disjoint even though only the representative response was solved.
			for(int target=0;target<nq;target++) if(representatives[target]==iq) {
				auto potentials=reconstructElectronPhononPotentials(e.gInfo,symmetries,q,qpoints[target],
					nSigns,nm,plan.columns,inducedPotentials);
				p.qvec=qpoints[target];
				PerturbationSolver exportSolver(e);
				int partner=qGridIndex(-qpoints[target],grid);
				bool reciprocity=p.phononOptimize && p.phononEphReciprocity && representatives[partner]==iq;
				for(int batch=0;batch<groundBatches;batch++) {
					if(p.phononGroundState) p.phononGroundState->loadBatch(batch);
					std::unique_ptr<ElectronPhononExportCache> batchCache;
					if(p.phononGroundState) batchCache.reset(new ElectronPhononExportCache(e,*ephMesh));
					ElectronPhononWriter writer(e,exportSolver,*ephMesh,batchCache ? *batchCache : *retainedExportCache,reciprocity);
					for(int a=0;a<nm;a++) {
						p.datom=modes[a];
						writer.writeDisplacement(a,massScale[a],potentials[a]);
					}
				}
				if(p.phononGroundState) p.phononGroundState->release();
			}
			p.qvec=q;
		}
	}
	if(batch) {
		bool finished=true; (parent ? parent : mpiWorld)->bcast(finished);
		logPrintf("DFPT q batch %d/%d complete; checkpoints saved. Run phonon-dfpt-merge yes after all batches finish.\n",p.phononQBatch,p.phononQBatches);
		p.datom.reset();
		return;
	}
	if(parent) {
		// Only each group head contributes: group-local reductions have already
		// produced identical matrices on all ranks in that group.
		std::vector<complex> merged(size_t(nq)*nm*nm,complex());
		if(mpiWorld->isHead()) {
			for(int iq=0;iq<nq;iq++) {
				const complex* data=Fq[iq].data();
				std::copy(data,data+nm*nm,merged.begin()+size_t(iq)*nm*nm);
			}
		} else {
			std::fill(counts.begin(),counts.end(),0);
			std::fill(residuals.begin(),residuals.end(),0.);
		}
		parent->allReduceData(merged,MPIUtil::ReduceSum);
		parent->allReduceData(counts,MPIUtil::ReduceSum);
		parent->allReduceData(residuals,MPIUtil::ReduceSum);
		for(int iq=0;iq<nq;iq++)
			std::copy(merged.begin()+size_t(iq)*nm*nm,merged.begin()+size_t(iq+1)*nm*nm,Fq[iq].data());
		double maximum=0.;
		for(double residual:residuals) {
			if(!std::isfinite(residual)) die("Nonfinite DFPT response residual collected from q groups.\n");
			maximum=std::max(maximum,residual);
		}
		logPrintf("DFPT q groups completed: %zu response solves; maximum relative residual: %.6le\n",residuals.size(),maximum);
	}
	for(int count:counts) if(count<=0) die("Missing DFPT force matrix after q-group gather.\n");
	for(int iq=0;iq<nq;iq++) for(complex& value:Fq[iq]) value*=1./counts[iq];
	// Reciprocal reality, including self-inverse boundary q points.
	for(int iq=0;iq<nq;iq++) {
		int i=iq/(grid[1]*grid[2]),j=(iq/grid[2])%grid[1],k=iq%grid[2];
		int partner=((grid[0]-i)%grid[0]*grid[1]+(grid[1]-j)%grid[1])*grid[2]+(grid[2]-k)%grid[2];
		if(partner<iq) continue;
		complex* positive=Fq[iq].data();
		complex* negative=Fq[partner].data();
		for(int a=0;a<nm*nm;a++) {
			complex mean=0.5*(positive[a]+negative[a].conj());
			positive[a]=mean; negative[a]=mean.conj();
		}
	}
	// Orthogonal projection of the Gamma force matrix onto translations' complement.
	matrix translation=zeroes(nm,3);
	for(int a=0;a<nm;a++) translation.set(a,a%3,1.);
	matrix P=(1./atoms.size())*(translation*dagger(translation));
	matrix oldGamma=Fq[0];
	Fq[0]+=(P*Fq[0]*P-Fq[0]*P-P*Fq[0]);
	logPrintf("DFPT acoustic sum rule correction: %.6le\n",nrm2(Fq[0]-oldGamma));
	auto cellMap=getCellMap(e.gInfo.R,e.gInfo.R*Diag(vector3<>(grid)),
		vector3<bool>(false,false,false),positions,positions,1.,outputHead ? e.dump.getFilename("phononCellMap") : string());
	std::vector<matrix> realForce;
	for(const auto& entry:cellMap) {
		matrix F(nm,nm);
		std::fill(F.begin(),F.end(),complex());
		complex* data=F.data();
		for(int iq=0;iq<nq;iq++) {
			complex phase=(1./nq)*cis(-2*M_PI*dot(qpoints[iq],entry.first));
			const complex* source=Fq[iq].data();
			for(int a=0;a<nm*nm;a++) data[a]+=phase*source[a];
		}
		const complex* weights=entry.second.data();
		for(int a=0;a<nm;a++) for(int b=0;b<nm;b++)
			data[a+nm*b]=data[a+nm*b].real()*weights[(a/3)+(nm/3)*(b/3)];
		realForce.push_back(F);
	}
	if(outputHead) {
		auto open=[&](const string& name,const char* mode) {
			string filename=e.dump.getFilename(name); FILE* fp=fopen(filename.c_str(),mode);
			if(!fp) die_alone("Cannot write DFPT output '%s'.\n",filename.c_str());
			return fp;
		};
		FILE* fp=open("phononForceMatrixQ","w");
		fprintf(fp,"# nq nmodes; each q followed by row-major force matrix (Re Im), Eh/bohr^2\n%d %d\n",nq,nm);
		for(int iq=0;iq<nq;iq++) {
			fprintf(fp,"%.17g %.17g %.17g\n",qpoints[iq][0],qpoints[iq][1],qpoints[iq][2]);
			for(int a=0;a<nm;a++) for(int b=0;b<nm;b++) fprintf(fp,"%.17g %.17g\n",Fq[iq](a,b).real(),Fq[iq](a,b).imag());
		}
		fclose(fp);
		if(p.phononElectronPhonon) {
			off_t expected=off_t(size_t(nm)*nq*nq*e.eInfo.nBands*e.eInfo.nBands*sizeof(complex));
			if(fileSize(e.dump.getFilename("phononHsub").c_str())!=expected)
				die_alone("Incomplete native electron-phonon output.\n");
		}
		fp=open("phononOmegaSq","wb");
		for(matrix& F:realForce) {
			complex* data=F.data();
			for(int a=0;a<nm;a++) for(int b=0;b<nm;b++) data[a+nm*b]*=massScale[a]*massScale[b];
			F.write_real(fp);
		}
		fclose(fp);
		fp=open("phononCellWeights","wb");
		for(const auto& entry:cellMap) entry.second.write_real(fp);
		fclose(fp);
		fp=open("phononBasis","w");
		fprintf(fp,"#species atom dx[bohrs] dy[bohrs] dz[bohrs] M[amu]\n");
		for(int a=0;a<nm;a++) {
			const AtomicMode& m=modes[a]->mode;
			vector3<> dir=massScale[a]*m.dirCartesian;
			fprintf(fp,"%s %u %.17g %.17g %.17g %.17g\n",e.iInfo.species[m.sp]->name.c_str(),m.at,
				dir[0],dir[1],dir[2],e.iInfo.species[m.sp]->mass);
		}
		fclose(fp);
	}
	p.datom.reset();
}
