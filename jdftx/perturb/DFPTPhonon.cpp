/* Primitive-cell DFPT phonons and validation. GPL v3 or later; part of JDFTx. */
#include <perturb/DFPTPhonon.h>
#include <perturb/PerturbationSolver.h>
#include <electronic/Everything.h>
#include <electronic/ElecMinimizer.h>
#include <electronic/ColumnBundleTransform.h>
#include <commands/parser.h>
#include <commands/command.h>
#include <core/LatticeUtils.h>
#include <core/Units.h>

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

// Keep the Cartesian derivatives free of occupation factors, k weights and
// mass factors. In particular, metallic dC is a variational density-matrix
// amplitude, not the Hamiltonian derivative needed for electron scattering.
class ElectronPhononWriter {
	Everything& e;
	PerturbationSolver& solver;
	std::vector<ColumnBundle> targets;
	std::vector<diagMatrix> targetEnergies;
	std::vector<int> targetIndices;
	FILE* fp=nullptr;
public:
	ElectronPhononWriter(Everything& e,PerturbationSolver& solver,int iq,int nm)
	: e(e),solver(solver),targets(e.eInfo.nStates),targetEnergies(e.eInfo.nStates),
	  targetIndices(e.eInfo.nStates,-1)
	{
		const ElecInfo& ei=e.eInfo;
		const PerturbationInfo& p=e.pertInfo;
		for(int k=0;k<ei.nStates;k++) for(int s=0;s<ei.nStates;s++)
			if(sameReciprocalPoint(p.kplusq_vectors[k].k,ei.qnums[s].k)) {targetIndices[k]=s; break;}
		// On-mesh targets use the original ground-state eigenvector gauge,
		// including reciprocal wrapping. Off-mesh targets use offset eigenstates.
		for(int s=0;s<ei.nStates;s++) {
			if(std::find(targetIndices.begin(),targetIndices.end(),s)==targetIndices.end()) continue;
			ColumnBundle received;
			ColumnBundle& source=ei.isMine(s) ? e.eVars.C[s] : received;
			if(!ei.isMine(s)) source.init(ei.nBands,e.basis[s].nbasis,&e.basis[s],&ei.qnums[s],isGpuEnabled());
			diagMatrix energies=ei.isMine(s) ? e.eVars.Hsub_eigs[s] : diagMatrix(ei.nBands);
			if(mpiWorld->nProcesses()>1) {
				mpiWorld->bcastData(source,ei.whose(s));
				mpiWorld->bcastData(energies,ei.whose(s));
			}
			for(int k=ei.qStart;k<ei.qStop;k++) if(targetIndices[k]==s) {
				targets[k]=p.Cinc[k].similar(); targets[k].zero();
				ColumnBundleTransform::BasisWrapper wrapper(*targets[k].basis);
				ColumnBundleTransform(ei.qnums[s].k,e.basis[s],targets[k].qnum->k,wrapper,
					1,SpaceGroupOp(),1).scatterAxpy(1.,source,targets[k],0,1);
				targetEnergies[k]=energies;
			}
		}
		for(int k=ei.qStart;k<ei.qStop;k++) if(targetIndices[k]<0) {
			targets[k]=p.Cinc[k];
			ColumnBundle HC;
			solver.applyH(*targets[k].qnum,eye(ei.nBands),HC,targets[k]);
			matrix U;
			dagger_symmetrize(targets[k]^HC).diagonalize(U,targetEnergies[k]);
			targets[k]=targets[k]*U;
		}
		char suffix[64]; snprintf(suffix,sizeof(suffix),"phononEphQ%06d",iq);
		string filename=e.dump.getFilename(suffix);
		FILE* meta=nullptr;
		if(mpiWorld->isHead()) {
			fp=fopen(filename.c_str(),"wb");
			meta=fopen((filename+".meta").c_str(),"w");
			if(!fp || !meta) die_alone("Cannot write electron-phonon output '%s'.\n",filename.c_str());
			fprintf(meta,"# DFPT electron-phonon version nk nb nm; complex128 column-major matrices, mode then k; Eh/bohr\n1 %d %d %d\n",ei.nStates,ei.nBands,nm);
			fprintf(meta,"%.17g %.17g %.17g\n",p.qvec[0],p.qvec[1],p.qvec[2]);
			fprintf(meta,"# per k: k[3] weight target-index k+q[3], then source and target energies [Eh]\n");
		}
		for(int k=0;k<ei.nStates;k++) {
			diagMatrix source=ei.isMine(k) ? e.eVars.Hsub_eigs[k] : diagMatrix(ei.nBands);
			diagMatrix target=ei.isMine(k) ? targetEnergies[k] : diagMatrix(ei.nBands);
			if(mpiWorld->nProcesses()>1) {
				mpiWorld->bcastData(source,ei.whose(k));
				mpiWorld->bcastData(target,ei.whose(k));
			}
			if(meta) {
				const vector3<>& ks=ei.qnums[k].k;
				const vector3<>& kt=p.kplusq_vectors[k].k;
				fprintf(meta,"%.17g %.17g %.17g %.17g %d %.17g %.17g %.17g\n",
					ks[0],ks[1],ks[2],ei.qnums[k].weight,targetIndices[k],kt[0],kt[1],kt[2]);
				for(double en:source) fprintf(meta,"%.17g ",en);
				fprintf(meta,"\n");
				for(double en:target) fprintf(meta,"%.17g ",en);
				fprintf(meta,"\n");
			}
		}
		if(meta) {
			if(fclose(meta)) die_alone("Error closing electron-phonon metadata.\n");
			logPrintf("Exporting screened electron-phonon derivatives to '%s'.\n",filename.c_str());
		}
	}
	void writeDisplacement()
	{
		const ElecInfo& ei=e.eInfo;
		const PerturbationInfo& p=e.pertInfo;
		std::vector<matrix> matrices(ei.nStates);
		for(int k=ei.qStart;k<ei.qStop;k++) {
			ColumnBundle bare=targets[k].similar(),induced=targets[k].similar();
			solver.dHtau(*targets[k].qnum,bare,e.eVars.C[k],1);
			solver.dHpsi(*targets[k].qnum,induced,e.eVars.C[k],p.dVsclocpq);
			matrices[k]=targets[k]^(bare+induced);
		}
		for(int k=0;k<ei.nStates;k++) {
			if(!ei.isMine(k)) matrices[k].init(ei.nBands,ei.nBands);
			if(mpiWorld->nProcesses()>1) mpiWorld->bcastData(matrices[k],ei.whose(k));
			if(fp) {
				matrices[k].write(fp);
				if(ferror(fp)) die_alone("Error writing electron-phonon derivatives.\n");
			}
		}
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

// Explicit second ionic-potential derivative at fixed electronic state.
double bareElectronicElement(Everything& e, const AtomicMode& a, const AtomicMode& b)
{
	if(a.sp!=b.sp || a.at!=b.at) return 0.;
	auto sp=e.iInfo.species[a.sp];
	ScalarFieldTilde V,rho,ball,core,tau;
	nullToZero(V,e.gInfo); nullToZero(rho,e.gInfo);
	sp->updateLocal(V,rho,ball,core,tau,a.at);
	V+=(*e.coulomb)(rho,Coulomb::PointChargeRight);
	double result=dot(J(e.eVars.get_nTot()),O(D(D(V,a.dirCartesian),b.dirCartesian)));
	double nl=0.;
	for(int k=e.eInfo.qStart;k<e.eInfo.qStop;k++) {
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

void loadSymmetryReducedDFPTState(Everything& e,bool dryRun)
{
	logPrintf("\nLoading symmetry-reduced DFPT ground state from %s.\n",e.pertInfo.phononInitialInput.c_str());
	// Parsing the source input sets this process-global tolerance. Retain its
	// value during source setup, then restore the target calculation's value.
	struct RestoreTolerance {
		double threshold=symmThreshold, squared=symmThresholdSq;
		~RestoreTolerance() {symmThreshold=threshold; symmThresholdSq=squared;}
	} restoreTolerance;
	Everything source;
	parse(readInputFile(e.pertInfo.phononInitialInput),source,false);
	if(source.pertInfo.phononGrid.isNonzero() || !source.pertInfo.phononInitialInput.empty()
		|| source.pertInfo.solverParams.nIterations || source.cntrl.fixed_H || source.vibrations
		|| source.ionicMinParams.nIterations || source.latticeMinParams.nIterations || source.ionicDynParams.nSteps)
		die("phonon-dfpt-initial-state requires a ground-state SCF input.\n");
	if(source.eInfo.spinType!=SpinNone || e.eInfo.spinType!=SpinNone
		|| source.cntrl.basisKdep!=BasisKpointDep || source.exCorr.exxFactor())
		die("Symmetry-reduced DFPT restart requires nonmagnetic scalar states in a k-dependent basis.\n");
	if(nrm2(source.gInfo.R-e.gInfo.R)>1e-8 || fabs(source.cntrl.Ecut-e.cntrl.Ecut)>1e-10
		|| source.iInfo.species.size()!=e.iInfo.species.size())
		die("Symmetry-reduced DFPT restart must match the lattice, cutoff and species.\n");
	for(size_t s=0;s<e.iInfo.species.size();s++) {
		const auto& a=*e.iInfo.species[s]; const auto& b=*source.iInfo.species[s];
		if(a.potfilename!=b.potfilename || a.atpos.size()!=b.atpos.size())
			die("Symmetry-reduced DFPT restart must use identical pseudopotentials and atom ordering.\n");
		for(size_t at=0;at<a.atpos.size();at++)
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
	source.eVars.eigsFilename.clear(); source.eVars.readConversion.reset();
	source.dump.clear(); source.eVars.skipWfnsInit=dryRun;
	source.setup();
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
	logPrintf("DFPT ground-state expansion: %d reduced states -> %d full-mesh states.\n",source.eInfo.nStates,e.eInfo.nStates);
	if(dryRun) return;
	std::vector<int> owners(e.eInfo.nStates),inverts(e.eInfo.nStates),operations(e.eInfo.nStates);
	for(int k=0;k<e.eInfo.nStates;k++) {
		size_t index;
		source.symm.getKpointMap(k,index,inverts[k],operations[k]);
		owners[k]=reducedToSource[index];
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

void calculateDFPTPhonons(Everything& e,const MPIUtil* parent)
{
	PerturbationInfo& p=e.pertInfo;
	int nGroups=parent ? p.phononQGroups : 1;
	int groupIndex=parent ? mpiWorld->procDivision.iGroup : 0;
	bool outputHead=parent ? parent->isHead() : mpiWorld->isHead();
	bool occupied=true;
	for(int k=e.eInfo.qStart;k<e.eInfo.qStop;k++)
		for(double f:e.eVars.F[k]) occupied &= fabs(f-1.)<1e-12;
	mpiWorld->allReduce(occupied,MPIUtil::ReduceLAnd);
	bool metallic=e.eInfo.fillingsUpdate==ElecInfo::FillingsHsub;
	if((!metallic && !occupied) || e.eInfo.spinType!=SpinNone)
		die("phonon-dfpt requires nonmagnetic scalar bands: fully occupied or using electronic smearing.\n");
	const vector3<int> grid=p.phononGrid;
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
		// Force-matrix symmetry transformations do not supply electronic sewing
		// matrices. Explicit responses preserve the actual screened operator.
		p.phononUseSymmetry=false;
		p.phononReduceDisplacements=false;
		logPrintf("Electron-phonon export: explicitly solving all q points and Cartesian displacements; offset reuse and MPI q groups retained.\n");
	}
	int nm=modes.size();
	matrix bare=zeroes(nm,nm);
	for(int a=0;a<nm;a++) for(int b=0;b<nm;b++)
		bare.set(a,b,bareElectronicElement(e,modes[a]->mode,modes[b]->mode));
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
	if(parent) {
		// Plans must agree before independent work: otherwise even the length
		// of the residual gather could differ between q communicators.
		std::vector<int> signature{nSigns,int(symmetries.size()),int(independent.size()),nm};
		signature.insert(signature.end(),independent.begin(),independent.end());
		signature.insert(signature.end(),responseOffsets.begin(),responseOffsets.end());
		for(const auto& plan:displacementPlans) signature.insert(signature.end(),plan.columns.begin(),plan.columns.end());
		for(const auto& sym:symmetries) {
			for(int i=0;i<3;i++) for(int j=0;j<3;j++) signature.push_back(sym.op.rot(i,j));
			signature.insert(signature.end(),sym.atomMap.begin(),sym.atomMap.end());
			for(const auto& shift:sym.cellShift) for(int d=0;d<3;d++) signature.push_back(shift[d]);
		}
		int length=signature.size(); parent->bcast(length);
		std::vector<int> reference(length);
		if(parent->isHead()) reference=signature;
		parent->bcastData(reference);
		bool consistent=reference==signature;
		parent->allReduce(consistent,MPIUtil::ReduceLAnd);
		if(!consistent) die("DFPT symmetry plans differ between q groups; converge the ground state more tightly or disable q symmetry.\n");
	}
	std::vector<matrix> Fq(nq);
	for(int iq=0;iq<nq;iq++) {
		Fq[iq]=matrix(nm,nm);
		std::fill(Fq[iq].begin(),Fq[iq].end(),complex());
	}
	std::vector<double> residuals(responseOffsets.back(),0.);
	for(size_t iRepresentative=0;iRepresentative<independent.size();iRepresentative++) {
		if(int(iRepresentative%nGroups)!=groupIndex) continue;
		int iq=independent[iRepresentative];
		vector3<> q=qpoints[iq];
		p.qvec=q; p.commensurate=false;
		p.setup(e,e.eVars);
		PerturbationSolver solver(e);
		std::unique_ptr<ElectronPhononWriter> eph;
		matrix F=bare+ionicForceMatrix(e.gInfo.R,atoms,q);
		const auto& plan=displacementPlans[iRepresentative];
		for(size_t ia=0;ia<plan.columns.size();ia++) {
			int a=plan.columns[ia];
			logPrintf("DFPT phonon q %d/%d (representative %zu/%zu, group %d/%d), displacement %zu/%zu (Cartesian column %d/%d)\n",iq+1,nq,iRepresentative+1,independent.size(),groupIndex+1,nGroups,ia+1,plan.columns.size(),a+1,nm);
			p.datom=modes[a];
			solver.solvePerturbation();
			if(p.phononElectronPhonon) {
				if(!eph) eph.reset(new ElectronPhononWriter(e,solver,iq,nm));
				eph->writeDisplacement(); // before calcdGradTau changes the displaced atom
			}
			residuals[responseOffsets[iRepresentative]+ia]=p.lastResponseResidual;
			// Freeze this response: calcdGradTau below does not change dC.
			for(int b=0;b<nm;b++) {
				p.datom=modes[b]; solver.calcdGradTau();
				complex response=0.;
				for(int state=e.eInfo.qStart;state<e.eInfo.qStop;state++) {
					response+=dotc(p.dGradTau.X[state],p.dC[state]);
					response+=conj(dotc(p.dGradTau.X[state+e.eInfo.nStates],p.dC[state+e.eInfo.nStates]));
				}
				mpiWorld->allReduce(response,MPIUtil::ReduceSum);
				F.set(b,a,F(b,a)+response);
			}
		}
		F=reconstructDisplacements(F,plan);
		double herm=nrm2(F-dagger(F))/std::max(1e-30,nrm2(F));
		logPrintf("DFPT force matrix Hermiticity relative error: %.6le\n",herm);
		if(!std::isfinite(herm) || herm>1e-3)
			die("DFPT force matrix is not Hermitian; converge the ground and offset states more tightly.\n");
		F=dagger_symmetrize(F);
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
			fp=open("phononEph","w");
			fprintf(fp,"# DFPT electron-phonon version nq nk nb nm\n1 %d %d %d %d\n",nq,e.eInfo.nStates,e.eInfo.nBands,nm);
			for(const auto& q:qpoints) fprintf(fp,"%.17g %.17g %.17g\n",q[0],q[1],q[2]);
			fclose(fp);
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
