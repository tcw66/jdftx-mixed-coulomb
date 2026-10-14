#include <electronic/BerryPhase.h>
#include <electronic/Everything.h>
#include <electronic/SpeciesInfo.h>
#include <core/BlasExtra.h>
#include <core/GpuUtil.h>
#include <map>
#include <array>
#include <algorithm>
#include <cmath>

extern "C" {
void zgesvd_(char*,char*,int*,int*,complex*,int*,double*,complex*,int*,complex*,int*,complex*,int*,double*,int*);
void zgetrf_(int*,int*,complex*,int*,int*,int*);
}

namespace {
int meshIndex(const vector3<int>& x, const vector3<int>& n)
{ return (x[0]*n[1]+x[1])*n[2]+x[2]; }
std::array<int,3> key(const vector3<int>& x)
{ return {{x[0],x[1],x[2]}}; }
matrix occupiedOverlap(const ColumnBundle& left, const ColumnBundle& right, double volume)
{
	// Mapping and MPI replication already put these coefficients on the CPU.
	// For small occupied manifolds, avoid uploading each neighbor pair for a
	// tiny GPU matrix product followed immediately by a CPU SVD.
	if(left.nCols()>32) return volume*(left^right);
	matrix result(left.nCols(),right.nCols());
	eblas_zgemm(CblasConjTrans,CblasNoTrans,left.nCols(),right.nCols(),left.colLength(),
		volume,left.data(),left.colLength(),right.data(),right.colLength(),0.,result.data(),left.nCols());
	return result;
}
double occupiedSVD(const matrix& overlap, matrix& u, diagMatrix& singular, matrix& vdag)
{
	int n = overlap.nRows();
	if(n>32)
	{	overlap.svd(u,singular,vdag);
		complex determinant = det(u*vdag);
		return std::atan2(determinant.imag(),determinant.real());
	}
	// The general matrix SVD/copy helpers prefer GPU memory even in their
	// small-matrix LAPACK fallback. Keep the whole tiny factorization on CPU.
	u.init(n,n); vdag.init(n,n); singular.resize(n);
	const complex* data = overlap.data();
	std::vector<complex> a(data,data+n*n), work(8*n*n+8*n);
	std::vector<double> rwork(5*n);
	char job = 'A'; int lwork = work.size(), info = 0;
	zgesvd_(&job,&job,&n,&n,a.data(),&n,singular.data(),u.data(),&n,vdag.data(),&n,
		work.data(),&lwork,rwork.data(),&info);
	if(info) return NAN;
	// Sum LU diagonal arguments, including pivot parity, to avoid multiplying
	// small determinants. SVD still supplies the singular-value guard / inverse.
	a.assign(data,data+n*n);
	std::vector<int> pivot(n);
	zgetrf_(&n,&n,a.data(),&n,pivot.data(),&info);
	if(info) return NAN;
	double phase = 0.;
	for(int j=0; j<n; j++)
	{	phase += std::atan2(a[j*(n+1)].imag(),a[j*(n+1)].real());
		if(pivot[j]!=j+1) phase += M_PI;
	}
	return std::remainder(phase,2*M_PI);
}
matrix occupiedInverse(const matrix& u, const diagMatrix& singular, const matrix& vdag)
{
	int n = u.nRows();
	if(n>32) return dagger(vdag)*inv(singular)*dagger(u);
	matrix result(n,n);
	const complex* ud = u.data(); const complex* vd = vdag.data();
	complex* out = result.data();
	for(int j=0; j<n; j++) for(int i=0; i<n; i++)
	{	complex value = 0.;
		for(int k=0; k<n; k++) value += conj(vd[i*n+k])*conj(ud[k*n+j])/singular[k];
		out[j*n+i] = value;
	}
	return result;
}
void addOccupiedProduct(ColumnBundle& result, const ColumnBundle& states, const matrix& inverse, complex scale, bool adjoint)
{
	if(inverse.nRows()>32)
	{	result += scale*ColumnBundle(states*(adjoint ? dagger(inverse) : scaled<matrix>(inverse)));
		return;
	}
	eblas_zgemm(CblasNoTrans,adjoint ? CblasConjTrans : CblasNoTrans,states.colLength(),states.nCols(),states.nCols(),
		scale,states.data(),states.colLength(),inverse.data(),states.nCols(),1.,result.data(),result.colLength());
}
}

void BerryPhase::setup(const Everything& e)
{
	const auto& ei = e.eInfo;
	if(e.cntrl.fixed_H || (e.cntrl.scf && field.length_squared()))
		die("berry-phase-field requires self-consistent electronic-minimize at nonzero field (no electronic-scf or fixed Hamiltonian).\n");
	if(e.cntrl.convergeEmptyStates)
		die("berry-phase-field forbids converge-empty-states: independent-k diagonalization would undo the polarized occupied subspace.\n");
	if(e.pertInfo.solverParams.nIterations || e.pertInfo.phononGrid.isNonzero())
		die("berry-phase-field cannot be combined with perturb-minimize or phonon-dfpt; finite-field DFPT Hessians are not implemented.\n");
	if(ei.fillingsUpdate != ElecInfo::FillingsConst || !ei.scalarFillings || !std::isnan(ei.mu))
		die("berry-phase-field requires constant, fully occupied bands and fixed electron number; no smearing.\n");
	int badFilling = 0;
	for(int q=ei.qStart; q<ei.qStop; q++)
		for(double f: e.eVars.F[q]) if(std::abs(f-1.) > 1e-12) badFilling = 1;
	mpiWorld->allReduce(badFilling, MPIUtil::ReduceMax);
	if(badFilling) die("berry-phase-field requires only filled bands (all internal occupations must be one).\n");
	if(e.coulombParams.geometry != CoulombParams::Periodic || e.coulombParams.Efield.length_squared()
		|| e.eVars.fluidParams.fluidType != FluidNone)
		die("berry-phase-field requires periodic Coulomb interaction, no electric-field, and no fluid.\n");
	if(e.vibrations || e.iInfo.ljOverride || e.ionicDynParams.nSteps)
		die("berry-phase-field does not support vibrations, ionic dynamics or pair-potential overrides.\n");
	for(auto sp: e.iInfo.species)
		if(sp->isUltrasoft()) die("berry-phase-field currently requires norm-conserving pseudopotentials; ultrasoft Berry overlaps/forces are not implemented.\n");
	double ionCharge = 0.;
	for(auto sp: e.iInfo.species) ionCharge += sp->Z * sp->atpos.size();
	if(std::abs(ionCharge-ei.nElectrons) > 1e-8)
		die("berry-phase-field requires a neutral insulating cell.\n");
	mesh = ei.kFoldingCount();
	const int nk = mesh[0]*mesh[1]*mesh[2];
	if(ei.nStates != nk*ei.nSpins())
		die("berry-phase-field requires one complete uniform kpoint-folding mesh per spin, without symmetry reduction.\n");
	std::vector<int> qAt(nk*ei.nSpins(), -1);
	std::vector<vector3<int>> coordinate(ei.nStates);
	const vector3<> origin = ei.qnums[0].k;
	for(int q=0; q<ei.nStates; q++)
	{	vector3<int> x;
		for(int d=0; d<3; d++)
		{	double v = (ei.qnums[q].k[d]-origin[d])*mesh[d];
			if(std::abs(v-std::round(v)) > 1e-7) die("berry-phase-field: k mesh is not a regular folded mesh.\n");
			x[d] = (int(std::round(v))%mesh[d]+mesh[d])%mesh[d];
		}
		int index = meshIndex(x,mesh)+nk*ei.qnums[q].index();
		if(qAt[index]>=0 || std::abs(ei.qnums[q].weight-double(ei.spinWeight)/nk)>1e-10)
			die("berry-phase-field requires unique, equally weighted k points.\n");
		qAt[index] = q;
		coordinate[q] = x;
	}
	std::vector<std::map<std::array<int,3>,int>> basisIndex(ei.nStates);
	for(int q=0; q<ei.nStates; q++)
	{	const auto& basis = e.basis[q];
		const auto* g = basis.iGarr.data();
		for(size_t j=0; j<basis.nbasis; j++) basisIndex[q][key(g[j])] = j;
	}
	links.clear();
	int stringOffset = 0;
	for(int d=0; d<3; d++)
	{	for(int s=0; s<ei.nSpins(); s++)
			for(int q=0; q<ei.nStates; q++) if(ei.qnums[q].index()==s && coordinate[q][d]==0)
			{	int current = q;
				for(int j=0; j<mesh[d]; j++)
				{	auto x = coordinate[current]; x[d] = (x[d]+1)%mesh[d];
					int next = qAt[meshIndex(x,mesh)+s*nk];
					vector3<int> wrap;
					for(int a=0; a<3; a++) wrap[a] = std::lround(ei.qnums[current].k[a]+(a==d ? 1./mesh[d] : 0.)-ei.qnums[next].k[a]);
					Link link{current,next,d,stringOffset,{}};
					const auto& basis = e.basis[current];
					const auto* g = basis.iGarr.data();
					for(size_t b=0; b<basis.nbasis; b++)
					{	auto it = basisIndex[next].find(key(g[b]+wrap));
						link.source.push_back(it==basisIndex[next].end() ? -1 : it->second);
					}
					links.push_back(std::move(link)); current = next;
				}
				stringOffset++;
			}
	}
	Citations::add("Berry-phase electric-enthalpy minimization",+		"I. Souza, J. Iniguez and D. Vanderbilt, First-Principles Approach to Insulators in Finite Electric Fields,\n"
		"Phys. Rev. Lett. 89, 117602 (2002), doi:10.1103/PhysRevLett.89.117602");
	if(e.eVars.skipWfnsInit) return; // dry run: validate parameters without evaluating absent orbitals
	ready = true;
	logPrintf("\nBerry-phase electric enthalpy: physical field [%g %g %g] a.u.; full %d x %d x %d occupied mesh.\n",
		field[0],field[1],field[2],mesh[0],mesh[1],mesh[2]);
	// Initialize the branch. Trial evaluations never modify accepted references.
	referencePhase.clear();
	if(!std::isfinite(evaluate(e)))
		die("Cannot initialize Berry polarization: adjacent occupied overlaps are singular or nonfinite; improve the initial state / k mesh.\n");
}

void BerryPhase::acceptPhase()
{	if(ready) referencePhase = currentPhase;
}

ColumnBundle BerryPhase::forward(const ColumnBundle& input, const Everything& e, const Link& link) const
{
	const auto& b = e.basis[link.q];
	ColumnBundle result(input.nCols(),b.nbasis*input.spinorLength(),&b,&e.eInfo.qnums[link.q]);
	const auto* src = input.data(); auto* dst = result.data();
	std::fill(dst,dst+result.nData(),complex(0.,0.));
	for(int n=0; n<input.nCols(); n++) for(int s=0; s<input.spinorLength(); s++)
		for(size_t j=0; j<b.nbasis; j++) if(link.source[j]>=0)
			dst[result.index(n,s*b.nbasis+j)] = src[input.index(n,s*input.basis->nbasis+link.source[j])];
	return result;
}

ColumnBundle BerryPhase::backward(const ColumnBundle& input, const Everything& e, const Link& link) const
{
	const auto& b = e.basis[link.next];
	ColumnBundle result(input.nCols(),b.nbasis*input.spinorLength(),&b,&e.eInfo.qnums[link.next]);
	const auto* src = input.data(); auto* dst = result.data();
	std::fill(dst,dst+result.nData(),complex(0.,0.));
	for(int n=0; n<input.nCols(); n++) for(int s=0; s<input.spinorLength(); s++)
		for(size_t j=0; j<link.source.size(); j++) if(link.source[j]>=0)
			dst[result.index(n,s*b.nbasis+link.source[j])] = src[input.index(n,s*input.basis->nbasis+j)];
	return result;
}

double BerryPhase::evaluate(const Everything& e, std::vector<ColumnBundle>* gradient)
{
	const auto& ei = e.eInfo;
	const double volume = e.gInfo.detR;
	// Replicate occupied states only. All ranks evaluate identical string phases;
	// each rank accumulates derivatives only for the k points it owns.
	std::vector<ColumnBundle> states(ei.nStates);
	std::vector<complex> broadcastBuffer;
	for(int q=0; q<ei.nStates; q++)
	{	if(ei.isMine(q)) states[q] = e.eVars.C[q];
		else states[q].init(ei.nBands,e.basis[q].nbasis*ei.spinorLength(),&e.basis[q],&ei.qnums[q]);
		if(isGpuEnabled() && mpiWorld->nProcesses()>1)
		{	// MPI shared-memory transfers must use ordinary host storage, rather
			// than managed allocations that can still be migrating from the GPU.
			broadcastBuffer.resize(states[q].nData());
			if(ei.isMine(q)) std::copy(states[q].data(),states[q].data()+states[q].nData(),broadcastBuffer.begin());
			mpiWorld->bcast(broadcastBuffer.data(),broadcastBuffer.size(),ei.whose(q));
			if(!ei.isMine(q)) std::copy(broadcastBuffer.begin(),broadcastBuffer.end(),states[q].data());
		}
		else mpiWorld->bcast(states[q].data(),states[q].nData(),ei.whose(q));
	}
	if(gradient)
	{	gradient->assign(ei.nStates,ColumnBundle());
		for(int q=ei.qStart; q<ei.qStop; q++)
		{	(*gradient)[q] = states[q].similar();
			std::fill((*gradient)[q].data(),(*gradient)[q].data()+(*gradient)[q].nData(),complex(0.,0.));
		}
	}
	int nStrings = links.empty() ? 0 : links.back().string+1;
	std::vector<double> phase(nStrings,0.);
	minOverlap = 1.;
	for(const auto& link: links)
	{	ColumnBundle right = forward(states[link.next],e,link);
		matrix overlap = occupiedOverlap(states[link.q],right,volume);
		matrix u, vdag; diagMatrix singular;
		double linkPhase = occupiedSVD(overlap,u,singular,vdag);
		if(!std::isfinite(linkPhase)) return NAN;
		double smallest = *std::min_element(singular.begin(),singular.end());
		if(!std::isfinite(smallest) || smallest < 1e-8) return NAN; // invalid trial: let the line search backtrack
		minOverlap = std::min(minOverlap,smallest);
		phase[link.string] += linkPhase;
		if(gradient)
		{	const int d = link.direction;
			double ea = 0.; for(int a=0; a<3; a++) ea += field[a]*e.gInfo.R(a,d);
			double coefficient = ei.spinWeight*ea/(2*M_PI*(ei.nStates/ei.nSpins()/mesh[d]));
			if(coefficient)
			{	matrix inverse = occupiedInverse(u,singular,vdag);
				if(ei.isMine(link.q)) addOccupiedProduct((*gradient)[link.q],right,inverse,complex(0.,0.5*coefficient*volume),false);
				if(ei.isMine(link.next)) addOccupiedProduct((*gradient)[link.next],backward(states[link.q],e,link),inverse,complex(0.,-0.5*coefficient*volume),true);
			}
		}
	}
	if(referencePhase.empty())
	{	// Choose a branch at initialization, separately for each direction / spin.
		// Accepted iterates advance these references; line-search trials keep them fixed.
		referencePhase.resize(nStrings);
		for(int d=0; d<3; d++) for(int s=0; s<ei.nSpins(); s++)
		{	std::map<vector3<int>,int> transverse;
			for(const auto& link: links) if(link.direction==d && ei.qnums[link.q].index()==s)
			{	vector3<int> x;
				for(int a=0; a<3; a++) x[a] = a==d ? 0 : (int(std::lround((ei.qnums[link.q].k[a]-ei.qnums[0].k[a])*mesh[a]))%mesh[a]+mesh[a])%mesh[a];
				transverse[x] = link.string;
			}
			for(const auto& entry: transverse)
			{	double anchor = 0.;
				for(int a=0; a<3; a++) if(entry.first[a]>0)
				{	auto previous = entry.first; previous[a]--;
					anchor = referencePhase[transverse.at(previous)]; break;
				}
				referencePhase[entry.second] = anchor+std::remainder(phase[entry.second]-anchor,2*M_PI);
			}
		}
	}
	electronic = vector3<>(); ionic = vector3<>();
	int previousString = -1;
	std::vector<double> unwrappedPhase(nStrings);
	phaseDirection.resize(nStrings);
	for(const auto& link: links) if(link.string != previousString)
	{	previousString = link.string;
		phaseDirection[link.string] = link.direction;
		double change = std::remainder(phase[link.string]-referencePhase[link.string],2*M_PI);
		double ea = dot(field,e.gInfo.R.column(link.direction));
		// Stay inside the local branch during a trial. Advancing accepted phases
		// permits cumulative changes beyond 2pi without resetting the enthalpy.
		if(!std::isfinite(change) || (ea && std::abs(change)>0.5*M_PI)) return NAN;
		double p = referencePhase[link.string]+change;
		unwrappedPhase[link.string] = p;
		double coefficient = ei.spinWeight*p/(2*M_PI*volume*(ei.nStates/ei.nSpins()/mesh[link.direction]));
		for(int a=0; a<3; a++) electronic[a] += e.gInfo.R(a,link.direction)*coefficient;
	}
	currentPhase = std::move(unwrappedPhase);
	electronic += (1./volume)*(e.gInfo.R*vector3<>(polarizationBranch));
	for(auto sp: e.iInfo.species) for(const auto& pos: sp->atpos) ionic += (sp->Z/volume)*(e.gInfo.R*pos);
	return -volume*dot(field,electronic+ionic);
}

matrix3<> BerryPhase::latticeGradient(const Everything& e) const
{
	// At fixed normalized reduced-coordinate orbitals and fractional ions, the
	// phases are strain-independent and the dipole transforms as D -> (1+eta) D.
	// Thus d(-E.D)/d eta = -E (x) D at fixed Cartesian E [SIV, Eq. (6)].
	// LatticeMinimizer varies symmetric strain; exclude the rigid-rotation torque.
	vector3<> dipole = e.gInfo.detR*(electronic+ionic);
	return -0.5*(outer(field,dipole)+outer(dipole,field));
}

void BerryPhase::dump(const Everything& e)
{
	double coupling = evaluate(e);
	if(!std::isfinite(coupling)) die("Cannot dump Berry polarization outside the accepted local branch.\n");
	string filename = e.dump.getFilename("polarization");
	if(mpiWorld->isHead())
	{	FILE* fp = fopen(filename.c_str(),"w");
		if(!fp) die("Could not write Berry polarization file '%s'.\n",filename.c_str());
		fprintf(fp,"# Berry-phase polarization: physical positive-ion convention, atomic units\n");
		fprintf(fp,"# Electronic phases follow accepted iterates; compare nearby fields on a common branch.\n");
		fprintf(fp,"field %.16e %.16e %.16e\n",field[0],field[1],field[2]);
		fprintf(fp,"electronic %.16e %.16e %.16e\n",electronic[0],electronic[1],electronic[2]);
		fprintf(fp,"ionic %.16e %.16e %.16e\n",ionic[0],ionic[1],ionic[2]);
		auto total = electronic+ionic;
		fprintf(fp,"total %.16e %.16e %.16e\n",total[0],total[1],total[2]);
		fprintf(fp,"field_energy %.16e\n",coupling);
		fprintf(fp,"internal_energy %.16e\n",double(e.ener.E)-e.ener.E["EfieldBerry"]);
		fprintf(fp,"enthalpy %.16e\n",double(e.ener.E)-e.ener.E["EfieldBerry"]+coupling);
		fprintf(fp,"min_overlap %.16e\n",minOverlap);
		fprintf(fp,"spin_weight %d\n",e.eInfo.spinWeight);
		fprintf(fp,"polarization_branch %d %d %d\n",polarizationBranch[0],polarizationBranch[1],polarizationBranch[2]);
		fprintf(fp,"lattice"); for(int a=0; a<3; a++) for(int d=0; d<3; d++) fprintf(fp," %.16e",e.gInfo.R(a,d)); fprintf(fp,"\n");
		fprintf(fp,"berry_phases"); for(double phase: currentPhase) fprintf(fp," %.16e",phase); fprintf(fp,"\n");
		fprintf(fp,"berry_phase_directions"); for(int direction: phaseDirection) fprintf(fp," %d",direction); fprintf(fp,"\n");
		fclose(fp);
	}
	logPrintf("Berry polarization (e/bohr^2): electronic [%+.10e %+.10e %+.10e], ionic [%+.10e %+.10e %+.10e]; min overlap %.6g\n",
		electronic[0],electronic[1],electronic[2],ionic[0],ionic[1],ionic[2],minOverlap);
}
