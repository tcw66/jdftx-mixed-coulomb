#include <electronic/BerryPhase.h>
#include <electronic/Everything.h>
#include <electronic/SpeciesInfo.h>
#include <map>
#include <array>
#include <algorithm>
#include <cmath>

namespace {
int meshIndex(const vector3<int>& x, const vector3<int>& n)
{ return (x[0]*n[1]+x[1])*n[2]+x[2]; }
std::array<int,3> key(const vector3<int>& x)
{ return {{x[0],x[1],x[2]}}; }
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
	if(e.iInfo.computeStress || e.latticeMinParams.nIterations || e.vibrations || e.iInfo.ljOverride)
		die("berry-phase-field supports fixed-strain electronic and internal-ion minimization; stress, lattice relaxation and vibrations are unsupported.\n");
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
	for(int q=0; q<ei.nStates; q++)
	{	if(ei.isMine(q)) states[q] = e.eVars.C[q];
		else states[q].init(ei.nBands,e.basis[q].nbasis*ei.spinorLength(),&e.basis[q],&ei.qnums[q]);
		mpiWorld->bcast(states[q].data(),states[q].nData(),ei.whose(q));
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
		matrix overlap = volume * (states[link.q]^right);
		matrix u, vdag; diagMatrix singular;
		overlap.svd(u,singular,vdag);
		double smallest = *std::min_element(singular.begin(),singular.end());
		if(!std::isfinite(smallest) || smallest < 1e-8) return NAN; // invalid trial: let the line search backtrack
		minOverlap = std::min(minOverlap,smallest);
		complex determinant = det(u*vdag); // phase only; avoids determinant underflow
		phase[link.string] += std::atan2(determinant.imag(),determinant.real());
		if(gradient)
		{	const int d = link.direction;
			double ea = 0.; for(int a=0; a<3; a++) ea += field[a]*e.gInfo.R(a,d);
			double coefficient = ei.spinWeight*ea/(2*M_PI*(ei.nStates/ei.nSpins()/mesh[d]));
			if(coefficient)
			{	matrix inverse = dagger(vdag)*inv(singular)*dagger(u);
				if(ei.isMine(link.q)) (*gradient)[link.q] += complex(0.,0.5*coefficient*volume) * ColumnBundle(right*inverse);
				if(ei.isMine(link.next)) (*gradient)[link.next] += complex(0.,-0.5*coefficient*volume) * ColumnBundle(backward(states[link.q],e,link)*dagger(inverse));
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
	for(auto sp: e.iInfo.species) for(const auto& pos: sp->atpos) ionic += (sp->Z/volume)*(e.gInfo.R*pos);
	return -volume*dot(field,electronic+ionic);
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
		fprintf(fp,"berry_phases"); for(double phase: currentPhase) fprintf(fp," %.16e",phase); fprintf(fp,"\n");
		fprintf(fp,"berry_phase_directions"); for(int direction: phaseDirection) fprintf(fp," %d",direction); fprintf(fp,"\n");
		fclose(fp);
	}
	logPrintf("Berry polarization (e/bohr^2): electronic [%+.10e %+.10e %+.10e], ionic [%+.10e %+.10e %+.10e]; min overlap %.6g\n",
		electronic[0],electronic[1],electronic[2],ionic[0],ionic[1],ionic[2],minOverlap);
}
