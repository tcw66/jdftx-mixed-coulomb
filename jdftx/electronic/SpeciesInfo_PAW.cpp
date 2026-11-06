/* Native PAW integration. GNU GPL v3 or later; see SpeciesInfo.h. */
#include <electronic/SpeciesInfo.h>
#include <electronic/PAW.h>
#include <electronic/Everything.h>
#include <electronic/ColumnBundle.h>
#include <electronic/SpeciesInfo_internal.h>
#include <core/SphericalHarmonics.h>
#include <stdexcept>

void SpeciesInfo::readPAW(istream &input)
{
	try
	{
		paw = std::make_shared<PAW>(input);
	}
	catch (const std::exception &ex)
	{
		die("%s\n", ex.what());
	}
	if (!e->exCorr.supportsPAW() || !e->exCorrDiff.empty())
		die("PAW requires internal PBE without alternate functionals.\n");
	if (e->eInfo.spinType != SpinNone)
		die("PAW currently requires unpolarized scalar wavefunctions.\n");
	if (e->ionicMinParams.nIterations || e->latticeMinParams.nIterations || e->ionicDynParams.nSteps ||
	    e->vibrations || e->iInfo.computeStress || e->pertInfo.solverParams.nIterations ||
	    e->pertInfo.phononGrid.isNonzero())
		die("PAW currently supports fixed ions only (no forces, stress, MD, or response).\n");
	if (!plusU.empty() || e->eVars.fluidParams.fluidType != FluidNone || e->berryPhase)
		die("PAW does not yet support DFT+U, solvation, or Berry-phase electric fields.\n");
	for (const auto &item : e->dump)
		if (item.second == DumpForces || item.second == DumpStress)
			die("PAW force and stress output is not implemented.\n");

	Z = paw->valence;
	atomicNumber = paw->atomicNumber;
	coreRadius = paw->radius;
	Citations::add("Projector augmented wave", "P. E. Blochl, Phys. Rev. B 50, 17953 (1994)");
	Citations::add("JTH PAW datasets",
	               "F. Jollet, M. Torrent and N. Holzwarth, Comput. Phys. Commun. 185, 1246 (2014)");
	const double dg = e->gInfo.dGradial;
	const int ng = int(ceil(e->gInfo.GmaxSphere / dg)) + 5, ngloc = int(ceil(e->gInfo.GmaxGrid / dg)) + 5;
	const int nl = paw->lMax + 1;
	VnlRadial.resize(nl);
	Mnl.resize(nl);
	Qint.resize(nl);
	std::vector<std::vector<int>> indices(nl);
	std::vector<std::vector<RadialFunctionR>> psi(nl);
	RadialFunctionR radial = paw->grid;
	for (size_t i = 0; i < paw->states.size(); i++)
	{
		const auto &st = paw->states[i];
		indices[st.l].push_back(i);
		VnlRadial[st.l].emplace_back();
		radial.f = st.projector;
		radial.transform(st.l, dg, ng, VnlRadial[st.l].back());
		if (st.n && st.occupation > 0.)
		{
			radial.f = st.ps;
			psi[st.l].push_back(radial);
			atomEigs.resize(nl);
			atomEigs[st.l].push_back(st.energy);
		}
	}
	for (int l = 0; l < nl; l++)
	{
		int count = indices[l].size();
		if (!count)
			continue;
		Mnl[l] = zeroes(count, count);
		Qint[l] = zeroes(count, count);
		for (int i = 0; i < count; i++)
			for (int j = 0; j < count; j++)
			{
				const auto &a = paw->states[indices[l][i]];
				const auto &b = paw->states[indices[l][j]];
				std::vector<double> diff(radial.r.size());
				for (size_t r = 0; r < diff.size(); r++)
					if (radial.r[r] <= coreRadius + 1e-8)
						diff[r] = a.ae[r] * b.ae[r] - a.ps[r] * b.ps[r];
				Mnl[l].set(i, j, paw->kinetic(indices[l][i], indices[l][j]));
				Qint[l].set(i, j, paw->moment(diff));
			}
	}
	// Smooth density augmentation carries compensation multipoles, not the sharp
	// AE-minus-PS products. The latter are retained independently for observables.
	for (size_t i = 0; i < paw->states.size(); i++)
		for (size_t j = 0; j <= i; j++)
		{
			const auto &a = paw->states[i];
			const auto &b = paw->states[j];
			std::vector<double> diff(radial.r.size());
			for (size_t r = 0; r < diff.size(); r++)
				if (radial.r[r] <= coreRadius + 1e-8)
					diff[r] = a.ae[r] * b.ae[r] - a.ps[r] * b.ps[r];
			for (int l = abs(a.l - b.l); l <= a.l + b.l; l += 2)
			{
				double q = paw->moment(diff, l);
				radial.f = paw->shape[l];
				for (double &x : radial.f)
					x *= q;
				radial.transform(l, dg, ngloc, Qradial[{a.l, a.p, b.l, b.p, l}]);
			}
		}
	// Smooth ionic potential has tail -Z/r, handled by the existing Coulomb code.
	radial = paw->grid;
	radial.f = paw->ionicPotential;
	for (size_t r = 0; r < radial.r.size(); r++)
		if (radial.r[r])
			radial.f[r] += Z / radial.r[r];
	radial.transform(0, dg, ngloc, VlocRadial);
	radial = paw->grid;
	radial.f = paw->corePS;
	setCore(radial);
	setPsi(psi);
	paw->rho.assign(atpos.size(), zeroes(paw->channels.size(), paw->channels.size()));
	for (matrix &rho : paw->rho)
		for (size_t i = 0; i < paw->channels.size(); i++)
		{
			const auto &st = paw->states[paw->channels[i].state];
			rho.set(i, i, st.occupation / (2 * st.l + 1));
		}
	paw->potential = paw->rho;
	logPrintf("  PAW: frozen-core all-electron valence reconstruction enabled; fixed ions, internal PBE.\n");
}

void SpeciesInfo::pawAccumulateDensity(const QuantumNumber &q, const diagMatrix &f, const matrix &projections)
{
	if (!paw)
		return;
	int n = paw->channels.size();
	for (size_t a = 0; a < atpos.size(); a++)
	{
		matrix p = projections(a * n, (a + 1) * n, 0, projections.nCols());
		paw->rho[a] += q.weight * paw->phaseMatrix(p * f * dagger(p), true);
	}
}

double SpeciesInfo::pawEnergyAndPotential(const ExCorr &xc)
{
	if (!paw)
		return 0.;
	double energy = 0.;
	for (size_t a = 0; a < atpos.size(); a++)
		energy += paw->energyAndPotential(paw->rho[a], xc, paw->potential[a]);
	return energy;
}

ColumnBundle SpeciesInfo::pawReconstruct(const ColumnBundle &c, const Basis &target) const
{
	ColumnBundle result(c.nCols(), target.nbasis, &target, c.qnum, isGpuEnabled());
	result.zero();
	if (!paw)
		return result;
	const int n = paw->channels.size();
	matrix proj = (*getV(c)) ^ c;
	const double dg = e->gInfo.dGradial;
	double gmax = 0.;
	for (size_t g = 0; g < target.nbasis; g++)
		gmax = std::max(gmax, ((target.iGarr.data()[g] + c.qnum->k) * e->gInfo.G).length());
	int ng = int(ceil(gmax / dg)) + 5;
	ColumnBundle delta(n * atpos.size(), target.nbasis, &target, c.qnum, isGpuEnabled());
	int channel = 0;
	RadialFunctionR radial = paw->grid;
	for (const auto &ch : paw->channels)
	{
		const auto &st = paw->states[ch.state];
		for (size_t r = 0; r < radial.f.size(); r++)
			radial.f[r] = radial.r[r] <= coreRadius + 1e-8 ? st.ae[r] - st.ps[r] : 0.;
		RadialFunctionG radialG;
		radial.transform(ch.l, dg, ng, radialG);
		// Fourier (-i)^l and physical-projection i^l cancel in JDFTx conventions.
		callPref(Vnl)(target.nbasis, n * target.nbasis, atpos.size(), ch.l, ch.m, c.qnum->k,
		              target.iGarr.dataPref(), e->gInfo.G, atposManaged.dataPref(), radialG,
		              delta.dataPref() + channel * target.nbasis);
		radialG.free();
		channel++;
	}
	result = (1. / e->gInfo.detR) * delta * proj;
	return result;
}

matrix SpeciesInfo::pawMomentum(const ColumnBundle &c, int dir) const
{
	matrix result = zeroes(c.nCols(), c.nCols());
	if (!paw)
		return result;
	matrix proj = (*getV(c)) ^ c;
	matrix correction = paw->phaseMatrix(paw->momentumCorrection()[dir]);
	return dagger(proj) * (tiledBlockMatrix(correction, atpos.size()) * proj);
}

void SpeciesInfo::pawSymmetrizeDensity()
{
	if (!paw)
		return;
	for (matrix &r : paw->rho)
	{
		mpiWorld->allReduceData(r, MPIUtil::ReduceSum);
		for (size_t i = 0; i < r.nData(); i++)
			r.data()[i] = r.data()[i].real();
	}
	size_t isp = 0;
	while (e->iInfo.species[isp].get() != this)
		++isp;
	const auto &map = e->symm.getAtomMap()[isp];
	size_t nsym = e->symm.getMatrices().size();
	std::vector<matrix> symrho(atpos.size(), zeroes(paw->channels.size(), paw->channels.size()));
	for (size_t s = 0; s < nsym; s++)
	{
		matrix rotation = zeroes(paw->channels.size(), paw->channels.size());
		int offset = 0;
		for (int l = 0; l <= paw->lMax; l++)
			for (const auto &st : paw->states)
				if (st.l == l)
				{
					int count = 2 * l + 1;
					rotation.set(offset, offset + count, offset, offset + count,
					             e->symm.getSphericalMatrices(l, false)[s]);
					offset += count;
				}
		for (size_t a = 0; a < atpos.size(); a++)
			symrho[map[a][s]] += (1. / nsym) * rotation * paw->rho[a] * dagger(rotation);
	}
	paw->rho = std::move(symrho);
}

matrix SpeciesInfo::pawOverlap(const vector3<> &dk) const
{
	matrix result = zeroes(paw->channels.size(), paw->channels.size());
	vector3<> cart = dk * e->gInfo.G;
	double length = cart.length();
	vector3<> direction = length ? cart * (1. / length) : vector3<>();
	RadialFunctionR radial = paw->grid;
	for (const auto &pair : paw->pairs)
	{
		int l = int(sqrt(double(pair.lm))), m = pair.lm - l * (l + 1);
		for (size_t r = 0; r < radial.f.size(); r++)
			radial.f[r] = pair.ae[r] - pair.ps[r] + pair.hat[r];
		complex x = pair.gaunt * cis(-0.5 * M_PI * l) * Ylm(l, m, direction) * radial.transform(l, length);
		result.data()[result.index(pair.i, pair.j)] += x;
		if (pair.i != pair.j)
			result.data()[result.index(pair.j, pair.i)] += x;
	}
	return paw->phaseMatrix(result);
}
