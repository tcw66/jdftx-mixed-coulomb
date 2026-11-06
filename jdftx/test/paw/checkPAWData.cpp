// Physical identities and finite-difference checks for native PAW datasets.
#include <electronic/PAW.h>
#include <electronic/ExCorr.h>
#include <core/Util.h>
#include <fstream>
#include <stdexcept>
void close(double actual, double expected, double tolerance, const char *label)
{
	logPrintf("%s: actual %.12g expected %.12g error %.3g\n", label, actual, expected,
	          fabs(actual - expected));
	if (!std::isfinite(actual) || fabs(actual - expected) > tolerance)
		die("PAW dataset regression failed: %s\n", label);
}
int main(int argc, char **argv)
{
	initSystem(argc, argv);
	if (argc != 2 && argc != 3)
		die("Usage: checkPAWData dataset.xml [--parse-only]\n");
	try
	{
		std::ifstream input(argv[1]);
		if (!input)
			die("Cannot open test dataset\n");
		PAW paw(input);
		if (argc != 3 || std::string(argv[2]) != "--parse-only")
		{
			ExCorr xc;
			close(4 * M_PI * paw.moment(paw.coreAE), paw.atomicNumber - paw.valence, 2e-3,
			      "Frozen core electron count");
			for (size_t l = 0; l < paw.shape.size(); l++)
				close(paw.moment(paw.shape[l], l), 1., 1e-12, "Compensation multipole normalization");
			std::vector<double> gaussian(paw.grid.r.size());
			for (size_t r = 0; r < gaussian.size(); r++)
				gaussian[r] = exp(-pow(paw.grid.r[r], 2)) / pow(M_PI, 1.5);
			auto v = paw.poisson(gaussian, 0);
			double energy = 0.;
			for (size_t r = 0; r < v.size(); r++)
				energy += 2 * M_PI * paw.grid.dr[r] * pow(paw.grid.r[r], 2) * gaussian[r] * v[r];
			close(v.front(), 2 / sqrt(M_PI), 1e-4, "Gaussian Coulomb potential at nucleus");
			close(energy, 1 / sqrt(2 * M_PI), 1e-4, "Gaussian Coulomb self energy");
			int n = paw.channels.size();
			matrix rho = zeroes(n, n), direction = zeroes(n, n);
			for (int i = 0; i < n; i++)
			{
				const auto &st = paw.states[paw.channels[i].state];
				rho.set(i, i, st.occupation / (2 * st.l + 1) + .01);
				for (int j = 0; j <= i; j++)
				{
					double x = .01 * cos((i + 1) * (j + 2));
					direction.set(i, j, x);
					direction.set(j, i, x);
				}
			}
			matrix potential, unused;
			double e = paw.energyAndPotential(rho, xc, potential);
			close(nrm2(potential - dagger(potential)), 0., 1e-10, "Hermitian on-site potential");
			logPrintf("Onsite reference energy %.17g\n", e);
			double h = 1e-4, fd = (paw.energyAndPotential(rho + h * direction, xc, unused) -
			                       paw.energyAndPotential(rho - h * direction, xc, unused)) /
			                      (2 * h);
			close(fd, trace(potential * direction).real(), 2e-6, "On-site PBE energy directional derivative");
			close(nrm2(paw.phaseMatrix(paw.phaseMatrix(potential), true) - potential), 0., 1e-10,
			      "Projector phase convention round trip");
			auto momentum = paw.momentumCorrection();
			for (int k = 0; k < 3; k++)
				close(nrm2(momentum[k] - dagger(momentum[k])), 0., 1e-12, "Hermitian AE momentum correction");
			logPrintf("PAW dataset tests passed (energy %g Hartree).\n", e);
		}
	}
	catch (const std::exception &error)
	{
		die("%s\n", error.what());
	}
	finalizeSystem();
}
