// Full periodic PAW functional versus orbital finite differences.
#include <electronic/Everything.h>
#include <electronic/ElecMinimizer.h>
#include <electronic/PAW.h>
#include <commands/parser.h>
void close(double a, double b, double tol, const char *label)
{
	logPrintf("%s: actual %.12g expected %.12g error %.3g\n", label, a, b, fabs(a - b));
	if (!std::isfinite(a) || fabs(a - b) > tol)
		die("PAW integration regression failed: %s\n", label);
}
int main(int argc, char **argv)
{
	InitParams ip("PAW energy and Hamiltonian regression");
	initSystemCmdline(argc, argv, ip);
	{
		Everything e;
		parse(readInputFile(ip.inputFilename), e, ip.printDefaults);
		e.setup();
		auto original = e.eVars.C;
		ElecGradient grad;
		grad.init(e);
		e.eVars.elecEnergyAndGrad(e.ener, &grad);
		std::vector<ColumnBundle> direction(e.eInfo.nStates);
		double norm = 0., derivative = 0.;
		for (int q = e.eInfo.qStart; q < e.eInfo.qStop; q++)
		{
			direction[q] = original[q].similar();
			for (size_t i = 0; i < direction[q].nData(); i++)
				direction[q].data()[i] =
				    complex(sin(.137 * (i + 1) * (q + 1)), cos(.071 * (i + 3) * (q + 2)));
			direction[q] -= original[q] * (original[q] ^ O(direction[q]));
			norm += dotc(direction[q], direction[q]).real();
		}
		mpiWorld->allReduce(norm, MPIUtil::ReduceSum);
		for (int q = e.eInfo.qStart; q < e.eInfo.qStop; q++)
		{
			direction[q] *= 1. / sqrt(norm);
			derivative += 2 * dotc(grad.C[q], direction[q]).real();
		}
		mpiWorld->allReduce(derivative, MPIUtil::ReduceSum);
		double onsiteDerivative = 0.;
		for (int q = e.eInfo.qStart; q < e.eInfo.qStop; q++)
			for (size_t is = 0; is < e.iInfo.species.size(); is++)
			{
				auto sp = e.iInfo.species[is];
				if (!sp->paw)
					continue;
				int n = sp->paw->channels.size();
				matrix p = (*sp->getV(original[q])) ^ original[q], hp = zeroes(p.nRows(), p.nCols());
				for (size_t a = 0; a < sp->atpos.size(); a++)
					hp.set(a * n, (a + 1) * n, 0, p.nCols(),
					       sp->paw->phaseMatrix(sp->paw->potential[a]) * p(a * n, (a + 1) * n, 0, p.nCols()));
				ColumnBundle hc = (*sp->getV(original[q])) * hp;
				onsiteDerivative +=
				    2 * e.eInfo.qnums[q].weight * dotc(hc * e.eVars.F[q], direction[q]).real();
			}
		mpiWorld->allReduce(onsiteDerivative, MPIUtil::ReduceSum);
		std::vector<matrix> basePotential, trialRho[2];
		for (auto sp : e.iInfo.species)
			if (sp->paw)
				for (const matrix &p : sp->paw->potential)
					basePotential.push_back(p);
		double energy[2], onsite[2], h = 2e-5;
		for (int s = 0; s < 2; s++)
		{
			e.eVars.C = original;
			for (int q = e.eInfo.qStart; q < e.eInfo.qStop; q++)
			{
				e.eVars.C[q] += (s ? h : -h) * direction[q];
				e.eVars.orthonormalize(q);
			}
			energy[s] = e.eVars.elecEnergyAndGrad(e.ener);
			onsite[s] = e.ener.E["EPAW"];
			for (auto sp : e.iInfo.species)
				if (sp->paw)
					for (const matrix &r : sp->paw->rho)
						trialRho[s].push_back(r);
		}
		double rhoDerivative = 0.;
		for (size_t i = 0; i < basePotential.size(); i++)
			rhoDerivative += trace(basePotential[i] * (trialRho[1][i] - trialRho[0][i])).real() / (2 * h);
		close(rhoDerivative, onsiteDerivative, 2e-7, "On-site density / orbital chain rule");
		close((onsite[1] - onsite[0]) / (2 * h), onsiteDerivative, 2e-6,
		      "On-site energy / orbital derivative");
		close((energy[1] - energy[0]) / (2 * h), derivative, 2e-6,
		      "Periodic PAW energy / Hamiltonian derivative");
		e.eVars.C = original;
		for (int q = e.eInfo.qStart; q < e.eInfo.qStop; q++)
		{
			for (auto sp : e.iInfo.species)
				if (sp->isPAW())
				{
					matrix p = (*sp->getV(original[q])) ^ original[q];
					int n = sp->paw->channels.size();
					matrix reconstructed = zeroes(original[q].nCols(), original[q].nCols());
					for (size_t a = 0; a < sp->atpos.size(); a++)
					{
						matrix pa = p(a * n, (a + 1) * n, 0, p.nCols());
						reconstructed += dagger(pa) * (sp->pawOverlap(vector3<>()) * pa);
					}
					ColumnBundle augmented = original[q].similar();
					augmented.zero();
					sp->augmentOverlap(original[q], augmented);
					close(nrm2(reconstructed - (original[q] ^ augmented)), 0., 2e-9,
					      "AE overlap and generalized eigenproblem metric");
					vector3<> dk(.07, -.03, .02);
					close(nrm2(sp->pawOverlap(dk) - dagger(sp->pawOverlap(-dk))), 0., 1e-10,
					      "Cross-k AE overlap adjoint relation");
				}
			for (int k = 0; k < 3; k++)
			{
				matrix p = e.iInfo.allElectronMomentum(original[q], k);
				close(nrm2(p - dagger(p)), 0., 1e-10, "Hermitian full AE momentum");
			}
		}
	}
	finalizeSystem();
}
