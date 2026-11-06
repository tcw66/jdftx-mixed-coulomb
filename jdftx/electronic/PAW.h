/* Native PAW-XML datasets and frozen-core one-center functionals.
   Distributed under the GNU GPL v3 or later, as part of JDFTx. */
#ifndef JDFTX_ELECTRONIC_PAW_H
#define JDFTX_ELECTRONIC_PAW_H
#include <core/RadialFunction.h>
#include <core/matrix.h>
#include <core/string.h>
class ExCorr;

struct PAW
{
	struct State
	{
		std::string id;
		int l, p, n;
		double occupation, energy;
		std::vector<double> ae, ps, projector;
	};
	struct Channel
	{
		int state, l, m;
	};
	struct Pair
	{
		int i, j, lm;
		double gaunt;
		std::vector<double> ae, ps, hat;
	};
	RadialFunctionR grid;
	std::vector<State> states;
	std::vector<Channel> channels;
	std::vector<Pair> pairs;
	std::vector<std::vector<double>> shape;
	std::vector<double> coreAE, corePS, zeroPotential, ionicPotential;
	matrix kinetic;
	std::string fingerprint; //!< dataset identity for restart validation
	int atomicNumber = 0, lMax = 0;
	mutable vector3<matrix> cachedMomentum;
	double valence = 0., radius = 0., coreKinetic = 0., ionicSelfEnergy = 0.;
	std::vector<matrix> rho, potential; // physical real-harmonic projector basis; one per atom
	explicit PAW(istream &);
	~PAW();
	double moment(const std::vector<double> &f, int l = 0) const;
	std::vector<double> poisson(const std::vector<double> &f, int l) const;
	std::vector<double> derivative(const std::vector<double> &f) const;
	double energyAndPotential(const matrix &density, const ExCorr &, matrix &potential) const;
	matrix phaseMatrix(const matrix &physical, bool inverse = false) const;
	vector3<matrix> momentumCorrection() const;

  private:
	std::vector<vector3<>> directions;
	std::vector<double> angularWeights;
	std::vector<std::vector<double>> harmonics;
	std::vector<std::vector<vector3<>>> harmonicGrad;
};
#endif
