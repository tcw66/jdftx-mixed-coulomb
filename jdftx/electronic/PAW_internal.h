/* GPU PAW quadrature interface. GNU GPL v3 or later; see PAW.h. */
#ifndef JDFTX_ELECTRONIC_PAW_INTERNAL_H
#define JDFTX_ELECTRONIC_PAW_INTERNAL_H
#include <core/vector3.h>

// Angular index is fastest in point arrays; radial index is fastest in LM arrays.
struct PAWXCView
{
	int nr, nlm, na;
	const double *inverseR, *radialWeight, *angularWeight, *harmonics;
	const vector3<> *directions, *harmonicGrad;
	const double *density, *densityDerivative;
	double *n, *sigma, *energy, *energyN, *energySigma;
	vector3<> *gradient;
	double *a, *b, *c;
};
#ifdef GPU_ENABLED
double pawXC_gpu(const PAWXCView &);
#endif
#endif
