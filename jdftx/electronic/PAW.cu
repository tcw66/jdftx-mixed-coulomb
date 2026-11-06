/* Batched PAW angular PBE quadrature. GNU GPL v3 or later; see PAW.h. */
#include <core/GpuKernelUtils.h>
#include <core/BlasExtra.h>
#include <electronic/PAW_internal.h>
#include <electronic/ExCorr_internal_GGA.h>

void GGA_gpu(GGA_Variant, int, std::vector<const double *>, std::vector<const double *>, double *,
             std::vector<double *>, std::vector<double *>, double);

__global__ void pawXCForward_kernel(int count, PAWXCView q)
{
	int i = kernelIndex1D();
	if (i >= count)
		return;
	int r = 1 + i / q.na, k = i % q.na;
	double n = 0.;
	vector3<> gradient;
	for (int lm = 0; lm < q.nlm; lm++)
	{
		double y = q.harmonics[lm * q.na + k], density = q.density[lm * q.nr + r];
		n += density * y;
		gradient += (q.densityDerivative[lm * q.nr + r] * y) * q.directions[k] +
		            (density * q.inverseR[r]) * q.harmonicGrad[lm * q.na + k];
	}
	q.n[i] = n;
	q.gradient[i] = gradient;
	q.sigma[i] = gradient.length_squared();
}

// A block integrates one radial point and harmonic. Angular reads are coalesced,
// and each block reduces locally instead of using floating-point atomics.
__global__ void pawXCProject_kernel(PAWXCView q)
{
	const int threads = 128;
	int lm = blockIdx.x % q.nlm, r = 1 + blockIdx.x / q.nlm, t = threadIdx.x;
	double a = 0., b = 0., c = 0.;
	for (int k = t; k < q.na; k += threads)
	{
		int i = (r - 1) * q.na + k;
		double weight = q.angularWeight[k], y = q.harmonics[lm * q.na + k];
		vector3<> gradient = q.gradient[i];
		a += weight * q.energyN[i] * y;
		b += weight * 2 * q.energySigma[i] * dot(gradient, q.directions[k]) * y;
		c += weight * 2 * q.energySigma[i] * dot(gradient, q.harmonicGrad[lm * q.na + k]) * q.inverseR[r];
	}
	__shared__ double partial[3][threads];
	partial[0][t] = a;
	partial[1][t] = b;
	partial[2][t] = c;
	__syncthreads();
	for (int stride = threads / 2; stride; stride /= 2)
	{
		if (t < stride)
			for (int v = 0; v < 3; v++)
				partial[v][t] += partial[v][t + stride];
		__syncthreads();
	}
	if (!t)
	{
		q.a[lm * q.nr + r] = partial[0][0];
		q.b[lm * q.nr + r] = partial[1][0];
		q.c[lm * q.nr + r] = partial[2][0];
	}
}

__global__ void pawXCEnergy_kernel(int count, PAWXCView q)
{
	int i = kernelIndex1D();
	if (i < count)
		q.energy[i] *= q.radialWeight[1 + i / q.na] * q.angularWeight[i % q.na];
}

double pawXC_gpu(const PAWXCView &q)
{
	int count = (q.nr - 1) * q.na;
	GpuLaunchConfig1D forward(pawXCForward_kernel, count);
	pawXCForward_kernel<<<forward.nBlocks, forward.nPerBlock>>>(count, q);
	gpuErrorCheck();
	eblas_zero_gpu(count, q.energy);
	eblas_zero_gpu(count, q.energyN);
	eblas_zero_gpu(count, q.energySigma);
	for (auto variant : {GGA_X_PBE, GGA_C_PBE})
		GGA_gpu(variant, count, {q.n}, {q.sigma}, q.energy, {q.energyN}, {q.energySigma}, 1.);
	eblas_zero_gpu(q.nr * q.nlm, q.a);
	eblas_zero_gpu(q.nr * q.nlm, q.b);
	eblas_zero_gpu(q.nr * q.nlm, q.c);
	pawXCProject_kernel<<<(q.nr - 1) * q.nlm, 128>>>(q);
	gpuErrorCheck();
	GpuLaunchConfig1D energy(pawXCEnergy_kernel, count);
	pawXCEnergy_kernel<<<energy.nBlocks, energy.nPerBlock>>>(count, q);
	gpuErrorCheck();
	return eblas_sum_gpu(count, q.energy);
}
