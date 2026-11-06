/* See PAW.h for license. Hartree atomic units throughout. */
#include <electronic/PAW.h>
#include <electronic/PAW_internal.h>
#include <electronic/ExCorr.h>
#include <core/SphericalHarmonics.h>
#include <libxml/parser.h>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <memory>
#ifdef GPU_ENABLED
// Only density multipoles and integrated derivatives cross the CPU/GPU boundary.
// The much larger angular arrays remain on the device between calls.
struct PAWQuadrature
{
	ManagedArray<double> inverseR, radialWeight, angularWeight, harmonics;
	ManagedArray<vector3<>> directions, harmonicGrad;
	ManagedArray<double> density, densityDerivative, n, sigma, energy, energyN, energySigma, a, b, c;
	ManagedArray<vector3<>> gradient;
	std::vector<double> hostDensity, hostDerivative;
	PAWQuadrature(const RadialFunctionR &grid, const std::vector<double> &weights,
	              const std::vector<vector3<>> &dirs, const std::vector<std::vector<double>> &y,
	              const std::vector<std::vector<vector3<>>> &dy)
	{
		size_t nr = grid.r.size(), nlm = y.size(), na = dirs.size();
		std::vector<double> inv(nr), radial(nr), flatY(nlm * na);
		std::vector<vector3<>> flatDY(nlm * na);
		for (size_t r = 0; r < nr; r++)
		{
			inv[r] = grid.r[r] ? 1. / grid.r[r] : 0.;
			radial[r] = grid.dr[r] * grid.r[r] * grid.r[r];
		}
		for (size_t lm = 0; lm < nlm; lm++)
			for (size_t k = 0; k < na; k++)
			{
				flatY[lm * na + k] = y[lm][k];
				flatDY[lm * na + k] = dy[lm][k];
			}
		inverseR = ManagedArray<double>(inv);
		radialWeight = ManagedArray<double>(radial);
		angularWeight = ManagedArray<double>(weights);
		directions = ManagedArray<vector3<>>(dirs);
		harmonics = ManagedArray<double>(flatY);
		harmonicGrad = ManagedArray<vector3<>>(flatDY);
		size_t points = (nr - 1) * na;
		for (auto *array : {&n, &sigma, &energy, &energyN, &energySigma})
			array->init(points, true);
		gradient.init(points, true);
		for (auto *array : {&density, &densityDerivative, &a, &b, &c})
			array->init(nr * nlm, true);
		hostDensity.resize(nr * nlm);
		hostDerivative.resize(nr * nlm);
	}
	double evaluate(const std::vector<std::vector<double>> &nLM, const std::vector<std::vector<double>> &dnLM,
	                std::vector<std::vector<double>> &outA, std::vector<std::vector<double>> &outB,
	                std::vector<std::vector<double>> &outC)
	{
		int nr = nLM[0].size(), nlm = nLM.size(), na = angularWeight.nData();
		for (int lm = 0; lm < nlm; lm++)
		{
			std::copy(nLM[lm].begin(), nLM[lm].end(), hostDensity.begin() + lm * nr);
			std::copy(dnLM[lm].begin(), dnLM[lm].end(), hostDerivative.begin() + lm * nr);
		}
		cudaMemcpy(density.dataGpu(), hostDensity.data(), hostDensity.size() * sizeof(double),
		           cudaMemcpyHostToDevice);
		cudaMemcpy(densityDerivative.dataGpu(), hostDerivative.data(), hostDerivative.size() * sizeof(double),
		           cudaMemcpyHostToDevice);
		PAWXCView view = {nr,
		                  nlm,
		                  na,
		                  inverseR.dataGpu(),
		                  radialWeight.dataGpu(),
		                  angularWeight.dataGpu(),
		                  harmonics.dataGpu(),
		                  directions.dataGpu(),
		                  harmonicGrad.dataGpu(),
		                  density.dataGpu(),
		                  densityDerivative.dataGpu(),
		                  n.dataGpu(),
		                  sigma.dataGpu(),
		                  energy.dataGpu(),
		                  energyN.dataGpu(),
		                  energySigma.dataGpu(),
		                  gradient.dataGpu(),
		                  a.dataGpu(),
		                  b.dataGpu(),
		                  c.dataGpu()};
		double result = pawXC_gpu(view);
		const double *pa = a.data(), *pb = b.data(), *pc = c.data();
		for (int lm = 0; lm < nlm; lm++)
		{
			std::copy(pa + lm * nr, pa + (lm + 1) * nr, outA[lm].begin());
			std::copy(pb + lm * nr, pb + (lm + 1) * nr, outB[lm].begin());
			std::copy(pc + lm * nr, pc + (lm + 1) * nr, outC[lm].begin());
		}
		return result;
	}
};
#endif

namespace
{
std::string attr(xmlNode *node, const char *key, bool required = true)
{
	xmlChar *value = xmlGetProp(node, BAD_CAST key);
	if (!value)
	{
		if (required)
			throw std::runtime_error("PAW XML: missing attribute '" + std::string(key) + "' in " +
			                         (const char *)node->name);
		return "";
	}
	std::string result((const char *)value);
	xmlFree(value);
	return result;
}
double number(const std::string &s)
{
	char *end = nullptr;
	double x = strtod(s.c_str(), &end);
	while (end && isspace(*end))
		++end;
	if (end == s.c_str() || !end || *end || !std::isfinite(x))
		throw std::runtime_error("PAW XML: invalid number '" + std::string(s) + "'");
	return x;
}
double value(xmlNode *n, const char *key, double fallback = 0., bool required = true)
{
	std::string s = attr(n, key, required);
	return s.empty() ? fallback : number(s);
}
std::vector<double> data(xmlNode *n, size_t count)
{
	xmlChar *content = xmlNodeGetContent(n);
	std::istringstream input((const char *)content);
	xmlFree(content);
	std::vector<double> result;
	std::string s;
	while (input >> s)
		result.push_back(number(s));
	if (result.size() != count)
		throw std::runtime_error("PAW XML: wrong data length in " + std::string((const char *)n->name) +
		                         ": expected " + std::to_string(count) + ", got " +
		                         std::to_string(result.size()));
	return result;
}
xmlNode *child(xmlNode *n, const char *name, bool required = true)
{
	xmlNode *found = nullptr;
	for (xmlNode *c = n->children; c; c = c->next)
		if (c->type == XML_ELEMENT_NODE && xmlStrEqual(c->name, BAD_CAST name))
		{
			if (found)
				throw std::runtime_error("PAW XML: duplicate element " + std::string(name));
			found = c;
		}
	if (!found && required)
		throw std::runtime_error("PAW XML: missing element " + std::string(name));
	return found;
}
double rootBessel(int l, int which)
{
	double left = 0.01, f = bessel_jl(l, left);
	int roots = 0;
	for (double right = left + 0.05; right < 40.; right += 0.05)
	{
		double g = bessel_jl(l, right);
		if (f * g < 0.)
		{
			double a = left, b = right;
			for (int it = 0; it < 60; it++)
			{
				double m = 0.5 * (a + b);
				if (bessel_jl(l, a) * bessel_jl(l, m) > 0.)
					a = m;
				else
					b = m;
			}
			if (++roots == which)
				return 0.5 * (a + b);
		}
		left = right;
		f = g;
	}
	throw std::runtime_error("PAW: failed to find spherical Bessel zero");
}
} // namespace
PAW::PAW(istream &input)
{
	std::ostringstream buffer;
	buffer << input.rdbuf();
	std::string xml = buffer.str();
	uint64_t hash = 14695981039346656037ULL;
	for (unsigned char c : xml)
	{
		hash ^= c;
		hash *= 1099511628211ULL;
	}
	std::ostringstream identity;
	identity << std::hex << hash;
	fingerprint = identity.str();
	std::unique_ptr<xmlDoc, decltype(&xmlFreeDoc)> doc(
	    xmlReadMemory(xml.data(), xml.size(), "PAW.xml", nullptr, XML_PARSE_NONET | XML_PARSE_NOBLANKS),
	    xmlFreeDoc);
	if (!doc)
		throw std::runtime_error("PAW XML: malformed document");
	if (doc->intSubset || doc->extSubset)
		throw std::runtime_error("PAW XML: DTDs are not supported");
	xmlNode *top = xmlDocGetRootElement(doc.get());
	if (!top ||
	    !(xmlStrEqual(top->name, BAD_CAST "paw_dataset") || xmlStrEqual(top->name, BAD_CAST "paw_setup")))
		throw std::runtime_error("PAW XML: expected paw_dataset or paw_setup");
	std::string version = attr(top, "version");
	if (version != "0.7" && version != "0.6")
		throw std::runtime_error("PAW XML: supported versions are 0.6 and 0.7");
	xmlNode *atom = child(top, "atom");
	double z = value(atom, "Z");
	atomicNumber = int(z);
	valence = value(atom, "valence");
	if (z != atomicNumber || atomicNumber < 1 || atomicNumber > 118 || valence <= 0. ||
	    fabs(value(atom, "core") + valence - z) > 1e-8)
		throw std::runtime_error("PAW XML: invalid atomic charges");
	xmlNode *xc = child(top, "xc_functional");
	if (attr(xc, "type") != "GGA" || attr(xc, "name") != "PBE")
		throw std::runtime_error("PAW XML: only PBE datasets are implemented");
	std::string relativity = attr(child(top, "generator"), "type");
	if (relativity != "scalar-relativistic" && relativity != "non-relativistic")
		throw std::runtime_error("PAW XML: spin-orbit datasets are not implemented");
	coreKinetic = value(child(top, "core_energy"), "kinetic");
	xmlNode *radial = child(top, "radial_grid");
	std::string gridId = attr(radial, "id");
	int start = int(value(radial, "istart")), stop = int(value(radial, "iend"));
	if (start < 0 || stop <= start || stop - start > 100000)
		throw std::runtime_error("PAW XML: invalid radial grid size");
	const size_t nr = stop - start + 1;
	grid = RadialFunctionR(nr);
	if (xmlNode *values = child(radial, "values", false))
		grid.r = data(values, nr);
	else
	{
		std::string eq = attr(radial, "eq");
		double a = value(radial, "a", 0., false), d = value(radial, "d", 0., false),
		       n = value(radial, "n", 0., false);
		for (size_t j = 0; j < nr; j++)
		{
			double i = start + j;
			if (eq == "r=a*(exp(d*i)-1)")
				grid.r[j] = a * expm1(d * i);
			else if (eq == "r=a*exp(d*i)")
				grid.r[j] = a * exp(d * i);
			else if (eq == "r=d*i")
				grid.r[j] = d * i;
			else if (eq == "r=a*i/(n-i)" && n > i)
				grid.r[j] = a * i / (n - i);
			else
				throw std::runtime_error("PAW XML: unsupported radial grid equation");
		}
	}
	for (size_t j = 0; j < nr; j++)
		if (!std::isfinite(grid.r[j]) || grid.r[j] < 0. || (j && grid.r[j] <= grid.r[j - 1]))
			throw std::runtime_error("PAW XML: radial grid must be finite and strictly increasing");
	grid.initWeights();
	auto radialData = [&](xmlNode *node)
	{
		if (attr(node, "grid") != gridId)
			throw std::runtime_error("PAW XML: inconsistent radial grid reference");
		return data(node, nr);
	};
	coreAE = radialData(child(top, "ae_core_density"));
	corePS = radialData(child(top, "pseudo_core_density"));
	zeroPotential = radialData(child(top, "zero_potential"));
	const double y00 = 1. / sqrt(4 * M_PI);
	for (size_t j = 0; j < nr; j++)
	{
		coreAE[j] *= y00;
		corePS[j] *= y00;
		zeroPotential[j] *= y00;
	}
	xmlNode *valenceStates = child(top, "valence_states");
	std::map<std::string, int> stateIndex;
	std::map<int, int> countL;
	radius = 0.;
	if (xmlNode *pr = child(top, "paw_radius", false))
		radius = value(pr, "rc");
	for (xmlNode *s = valenceStates->children; s; s = s->next)
		if (s->type == XML_ELEMENT_NODE)
		{
			if (!xmlStrEqual(s->name, BAD_CAST "state"))
				throw std::runtime_error("PAW XML: unknown valence-state element");
			State state;
			state.id = attr(s, "id");
			state.l = int(value(s, "l"));
			state.n = int(value(s, "n", 0., false));
			state.occupation = value(s, "f", 0., false);
			state.energy = value(s, "e");
			if (state.l < 0 || state.l > 3)
				throw std::runtime_error("PAW XML: partial waves require 0 <= l <= 3");
			if (stateIndex.count(state.id))
				throw std::runtime_error("PAW XML: duplicate state ID");
			state.p = countL[state.l]++;
			lMax = std::max(lMax, state.l);
			radius = std::max(radius, value(s, "rc"));
			stateIndex[state.id] = states.size();
			states.push_back(state);
		}
	if (states.empty() || radius <= 0. || radius >= grid.r.back())
		throw std::runtime_error("PAW XML: invalid augmentation sphere");
	for (xmlNode *node = top->children; node; node = node->next)
		if (node->type == XML_ELEMENT_NODE)
		{
			std::string tag((const char *)node->name);
			if (tag == "ae_partial_wave" || tag == "pseudo_partial_wave" || tag == "projector_function")
			{
				std::string id = attr(node, "state");
				if (!stateIndex.count(id))
					throw std::runtime_error("PAW XML: unknown state reference");
				State &s = states[stateIndex[id]];
				std::vector<double> &target =
				    tag == "ae_partial_wave" ? s.ae : (tag == "pseudo_partial_wave" ? s.ps : s.projector);
				if (!target.empty())
					throw std::runtime_error("PAW XML: duplicate radial wave or projector");
				target = radialData(node);
			}
		}
	for (const State &s : states)
		if (s.ae.empty() || s.ps.empty() || s.projector.empty())
			throw std::runtime_error("PAW XML: missing partial wave or projector");
	// Check the fundamental PAW biorthogonality before using a dataset.
	double dualError = 0.;
	for (size_t i = 0; i < states.size(); i++)
		for (size_t j = 0; j < states.size(); j++)
			if (states[i].l == states[j].l)
			{
				std::vector<double> prod(nr);
				for (size_t r = 0; r < nr; r++)
					prod[r] = states[i].projector[r] * states[j].ps[r];
				dualError = std::max(dualError, fabs(moment(prod) - (i == j ? 1. : 0.)));
			}
	if (dualError > 5e-2)
		throw std::runtime_error("PAW XML: projector/partial-wave duality error exceeds 0.05");
	// Restore biorthogonality on the integration mesh used by this code. All
	// partial waves and PAW operator matrices retain the dataset convention;
	// only the dual projectors change to compensate for radial quadrature error.
	for (int l = 0; l <= lMax; l++)
	{
		std::vector<size_t> indices;
		for (size_t i = 0; i < states.size(); i++)
			if (states[i].l == l)
				indices.push_back(i);
		if (indices.empty())
			continue;
		matrix dual = zeroes(indices.size(), indices.size());
		for (size_t i = 0; i < indices.size(); i++)
			for (size_t j = 0; j < indices.size(); j++)
			{
				std::vector<double> prod(nr);
				for (size_t r = 0; r < nr; r++)
					prod[r] = states[indices[i]].projector[r] * states[indices[j]].ps[r];
				dual.set(i, j, moment(prod));
			}
		if (fabs(det(dual).real()) < 1e-8)
			throw std::runtime_error("PAW XML: singular projector duality matrix");
		matrix inverse = inv(dual);
		std::vector<std::vector<double>> corrected(indices.size(), std::vector<double>(nr));
		for (size_t i = 0; i < indices.size(); i++)
			for (size_t j = 0; j < indices.size(); j++)
				for (size_t r = 0; r < nr; r++)
					corrected[i][r] += inverse(i, j).real() * states[indices[j]].projector[r];
		for (size_t i = 0; i < indices.size(); i++)
			states[indices[i]].projector = std::move(corrected[i]);
	}
	logPrintf("  PAW XML %s: Z=%d, valence=%g, %zu partial waves, input duality error %.3g (corrected on "
	          "radial mesh)\n",
	          version.c_str(), atomicNumber, valence, states.size(), dualError);
	auto ked = data(child(top, "kinetic_energy_differences"), states.size() * states.size());
	kinetic = zeroes(states.size(), states.size());
	for (size_t i = 0; i < states.size(); i++)
		for (size_t j = 0; j < states.size(); j++)
		{
			if (fabs(ked[i * states.size() + j] - ked[j * states.size() + i]) > 1e-8)
				throw std::runtime_error("PAW XML: nonsymmetric kinetic matrix");
			kinetic.set(i, j, ked[i * states.size() + j]);
		}
	for (int l = 0; l <= lMax; l++)
		for (size_t s = 0; s < states.size(); s++)
			if (states[s].l == l)
				for (int m = -l; m <= l; m++)
					channels.push_back({int(s), l, m});
	// Bessel compensation functions: unit multipole moment, zero value and slope at rc.
	xmlNode *sh = child(top, "shape_function");
	std::string shapeType = attr(sh, "type");
	double rc = value(sh, "rc");
	if (rc <= 0. || rc > radius + 1e-8)
		throw std::runtime_error("PAW XML: invalid shape radius");
	shape.resize(2 * lMax + 1, std::vector<double>(nr));
	for (int l = 0; l <= 2 * lMax; l++)
	{
		double q1 = rootBessel(l, 1), q2 = rootBessel(l, 2);
		double ratio = l ? -q1 * bessel_jl(l - 1, q1) / (q2 * bessel_jl(l - 1, q2)) : -cos(q1) / cos(q2);
		for (size_t r = 0; r < nr; r++)
			if (grid.r[r] < rc)
			{
				double x = grid.r[r] / rc;
				if (shapeType == "bessel")
					shape[l][r] = bessel_jl(l, q1 * x) + ratio * bessel_jl(l, q2 * x);
				else if (shapeType == "sinc")
					shape[l][r] = pow(grid.r[r], l) * pow(bessel_jl(0, M_PI * x), 2);
				else
					throw std::runtime_error(
					    "PAW XML: unsupported compensation shape (expected bessel or sinc)");
			}
		double norm = moment(shape[l], l);
		if (fabs(norm) < 1e-15)
			throw std::runtime_error("PAW XML: vanishing compensation moment");
		for (double &x : shape[l])
			x /= norm;
	}
	// Fixed smooth ionic charge rho0 = ncorePS + (Ncore-Z-NcorePS) g0/(4 pi).
	std::vector<double> rho0(nr);
	double q0 = -valence - 4 * M_PI * moment(corePS);
	for (size_t r = 0; r < nr; r++)
		rho0[r] = corePS[r] + q0 * shape[0][r] / (4 * M_PI);
	ionicPotential = poisson(rho0, 0);
	for (size_t r = 0; r < nr; r++)
		ionicSelfEnergy += 0.5 * 4 * M_PI * grid.dr[r] * pow(grid.r[r], 2) * rho0[r] * ionicPotential[r];
	for (size_t r = 0; r < nr; r++)
		ionicPotential[r] += zeroPotential[r];
	// Pair products in real harmonics. Compensation retains every allowed multipole.
	for (size_t i = 0; i < channels.size(); i++)
		for (size_t j = 0; j <= i; j++)
		{
			const Channel &a = channels[i];
			const Channel &b = channels[j];
			std::vector<double> diff(nr);
			for (size_t r = 0; r < nr; r++)
				if (grid.r[r] <= radius + 1e-8)
					diff[r] = states[a.state].ae[r] * states[b.state].ae[r] -
					          states[a.state].ps[r] * states[b.state].ps[r];
			for (const auto &term : expandYlmProd(a.l, a.m, b.l, b.m))
			{
				Pair pair;
				pair.i = i;
				pair.j = j;
				pair.lm = term.l * (term.l + 1) + term.m;
				pair.gaunt = term.coeff;
				pair.ae.resize(nr);
				pair.ps.resize(nr);
				pair.hat.resize(nr);
				double q = moment(diff, term.l);
				for (size_t r = 0; r < nr; r++)
				{
					bool inside = grid.r[r] <= radius + 1e-8;
					// Continue both partial densities identically outside the sphere, with a
					// common smooth taper. Their energy difference is unchanged (all charge
					// difference multipoles vanish), and no artificial boundary enters GGA.
					double taper = exp(-pow(std::max(0., grid.r[r] - radius) / (0.3 * radius), 4));
					double common = states[a.state].ps[r] * states[b.state].ps[r] * taper;
					pair.ae[r] = inside ? states[a.state].ae[r] * states[b.state].ae[r] : common;
					pair.hat[r] = q * shape[term.l][r];
					pair.ps[r] =
					    (inside ? states[a.state].ps[r] * states[b.state].ps[r] : common) + pair.hat[r];
				}
				pairs.push_back(std::move(pair));
			}
		}
	// Tensor Gauss-Legendre x uniform phi quadrature; nonlinear XC uses extra angles.
	const int nt = 16 + 2 * lMax, np = 2 * nt;
	for (int t = 0; t < nt; t++)
	{
		double z = cos(M_PI * (t + 0.75) / (nt + 0.5)), pp = 0.;
		for (int it = 0; it < 30; it++)
		{
			double p0 = 1., p1 = z;
			for (int l = 2; l <= nt; l++)
			{
				double p = ((2 * l - 1) * z * p1 - (l - 1) * p0) / l;
				p0 = p1;
				p1 = p;
			}
			pp = nt * (z * p1 - p0) / (z * z - 1.);
			double dz = p1 / pp;
			z -= dz;
			if (fabs(dz) < 1e-15)
				break;
		}
		double weight = 2. / ((1 - z * z) * pp * pp) * 2 * M_PI / np;
		for (int p = 0; p < np; p++)
		{
			double phi = 2 * M_PI * p / np;
			directions.push_back(vector3<>(sqrt(1 - z * z) * cos(phi), sqrt(1 - z * z) * sin(phi), z));
			angularWeights.push_back(weight);
		}
	}
	int nlm = pow(2 * lMax + 1, 2);
	harmonics.resize(nlm);
	harmonicGrad.resize(nlm);
	for (int l = 0; l <= 2 * lMax; l++)
		for (int m = -l; m <= l; m++)
		{
			int lm = l * (l + 1) + m;
			for (auto dir : directions)
			{
				double y = Ylm(l, m, dir);
				harmonics[lm].push_back(y);
				harmonicGrad[lm].push_back(YlmPrime(l, m, dir) - double(l) * y * dir);
			}
		}
}
PAW::~PAW() {}
double PAW::moment(const std::vector<double> &f, int l) const
{
	double result = 0.;
	for (size_t r = 0; r < f.size(); r++)
		result += grid.dr[r] * pow(grid.r[r], l + 2) * f[r];
	return result;
}
std::vector<double> PAW::derivative(const std::vector<double> &f) const
{
	std::vector<double> result(f.size());
	for (size_t i = 0; i < f.size(); i++)
	{
		size_t j = std::max(size_t(1), std::min(i, f.size() - 2));
		double x = grid.r[i], a = grid.r[j - 1], b = grid.r[j], c = grid.r[j + 1];
		result[i] = (f[j - 1] - f[j]) * (2 * x - b - c) / ((a - b) * (a - c)) +
		            (f[j + 1] - f[j]) * (2 * x - a - b) / ((c - a) * (c - b));
	}
	return result;
}
std::vector<double> PAW::poisson(const std::vector<double> &f, int l) const
{
	size_t n = f.size();
	std::vector<double> result(n), outer(n);
	double sum = 0., inner = 0.;
	// Build the outer integral backwards. Subtracting it from a total loses
	// precision near the origin for high multipoles, especially on log grids.
	for (size_t r = n; r-- > 0;)
	{
		outer[r] = sum;
		if (grid.r[r])
			sum += grid.dr[r] * pow(grid.r[r], 1 - l) * f[r];
	}
	for (size_t r = 0; r < n; r++)
	{
		double x = grid.r[r];
		if (!x)
		{
			result[r] = l ? 0. : 4 * M_PI * outer[r];
			continue;
		}
		inner += grid.dr[r] * pow(x, l + 2) * f[r];
		result[r] = 4 * M_PI / (2 * l + 1) * (inner * pow(x, -l - 1) + outer[r] * pow(x, l));
	}
	return result;
}
matrix PAW::phaseMatrix(const matrix &physical, bool inverse) const
{
	matrix result = physical;
	complex *out = result.data();
	for (size_t i = 0; i < channels.size(); i++)
		for (size_t j = 0; j < channels.size(); j++)
			out[result.index(i, j)] *=
			    cis((inverse ? -1. : 1.) * 0.5 * M_PI * (channels[j].l - channels[i].l));
	return result;
}

double PAW::energyAndPotential(const matrix &density, const ExCorr &xc, matrix &grad) const
{
	if (!xc.supportsPAW())
		die("PAW one-center XC requires internal PBE.\n");
	const size_t nr = grid.r.size(), nlm = harmonics.size();
	const double sqrt4pi = sqrt(4 * M_PI);
	typedef std::vector<double> Vec;
	typedef std::vector<Vec> Multipoles;
	Multipoles ae(nlm, Vec(nr)), ps(nlm, Vec(nr)), coulAE, coulPS;
	for (const Pair &pair : pairs)
	{
		double coeff = pair.gaunt * density(pair.i, pair.j).real() * (pair.i == pair.j ? 1. : 2.);
		for (size_t r = 0; r < nr; r++)
		{
			ae[pair.lm][r] += coeff * pair.ae[r];
			ps[pair.lm][r] += coeff * pair.ps[r];
		}
	}
	long double energy = coreKinetic + ionicSelfEnergy;
	// Local zero potential: its smooth-grid contribution is subtracted on site.
	for (size_t r = 0; r < nr; r++)
		energy -= grid.dr[r] * pow(grid.r[r], 2) * sqrt4pi * zeroPotential[r] * ps[0][r];
	double q0 = -valence - 4 * M_PI * moment(corePS);
	coulAE = ae;
	coulPS = ps;
	for (size_t r = 0; r < nr; r++)
	{
		coulAE[0][r] += sqrt4pi * coreAE[r];
		coulPS[0][r] += sqrt4pi * (corePS[r] + q0 * shape[0][r] / (4 * M_PI));
		ae[0][r] += sqrt4pi * coreAE[r];
		ps[0][r] += sqrt4pi * corePS[r];
	}
	Multipoles vAE(nlm), vPS(nlm);
	for (size_t lm = 0; lm < nlm; lm++)
	{
		int l = int(sqrt(double(lm)));
		vAE[lm] = poisson(coulAE[lm], l);
		vPS[lm] = poisson(coulPS[lm], l);

		for (size_t r = 0; r < nr; r++)
		{
			double w = grid.dr[r] * pow(grid.r[r], 2);
			energy += 0.5 * w * (coulAE[lm][r] * vAE[lm][r] - coulPS[lm][r] * vPS[lm][r]);
			if (!lm && grid.r[r])
				energy -= w * atomicNumber * sqrt4pi * coulAE[0][r] / grid.r[r];
		}
	}
	grad = zeroes(channels.size(), channels.size());
	auto addPairGrad = [&](const Pair &pair, double x)
	{
		x *= pair.gaunt;
		grad.data()[grad.index(pair.i, pair.j)] += x;
		if (pair.i != pair.j)
			grad.data()[grad.index(pair.j, pair.i)] += x;
	};
	for (const Pair &pair : pairs)
	{
		double x = 0.;
		for (size_t r = 0; r < nr; r++)
		{
			double va = vAE[pair.lm][r], vp = vPS[pair.lm][r];
			if (!pair.lm)
			{
				if (grid.r[r])
					va -= atomicNumber * sqrt4pi / grid.r[r];
				vp += sqrt4pi * zeroPotential[r];
			}
			x += grid.dr[r] * pow(grid.r[r], 2) * (va * pair.ae[r] - vp * pair.ps[r]);
		}
		addPairGrad(pair, x);
	}
	// Nonlinear XC is evaluated on an atom-centered radial/angular quadrature.
	// Differentiate the discretized energy itself, including radial and tangential
	// GGA gradients; this avoids an inconsistent finite-difference divergence.
	for (int side = 0; side < 2; side++)
	{
		const Multipoles &nLM = side ? ps : ae;
		double sign = side ? -1. : 1.;
		Multipoles dnLM(nlm);
		for (size_t lm = 0; lm < nlm; lm++)
			dnLM[lm] = derivative(nLM[lm]);
		Multipoles a(nlm, Vec(nr)), b = a, c = a;
#ifdef GPU_ENABLED
		if (!quadrature)
			quadrature =
			    std::make_shared<PAWQuadrature>(grid, angularWeights, directions, harmonics, harmonicGrad);
		energy += sign * quadrature->evaluate(nLM, dnLM, a, b, c);
#else
		// Batch radial shells so PBE launches a few large threaded loops rather
		// than thousands of tiny ones. Bound the host workspace for heavy atoms.
		const size_t batchRows = 64, na = directions.size();
		for (size_t begin = 1; begin < nr; begin += batchRows)
		{
			size_t rows = std::min(batchRows, nr - begin), count = rows * na;
			Vec n(count), sigma(count), exc(count), en(count), es(count);
			std::vector<vector3<>> gn(count);
			for (size_t row = 0; row < rows; row++)
			{
				size_t r = begin + row, offset = row * na;
				for (size_t lm = 0; lm < nlm; lm++)
					for (size_t k = 0; k < na; k++)
					{
						n[offset + k] += nLM[lm][r] * harmonics[lm][k];
						gn[offset + k] += dnLM[lm][r] * harmonics[lm][k] * directions[k] +
						                  (nLM[lm][r] / grid.r[r]) * harmonicGrad[lm][k];
					}
			}
			for (size_t k = 0; k < count; k++)
				sigma[k] = gn[k].length_squared();
			xc.evaluatePAW(count, n.data(), sigma.data(), exc.data(), en.data(), es.data());
			for (size_t row = 0; row < rows; row++)
			{
				size_t r = begin + row, offset = row * na;
				for (size_t k = 0; k < na; k++)
					energy += sign * grid.dr[r] * pow(grid.r[r], 2) * angularWeights[k] * exc[offset + k];
				for (size_t lm = 0; lm < nlm; lm++)
					for (size_t k = 0; k < na; k++)
					{
						size_t index = offset + k;
						double w = angularWeights[k];
						a[lm][r] += w * en[index] * harmonics[lm][k];
						b[lm][r] += w * 2 * es[index] * dot(gn[index], directions[k]) * harmonics[lm][k];
						c[lm][r] += w * 2 * es[index] * dot(gn[index], harmonicGrad[lm][k]) / grid.r[r];
					}
			}
		}
#endif
		for (const Pair &pair : pairs)
		{
			const Vec &f = side ? pair.ps : pair.ae;
			Vec df = derivative(f);
			double x = 0.;
			for (size_t r = 1; r < nr; r++)
				x += grid.dr[r] * pow(grid.r[r], 2) *
				     ((a[pair.lm][r] + c[pair.lm][r]) * f[r] + b[pair.lm][r] * df[r]);
			addPairGrad(pair, sign * x);
		}
	}
	return energy;
}

vector3<matrix> PAW::momentumCorrection() const
{
	if (cachedMomentum[0])
		return cachedMomentum;
	const int n = channels.size();
	vector3<matrix> result;
	for (int k = 0; k < 3; k++)
		result[k] = zeroes(n, n);
	// <phi_i|-i grad|phi_j> - <phitilde_i|-i grad|phitilde_j>.
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			const Channel &ci = channels[i];
			const Channel &cj = channels[j];
			if (abs(ci.l - cj.l) != 1)
				continue;
			const State &si = states[ci.state];
			const State &sj = states[cj.state];
			vector3<> ar, at;
			for (size_t a = 0; a < directions.size(); a++)
			{
				double yi = Ylm(ci.l, ci.m, directions[a]), yj = Ylm(cj.l, cj.m, directions[a]);
				ar += angularWeights[a] * yi * yj * directions[a];
				at += angularWeights[a] * yi *
				      (YlmPrime(cj.l, cj.m, directions[a]) - double(cj.l) * yj * directions[a]);
			}
			auto dae = derivative(sj.ae), dps = derivative(sj.ps);
			vector3<> sum;
			for (size_t r = 1; r < grid.r.size(); r++)
				if (grid.r[r] <= radius + 1e-8)
				{
					double x = grid.r[r], w = grid.dr[r] * x * x;
					sum += w * ((si.ae[r] * dae[r] - si.ps[r] * dps[r]) * ar +
					            (si.ae[r] * sj.ae[r] - si.ps[r] * sj.ps[r]) / x * at);
				}
			for (int k = 0; k < 3; k++)
				result[k].set(i, j, complex(0., -sum[k]));
		}
	for (int k = 0; k < 3; k++)
		result[k] = 0.5 * (result[k] + dagger(result[k]));
	cachedMomentum = result;
	return result;
}
