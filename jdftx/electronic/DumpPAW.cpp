/* PAW valence wavefunction and operator export. GNU GPL v3 or later. */
#include <electronic/Dump.h>
#include <electronic/Everything.h>
#include <electronic/ColumnBundle.h>
#include <core/ScalarFieldIO.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <array>
#include <cstdint>

void Dump::dumpPAW(bool wavefunctions, bool matrices) const
{
	bool hasPAW = false;
	for (auto sp : e->iInfo.species)
		hasPAW |= sp->isPAW();
	if (!hasPAW)
		die("AEWfns and AEMatrices require at least one PAW XML species.\n");
	for (auto sp : e->iInfo.species)
		if (!sp->atpos.empty() && !sp->isPAW())
			die("All-electron exports require PAW datasets for every species.\n");
	const auto &info = e->eInfo;
	const auto &vars = e->eVars;
	double cutoff = pawReconstructionCutoff ? pawReconstructionCutoff : 4 * e->cntrl.Ecut;
	if (cutoff < e->cntrl.Ecut)
		die("paw-reconstruction-cutoff must be at least elec-cutoff.\n");
	string prefix = getFilename("paw");
	// Eigenvalues are invariant under the PAW transformation.
	info.write(vars.Hsub_eigs, (prefix + ".eigenvals").c_str());
	if (mpiWorld->isHead())
	{
		std::ofstream meta((prefix + ".json").c_str());
		if (!meta)
			die_alone("Cannot write PAW export metadata.\n");
		uint16_t endian = 1;
		const char *byteOrder = (*reinterpret_cast<unsigned char *>(&endian)) ? "little" : "big";
		meta << std::setprecision(17)
		     << "{\n  \"format\": \"JDFTx-PAW-1\",\n  \"units\": \"Hartree atomic units\",\n  \"states\": "
		     << info.nStates << ",\n  \"byte_order\": \"" << byteOrder << "\",\n  \"bands\": " << info.nBands
		     << ",\n  \"volume\": " << e->gInfo.detR << ",\n  \"reconstruction_cutoff\": " << cutoff << ",\n"
		     << "  \"wavefunctions\": \"q-indexed complex128 files, band outer / G inner; periodic part "
		        "u_nk; psi=exp(i k.r) u; no renormalization\",\n"
		     << "  \"wavefunction_normalization\": \"integral_cell |psi|^2 = volume * sum_G |c_G|^2; finite "
		        "AE cutoff loses norm\",\n"
		     << "  \"matrix_layout\": \"complex128: state, operator [overlap,H,px,py,pz], ket band, bra band "
		        "(fastest)\",\n"
		     << "  \"eigenvalue_layout\": \"float64: state, band (fastest); identical to BandEigs\",\n"
		     << "  \"scope\": \"frozen-core valence states; canonical momentum differs from velocity for "
		        "scalar-relativistic datasets\",\n"
		     << "  \"lattice_columns_bohr\": [";
		for (int j = 0; j < 3; j++)
		{
			if (j)
				meta << ",";
			meta << "[";
			for (int i = 0; i < 3; i++)
			{
				if (i)
					meta << ",";
				meta << e->gInfo.R(i, j);
			}
			meta << "]";
		}
		meta << "],\n  \"kpoints_reciprocal\": [";
		for (int q = 0; q < info.nStates; q++)
		{
			if (q)
				meta << ",";
			meta << "[" << info.qnums[q].k[0] << "," << info.qnums[q].k[1] << "," << info.qnums[q].k[2]
			     << "]";
		}
		meta << "]\n}\n";
	}
	if (matrices)
	{
		std::vector<matrix> output(info.nStates);
		for (int q = info.qStart; q < info.qStop; q++)
		{
			int n = info.nBands;
			output[q] = zeroes(n, 5 * n);
			output[q].set(0, n, 0, n, vars.C[q] ^ O(vars.C[q]));
			output[q].set(0, n, n, 2 * n, vars.Hsub_eigs[q]);
			for (int k = 0; k < 3; k++)
				output[q].set(0, n, (2 + k) * n, (3 + k) * n, e->iInfo.allElectronMomentum(vars.C[q], k));
		}
		info.write(output, (prefix + ".matrices").c_str(), info.nBands, 5 * info.nBands);
	}
	if (wavefunctions)
	{
		GridInfo grid;
		grid.R = e->gInfo.R;
		grid.Gmax = sqrt(2 * cutoff);
		grid.initialize(true, e->symm.getMatrices());
		for (int q = info.qStart; q < info.qStop; q++)
		{
			const ColumnBundle &c = vars.C[q];
			Basis basis;
			basis.setup(grid, e->iInfo, cutoff, c.qnum->k);
			ColumnBundle ae(c.nCols(), basis.nbasis, &basis, c.qnum, isGpuEnabled());
			ae.zero();
			std::map<std::array<int, 3>, size_t> index;
			for (size_t g = 0; g < basis.nbasis; g++)
			{
				auto v = basis.iGarr.data()[g];
				index[{{v[0], v[1], v[2]}}] = g;
			}
			const complex *src = c.data();
			complex *dst = ae.data();
			for (size_t g = 0; g < c.basis->nbasis; g++)
			{
				auto v = c.basis->iGarr.data()[g];
				auto entry = index.find({{v[0], v[1], v[2]}});
				if (entry == index.end())
					die("PAW output basis does not contain the smooth wavefunction basis.\n");
				for (int b = 0; b < c.nCols(); b++)
					dst[b * basis.nbasis + entry->second] = src[b * c.basis->nbasis + g];
			}
			for (auto sp : e->iInfo.species)
				ae += sp->pawReconstruct(c, basis);
			char suffix[80];
			snprintf(suffix, sizeof(suffix), ".q%d", q);
			string path = prefix + suffix;
			FILE *fp = fopen((path + ".wfns").c_str(), "wb");
			if (!fp)
				die_alone("Cannot write PAW wavefunction export.\n");
			if (fwrite(ae.data(), sizeof(complex), ae.nData(), fp) != ae.nData())
				die_alone("Failed writing PAW wavefunction export.\n");
			fclose(fp);
			std::ofstream gv((path + ".Gvectors").c_str());
			if (!gv)
				die_alone("Cannot write PAW G vectors.\n");
			gv << "# G lattice integers; same order as wavefunction coefficients\n";
			for (size_t g = 0; g < basis.nbasis; g++)
			{
				auto v = basis.iGarr.data()[g];
				gv << v[0] << " " << v[1] << " " << v[2] << "\n";
			}
			logPrintf("PAW export state %d: %zu G vectors at %g Hartree -> %s.wfns\n", q, basis.nbasis,
			          cutoff, path.c_str());
		}
	}
}
