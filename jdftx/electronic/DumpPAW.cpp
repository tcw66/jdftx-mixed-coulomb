/* PAW valence wavefunction and operator export. GNU GPL v3 or later. */
#include <electronic/Dump.h>
#include <electronic/Everything.h>
#include <electronic/ColumnBundle.h>
#include <core/ScalarFieldIO.h>
#include <fstream>
#include <iomanip>

namespace
{
// Same text blocks as the native Gvectors dump. Only the owning rank builds
// each basis; gather these small index arrays, never the reconstructed orbitals.
void writeGvectors(const ElecInfo &info, const std::vector<Basis> &bases, const std::vector<size_t> &sizes,
                   const string &filename)
{
	if (mpiWorld->isHead())
	{
		FILE *fp = fopen(filename.c_str(), "w");
		if (!fp)
			die_alone("Cannot write PAW G vectors.\n");
		for (int q = 0; q < info.nStates; q++)
		{
			ManagedArray<vector3<int>> buffer;
			const vector3<int> *vectors = nullptr;
			if (info.isMine(q))
				vectors = bases[q].iGarr.data();
			else
			{
				buffer.init(sizes[q]);
				mpiWorld->recvData(buffer, info.whose(q), q);
				vectors = buffer.data();
			}
			fprintf(fp, "# ");
			info.kpointPrint(fp, q, true);
			fprintf(fp, "\n");
			for (size_t g = 0; g < sizes[q]; g++)
				fprintf(fp, "%d %d %d\n", vectors[g][0], vectors[g][1], vectors[g][2]);
			fprintf(fp, "\n");
		}
		if (fclose(fp))
			die_alone("Failed writing PAW G vectors.\n");
	}
	else
		for (int q = info.qStart; q < info.qStop; q++)
			mpiWorld->sendData(bases[q].iGarr, 0, q);
}
} // namespace

void Dump::dumpPAW(bool wavefunctions, bool realSpace, bool overlap, bool momenta, bool eigenvalues) const
{
	bool hasPAW = false;
	for (auto sp : e->iInfo.species)
		hasPAW |= sp->isPAW();
	if (!hasPAW)
		die("All-electron dumps require at least one PAW XML species.\n");
	for (auto sp : e->iInfo.species)
		if (!sp->atpos.empty() && !sp->isPAW())
			die("All-electron exports require PAW datasets for every species.\n");
	const auto &info = e->eInfo;
	const auto &vars = e->eVars;
	const string prefix = getFilename("paw");
	const double cutoff = pawReconstructionCutoff ? pawReconstructionCutoff : 4 * e->cntrl.Ecut;

	// The PAW transformation preserves eigenvalues. Do not store a diagonal H.
	if (eigenvalues)
		info.write(vars.Hsub_eigs, (prefix + ".eigenvals").c_str());
	if (overlap)
	{
		std::vector<matrix> output(info.nStates);
		for (int q = info.qStart; q < info.qStop; q++)
			output[q] = vars.C[q] ^ O(vars.C[q]);
		info.write(output, (prefix + ".overlap").c_str());
	}
	if (momenta)
	{
		std::vector<matrix> output(info.nStates);
		for (int q = info.qStart; q < info.qStop; q++)
		{
			int n = info.nBands;
			output[q] = zeroes(n, 3 * n);
			for (int axis = 0; axis < 3; axis++)
				output[q].set(0, n, axis * n, (axis + 1) * n, e->iInfo.allElectronMomentum(vars.C[q], axis));
		}
		info.write(output, (prefix + ".momenta").c_str(), info.nBands, 3 * info.nBands);
	}

	std::vector<size_t> sizes(info.nStates, 0), offsets(info.nStates, 0);
	vector3<int> fftGrid;
	if (wavefunctions || realSpace)
	{
		if (cutoff < e->cntrl.Ecut)
			die("paw-reconstruction-cutoff must be at least elec-cutoff.\n");
		GridInfo grid;
		grid.R = e->gInfo.R;
		grid.Gmax = sqrt(2 * cutoff);
		grid.initialize(true, e->symm.getMatrices());
		fftGrid = grid.S;
		std::vector<Basis> bases(info.nStates);
		for (int q = info.qStart; q < info.qStop; q++)
		{
			bases[q].setup(grid, e->iInfo, cutoff, info.qnums[q].k);
			sizes[q] = bases[q].nbasis;
		}
		// All collectives stay outside the rank-local reconstruction loop.
		mpiWorld->allReduceData(sizes, MPIUtil::ReduceSum);
		for (int q = 1; q < info.nStates; q++)
			offsets[q] = offsets[q - 1] + sizes[q - 1] * info.nBands * sizeof(complex);
#if MPI_SAFE_WRITE
		FILE *fp = nullptr;
		if (wavefunctions && mpiWorld->isHead())
		{
			fp = fopen((prefix + ".wfns").c_str(), "wb");
			if (!fp)
				die_alone("Cannot write PAW wavefunctions.\n");
		}
		const bool gather = wavefunctions && mpiWorld->isHead();
		const int qFirst = gather ? 0 : info.qStart, qLast = gather ? info.nStates : info.qStop;
#else
		MPIUtil::File fp;
		if (wavefunctions)
			mpiWorld->fopenWrite(fp, (prefix + ".wfns").c_str());
		const int qFirst = info.qStart, qLast = info.qStop;
#endif
		for (int q = qFirst; q < qLast; q++)
		{
#if MPI_SAFE_WRITE
			if (!info.isMine(q))
			{
				ManagedArray<complex> buffer;
				buffer.init(sizes[q] * info.nBands);
				mpiWorld->recvData(buffer, info.whose(q), q);
				buffer.write(fp);
				continue;
			}
#endif
			ColumnBundle ae = e->iInfo.allElectronWavefunctions(vars.C[q], bases[q]);
			if (wavefunctions)
			{
				// Byte-identical layout to ElecInfo::write(vector<ColumnBundle>):
				// state, band, G (fastest), little-endian complex128, no header.
				// Stream one bundle at a time to bound host and GPU memory.
#if MPI_SAFE_WRITE
				if (mpiWorld->isHead())
					ae.write(fp);
				else
					mpiWorld->sendData(ae, 0, q);
#else
				mpiWorld->fseek(fp, offsets[q], SEEK_SET);
				mpiWorld->fwriteData(ae, fp);
#endif
			}
			if (realSpace)
				for (int b = 0; b < info.nBands; b++)
				{
					char suffix[80];
					snprintf(suffix, sizeof(suffix), ".q%d.b%d.rs", q, b);
					saveRawBinary(I(ae.getColumn(b, 0)), (prefix + suffix).c_str());
				}
			logPrintf("PAW export state %d: %zu G vectors at %g Hartree\n", q, sizes[q], cutoff);
		}
		if (wavefunctions)
		{
#if MPI_SAFE_WRITE
			if (fp && fclose(fp))
				die_alone("Failed writing PAW wavefunctions.\n");
#else
			mpiWorld->fclose(fp);
#endif
			writeGvectors(info, bases, sizes, prefix + ".Gvectors");
		}
	}
	if (mpiWorld->isHead())
	{
		std::ofstream meta((prefix + ".json").c_str());
		if (!meta)
			die_alone("Cannot write PAW export metadata.\n");
		meta << std::setprecision(17) << std::boolalpha
		     << "{\n  \"format\": \"JDFTx-PAW-2\",\n  \"units\": \"Hartree atomic units\",\n"
		     << "  \"states\": " << info.nStates << ",\n  \"bands\": " << info.nBands
		     << ",\n  \"byte_order\": \"little\",\n  \"volume\": " << e->gInfo.detR
		     << ",\n  \"reconstruction_cutoff\": " << cutoff << ",\n"
		     << "  \"outputs\": {\"wavefunctions\": " << wavefunctions
		     << ", \"real_space_wavefunctions\": " << realSpace << ", \"overlap\": " << overlap
		     << ", \"momenta\": " << momenta << ", \"eigenvalues\": " << eigenvalues << "},\n"
		     << "  \"wavefunctions\": \"paw.wfns: native ColumnBundle layout; state, band, G (fastest); "
		        "complex128, no header; periodic part u_nk; psi=exp(i k.r) u; no renormalization\",\n"
		     << "  \"wavefunction_normalization\": \"integral_cell |psi|^2 = volume * sum_G |c_G|^2; finite "
		        "AE cutoff loses norm\",\n"
		     << "  \"real_space_layout\": \"paw.q<state>.b<band>.rs: periodic u_nk, complex128, x/y/z (z "
		        "fastest), "
		        "r=R*(ix/Sx,iy/Sy,iz/Sz); same convention as RealSpaceWfns\",\n"
		     << "  \"overlap_layout\": \"complex128: state, ket band, bra band (fastest)\",\n"
		     << "  \"momenta_layout\": \"complex128: state, direction [x,y,z], ket band, bra band "
		        "(fastest)\",\n"
		     << "  \"eigenvalue_layout\": \"float64: state, band (fastest); identical to BandEigs\",\n"
		     << "  \"scope\": \"frozen-core valence states; canonical momentum differs from velocity for "
		        "scalar-relativistic datasets\",\n"
		     << "  \"fft_grid\": ";
		if (wavefunctions || realSpace)
			meta << "[" << fftGrid[0] << "," << fftGrid[1] << "," << fftGrid[2] << "]";
		else
			meta << "null";
		meta << ",\n  \"basis_sizes\": [";
		for (int q = 0; q < info.nStates; q++)
		{
			if (q)
				meta << ",";
			meta << sizes[q];
		}
		meta << "],\n  \"wavefunction_offsets_bytes\": [";
		for (int q = 0; q < info.nStates; q++)
		{
			if (q)
				meta << ",";
			meta << offsets[q];
		}
		meta << "],\n  \"lattice_columns_bohr\": [";
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
		meta.close();
		if (!meta)
			die_alone("Failed writing PAW export metadata.\n");
	}
}
