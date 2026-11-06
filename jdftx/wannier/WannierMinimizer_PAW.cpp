/* PAW-reconstructed Wannier orbitals. GNU GPL v3 or later; see Wannier.h. */
#include <wannier/WannierMinimizer.h>
#include <wannier/WannierMinimizerFD.h>
#include <core/ScalarFieldIO.h>
#include <fstream>
#include <iomanip>

namespace
{
template <typename T> void jsonVector(std::ostream &out, const vector3<T> &v)
{
	out << "[" << v[0] << "," << v[1] << "," << v[2] << "]";
}

template <typename T> void jsonMatrix(std::ostream &out, const matrix3<T> &m)
{
	out << "[";
	for (int i = 0; i < 3; i++)
	{
		if (i)
			out << ",";
		jsonVector(out, m.row(i));
	}
	out << "]";
}
} // namespace

double WannierMinimizer::pawCutoff() const
{
	return e.dump.pawReconstructionCutoff ? e.dump.pawReconstructionCutoff : 4 * e.cntrl.Ecut;
}

void WannierMinimizer::initPAWCommonBasis(GridInfo &grid, Basis &common) const
{
	double kmax = 0.;
	for (const Kpoint &k : kpoints)
		kmax = std::max(kmax, sqrt(e.gInfo.GGT.metric_length_squared(k.k)));
	const double gmax = sqrt(2 * pawCutoff()) + kmax;
	grid.R = e.gInfo.R;
	// Cover the union of all physical G+k spheres and their shifted indices.
	grid.Gmax = gmax + kmax;
	grid.initialize(true, sym);
	common.setup(grid, e.iInfo, .5 * gmax * gmax * (1 + symmThreshold), vector3<>());
}

ColumnBundle WannierMinimizer::getAEWfns(const ColumnBundle &smooth,
                                         const ColumnBundleTransform::BasisWrapper &wrapper) const
{
	Basis stateBasis;
	stateBasis.setup(*wrapper.basis.gInfo, e.iInfo, pawCutoff(), smooth.qnum->k);
	ColumnBundle ae = e.iInfo.allElectronWavefunctions(smooth, stateBasis);
	ColumnBundle result(smooth.nCols(), wrapper.basis.nbasis, &wrapper.basis, smooth.qnum, isGpuEnabled());
	result.zero();
	SpaceGroupOp identity;
	ColumnBundleTransform transform(smooth.qnum->k, stateBasis, smooth.qnum->k, wrapper, 1, identity, 1);
	transform.scatterAxpy(1., ae, result, 0, 1);
	return result;
}

void WannierMinimizer::saveMLWF_AE(int iSpin)
{
	resumeOperatorThreading();
	const double cutoff = pawCutoff();
	const Supercell &supercell = *e.coulombParams.supercell;
	GridInfo unitGrid, superGrid;
	unitGrid.R = e.gInfo.R;
	unitGrid.Gmax = sqrt(2 * cutoff);
	unitGrid.initialize(true, sym);
	superGrid.R = supercell.Rsuper;
	// A wavefunction grid suffices; do not allocate a density-resolution FFT box.
	superGrid.GmaxRho = sqrt(2 * cutoff) * (1 + symmThreshold);
	superGrid.initialize(true);
	Basis superBasis;
	superBasis.setup(superGrid, e.iInfo, cutoff * (1 + symmThreshold), qnumSuper.k);
	ColumnBundleTransform::BasisWrapper wrapper(superBasis);
	ColumnBundle result(nCenters, superBasis.nbasis, &superBasis, &qnumSuper, isGpuEnabled());
	result.zero();
	std::vector<std::vector<size_t>> byReduced(qCount);
	for (size_t ik = 0; ik < kMesh.size(); ik++)
		byReduced[kMesh[ik].point.iReduced].push_back(ik);
	logPrintf("Computing all-electron Wannier functions at %g Hartree ...\n", cutoff);
	for (int q = e.eInfo.qStart; q < e.eInfo.qStop; q++)
		if (e.eInfo.qnums[q].index() == iSpin)
		{
			Basis unitBasis;
			unitBasis.setup(unitGrid, e.iInfo, cutoff, e.eInfo.qnums[q].k);
			ColumnBundle ae = e.iInfo.allElectronWavefunctions(e.eVars.C[q], unitBasis);
			for (size_t ik : byReduced[q - iSpin * qCount])
			{
				const auto &ki = kMesh[ik];
				// Antiunitary scatter conjugates both the orbitals and these pre-rotations.
				ColumnBundle rotated = ae * (ki.point.invert < 0 ? conj(ki.U) : ki.U);
				ColumnBundleTransform transform(e.eInfo.qnums[q].k, unitBasis, qnumSuper.k, wrapper, 1,
				                                sym[ki.point.iSym], ki.point.invert, supercell.super);
				transform.scatterAxpy(ki.point.weight, rotated, result, 0, 1);
			}
		}
	// Every rank participates, including ranks without any reduced states.
	mpiWorld->reduceData(result, MPIUtil::ReduceSum);
	if (mpiWorld->isHead())
	{
		result = translate(std::move(result), vector3<>(.5, .5, .5));
		const string prefix = wannier.getFilename(Wannier::FilenameDump, "mlwfCAE", &iSpin);
		if (wannier.saveAEWfns)
			result.write(prefix.c_str());
		// Retain the native mlwfC header convention so basis consumers can reuse it.
		FILE *fp = fopen((prefix + ".header").c_str(), "w");
		if (!fp)
			die_alone("Cannot write all-electron Wannier basis header.\n");
		fprintf(fp, "%d %zu #nColumns, columnLength\n", nCenters, superBasis.nbasis);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				fprintf(fp, "%.17g ", superGrid.GT(i, j));
		fprintf(fp, "#GT row-major (G col-major), iGarr follows:\n");
		for (const auto &g : superBasis.iGarr)
			fprintf(fp, "%d %d %d\n", g[0], g[1], g[2]);
		if (fclose(fp))
			die_alone("Failed writing all-electron Wannier basis header.\n");
		if (wannier.saveAEWfnsRealSpace)
			for (int n = 0; n < nCenters; n++)
			{
				complexScalarField psi = I(result.getColumn(n, 0));
				if (qnumSuper.k.length_squared() > symmThresholdSq)
					multiplyBlochPhase(psi, qnumSuper.k);
				ostringstream name;
				name << n << ".mlwfAE";
				const string filename = wannier.getFilename(Wannier::FilenameDump, name.str(), &iSpin);
				saveRawBinary(psi, filename.c_str());
			}
		std::ofstream meta((prefix + ".json").c_str());
		if (!meta)
			die_alone("Cannot write all-electron Wannier metadata.\n");
		meta << std::setprecision(17) << std::boolalpha
		     << "{\n\"format\":\"JDFTx-PAW-Wannier-1\",\n\"byte_order\":\"little\",\n"
		     << "\"orbitals\":" << nCenters << ",\n\"source_bands\":" << nBands
		     << ",\n\"basis_size\":" << superBasis.nbasis << ",\n\"reconstruction_cutoff\":" << cutoff
		     << ",\n\"supercell_volume\":" << superGrid.detR
		     << ",\n\"reciprocal_wavefunctions\":" << wannier.saveAEWfns
		     << ",\n\"real_space_wavefunctions\":" << wannier.saveAEWfnsRealSpace
		     << ",\n\"layout\":\"little-endian complex128; reciprocal: orbital,G; real space: x,y,z (z "
		        "fastest)\",\n"
		     << "\"scope\":\"frozen-core valence PAW; no renormalization or phase removal\",\n"
		     << "\"translation_supercell\":[0.5,0.5,0.5],\n\"fft_grid\":";
		jsonVector(meta, superGrid.S);
		meta << ",\n\"supercell_kpoint\":";
		jsonVector(meta, qnumSuper.k);
		meta << ",\n\"lattice_columns_bohr\":";
		jsonMatrix(meta, superGrid.RT);
		meta << ",\n\"primitive_lattice_columns_bohr\":";
		jsonMatrix(meta, e.gInfo.RT);
		meta << ",\n\"supercell_matrix\":";
		jsonMatrix(meta, supercell.super);
		meta << ",\n\"kmesh\":[";
		for (size_t ik = 0; ik < kMesh.size(); ik++)
		{
			if (ik)
				meta << ",";
			const auto &k = kMesh[ik].point;
			meta << "{\"reduced_state\":" << k.iReduced + iSpin * qCount << ",\"k\":";
			jsonVector(meta, k.k);
			meta << ",\"weight\":" << k.weight << ",\"invert\":" << k.invert << ",\"rotation\":";
			jsonMatrix(meta, sym[k.iSym].rot);
			meta << ",\"translation\":";
			jsonVector(meta, sym[k.iSym].a);
			meta << "}";
		}
		meta << "]\n}\n";
		meta.close();
		if (!meta)
			die_alone("Failed writing all-electron Wannier metadata.\n");
		logPrintf("Saved all-electron Wannier orbitals and basis: %s\n", prefix.c_str());
	}
	suspendOperatorThreading();
}

void WannierMinimizer::saveMLWF_PAWMetadata(int iSpin) const
{
	if (!mpiWorld->isHead())
		return;
	const string filename = wannier.getFilename(Wannier::FilenameDump, "mlwfPAW.json", &iSpin);
	std::ofstream out(filename.c_str());
	out << std::setprecision(17) << std::boolalpha
	    << "{\n\"format\":\"JDFTx-PAW-Wannier-operators-1\",\n\"scope\":\"frozen-core PAW valence states\",\n"
	    << "\"reconstruction_cutoff\":" << pawCutoff() << ",\n\"saveMomenta\":" << wannier.saveMomenta
	    << ",\n\"saveR\":" << wannier.saveR << ",\n\"saveRP\":" << wannier.saveRP
	    << ",\n\"slabWeight\":" << bool(wannier.zH) << ",\n\"saveZ\":" << bool(wannier.zVfilename.length())
	    << ",\n\"mlwfH\":\"PAW valence eigenvalues rotated into the Wannier basis\",\n"
	    << "\"mlwfP\":\"analytic PAW canonical i*p, including AE-minus-PS augmentation\",\n"
	    << "\"mlwfR\":\"finite-difference position from analytic finite-dk PAW overlaps\",\n"
	    << "\"mlwfRP\":\"finite-difference short-range position times i*p from reconstructed AE orbitals\",\n"
	    << "\"mlwfZ_and_mlwfW\":\"scalar-weight integrals of reconstructed AE orbitals\",\n"
	    << "\"convergence\":\"converge k mesh for R/RP; converge reconstruction cutoff for RP/Z/W; no "
	       "renormalization\",\n"
	    << "\"derivative_stencil\":[";
	if (wannier.saveR || wannier.saveRP)
	{
		const auto &fd = static_cast<const WannierMinimizerFD &>(*this);
		bool first = true;
		for (const auto &edge : fd.edges_bi[0])
		{
			if (!first)
				out << ",";
			first = false;
			out << "{\"dk\":";
			jsonVector(out, edge.point.k - kMesh[0].point.k);
			out << ",\"b_cartesian\":";
			jsonVector(out, edge.b);
			out << ",\"weight\":" << edge.wb << "}";
		}
	}
	out << "]\n}\n";
	out.close();
	if (!out)
		die_alone("Cannot write PAW Wannier operator metadata.\n");
}
