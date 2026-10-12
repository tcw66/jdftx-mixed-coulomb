/*-------------------------------------------------------------------
Copyright 2022 Brandon Li

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#include <commands/minimize.h>
#include <climits>

//An enum entry for each configurable option of MinimizeParams
enum VPTParamsMember
{	VPTp_nIterations,
	VPTp_algorithm,
	VPTp_residualTol,
	VPTp_residualDiffThreshold,
	VPTp_CGBypass,
	VPTp_recomputeResidual,
	VPTp_Delim //!< delimiter to detect end of input
};

EnumStringMap<VPTParamsMember> VPTpMap
(	VPTp_nIterations, "nIterations",
	VPTp_algorithm, "algorithm",
	VPTp_residualTol, "residualTol",
	VPTp_residualDiffThreshold, "residualDiffThreshold",
	VPTp_CGBypass, "CGBypass",
	VPTp_recomputeResidual, "recomputeResidual"
);
EnumStringMap<VPTParamsMember> VPTpDescMap
(	VPTp_nIterations, "maximum number of iterations",
	VPTp_algorithm, "CG|MINRES (linear solver algorithm)",
	VPTp_residualTol, "relative error of residual required to terminate solver (default value of 1e-4 is sufficient most of the time)",
	VPTp_residualDiffThreshold, "terminate if difference in consecutive residuals falls below threshold (MINRES only)",
	VPTp_CGBypass, "bypass CG stopping criterion (residual increases 3 iterations in a row)",
	VPTp_recomputeResidual, "recompute residual every ten iterations (CG only)"
);

EnumStringMap<SolverParams::Algorithms> algMap
(	SolverParams::CG, "CG",
	SolverParams::MINRES, "MINRES"
);

struct CommandPerturbMinimize : public Command
{
	CommandPerturbMinimize() : Command("perturb-minimize", "jdftx/Perturb")
	{	emptyParamError =
			"   Note: nIterations defaults to 0 and must be set manually to enable variational perturbation solver.\n";
		format = "<key1> <value1> <key2> <value2> ...";
		comments = "where possible keys and value types are:"
			+ addDescriptions(VPTpMap.optionList(), linkDescription(VPTpMap, VPTpDescMap))
			+ "\n\nAny number of these key-value pairs may be specified in any order.";
		hasDefault = true;
	}
	
    void process(ParamList& pl, Everything& e)
    {	SolverParams& p = e.pertInfo.solverParams;
		while(true)
		{	VPTParamsMember key;
			pl.get(key, VPTp_Delim, VPTpMap, "key");
			switch(key)
			{	case VPTp_nIterations: pl.get(p.nIterations, 0, "nIterations", true); break;
				case VPTp_algorithm: pl.get(p.algorithm, SolverParams::MINRES, algMap, "algorithm", true); break;
				case VPTp_residualTol: pl.get(p.residualTol, 1e-4, "residualTol", true); break;
				case VPTp_residualDiffThreshold: pl.get(p.residualDiffThreshold, 1e-4, "residualDiffThreshold", true); break;
				case VPTp_CGBypass: pl.get(p.CGBypass, false, boolMap, "CGBypass", true); break;
				case VPTp_recomputeResidual: pl.get(p.recomputeResidual, false, boolMap, "recomputeResidual", true); break;
				case VPTp_Delim:
					if(p.nIterations<0 || !(p.residualTol>0.) || !(p.residualDiffThreshold>=0.))
						throw string("Require nIterations >= 0, residualTol > 0 and residualDiffThreshold >= 0");
					return; //end of input
			}
		}
	}
	
	void printStatus(Everything& e, int iRep)
	{	SolverParams& p = e.pertInfo.solverParams;
		logPrintf(" \\\n\tnIterations            %d", p.nIterations);
		logPrintf(" \\\n\talgorithm              %s", algMap.getString(p.algorithm));
		logPrintf(" \\\n\tresidualTol            %g", p.residualTol);
		logPrintf(" \\\n\tresidualDiffThreshold  %g", p.residualDiffThreshold);
		logPrintf(" \\\n\tCGBypass               %s", boolMap.getString(p.CGBypass));
		logPrintf(" \\\n\trecomputeResidual      %s", boolMap.getString(p.recomputeResidual));
	}
}
commandPerturbMinimize;


struct CommandPerturbVexternal : public Command
{
	CommandPerturbVexternal() : Command("perturb-Vexternal", "jdftx/Perturb")
	{
		format = "<filename>";
		comments =
			"Introduce external potential perturbation (in Hartrees).\n"
			"Must be specified as a real space field, or complex field for incommensurate perturbations.";
	}

	void process(ParamList& pl, Everything& e)
	{	e.pertInfo.dVext = std::make_shared<VextPerturbation>(e);
		e.pertInfo.dVext->dVexternalFilename.resize(1);
		pl.get(e.pertInfo.dVext->dVexternalFilename[0], string(), "filename", true);
	}

	void printStatus(Everything& e, int iRep)
	{	for(const string& filename: e.pertInfo.dVext->dVexternalFilename)
			logPrintf("%s ", filename.c_str());
	}
}
commandPerturbVexternal;


struct CommandPerturbRhoExternal : public Command
{
	CommandPerturbRhoExternal() : Command("perturb-rhoExternal", "jdftx/Perturb")
	{
		format = "<filename>";
		comments =
			"Introduce external charge density perturbation.\n"
			"Must be specified as a real space field, or complex field for incommensurate perturbations.";
	}

	void process(ParamList& pl, Everything& e)
	{	e.pertInfo.drhoExt = std::make_shared<RhoPerturbation>(e);
		e.pertInfo.drhoExt->drhoExtFilename.resize(1);
		pl.get(e.pertInfo.drhoExt->drhoExtFilename[0], string(), "filename", true);
	}

	void printStatus(Everything& e, int iRep)
	{	for(const string& filename: e.pertInfo.drhoExt->drhoExtFilename)
			logPrintf("%s ", filename.c_str());
	}
}
commandPerturbRhoExternal;


struct CommandPerturbIon : public Command
{
	CommandPerturbIon() : Command("perturb-ion", "jdftx/Perturb")
	{
		format = "<species> <atom> <dx0> <dx1> <dx2>";
		comments =
			"Perturb atom position. Coordinate system determined by coords-type\n";
		
		require("latt-scale");
		require("coords-type");
	}

	void process(ParamList& pl, Everything& e)
	{	int sp, at;
		vector3<> dx;
		pl.get(sp, 0, "species", true);
		pl.get(at, 0, "atom", true);
		pl.get(dx[0], 0.0, "dx0", true);
		pl.get(dx[1], 0.0, "dx1", true);
		pl.get(dx[2], 0.0, "dx2", true);
		
		if(e.iInfo.coordsType == CoordsLattice) dx = e.gInfo.R*dx;
		
		e.pertInfo.datom = std::make_shared<AtomPerturbation>(sp, at, dx, e);
		e.pertInfo.datom->mode.dirLattice = inv(e.gInfo.R)*e.pertInfo.datom->mode.dirCartesian; //Constructor does not initialize dirLattice properly because gInfo not set up yet
	}

	void printStatus(Everything& e, int iRep)
	{	
		auto pert = e.pertInfo.datom;
		logPrintf("%i %i %g %g %g", pert->mode.sp, pert->mode.at, pert->mode.dirCartesian[0], pert->mode.dirCartesian[1], pert->mode.dirCartesian[2]);
	}
}
commandPerturbIon;

struct CommandPerturbElectricField : public Command
{
	CommandPerturbElectricField() : Command("perturb-electric-field", "jdftx/Perturb")
	{
		format = "<Ex> <Ey> <Ez>";
		comments =
			"Perturb external electric field. This field has the same properties as the one created by the command electric-field.";
	}

	void process(ParamList& pl, Everything& e)
	{
		e.pertInfo.dElectricField = std::make_shared<ElectricFieldPerturbation>(e);
		pl.get(e.pertInfo.dElectricField->Efield[0], 0., "Ex", true);
		pl.get(e.pertInfo.dElectricField->Efield[1], 0., "Ey", true);
		pl.get(e.pertInfo.dElectricField->Efield[2], 0., "Ez", true);
	}

	void printStatus(Everything& e, int iRep)
	{	for(int k=0; k<3; k++) logPrintf("%lg ", e.pertInfo.dElectricField->Efield[k]);
	}
}
commandPerturbElectricField;


struct CommandPerturbWavevector : public Command
{
	CommandPerturbWavevector() : Command("perturb-wavevector", "jdftx/Perturb")
	{
		format = "<q0> <q1> <q2>";
		comments =
			"Set the bloch phase of the perturbation potential to exp(iGq*r) where q = (<q0>, <q1>, <q2>) is a reciprocal lattice vector\n."
			"If perturbation is not specified, will perform band structure calculation with k+q and k-q vectors.";
	}

    void process(ParamList& pl, Everything& e)
    {
		pl.get(e.pertInfo.qvec[0], 0.0, "q0");
		pl.get(e.pertInfo.qvec[1], 0.0, "q1");
		pl.get(e.pertInfo.qvec[2], 0.0, "q2");

		e.pertInfo.commensurate = !e.pertInfo.qvec.isNonzero();
	}

	void printStatus(Everything& e, int iRep)
	{
		logPrintf("%f %f %f", e.pertInfo.qvec[0], e.pertInfo.qvec[1], e.pertInfo.qvec[2]);
	}
}
commandPerturbWavevector;

struct CommandPerturbIncommensurateWavefunctions : public Command
{
	CommandPerturbIncommensurateWavefunctions() : Command("perturb-incommensurate-wavefunctions", "jdftx/Perturb")
	{
		format = "<filename> [<EcutOld>]";
		comments =
			"Optionally supply occupied states at k+q and k-q for a finite-q perturbation.\n"
			"Without this command, offset states are computed automatically in the ground-state potential.\n";
	}

    void process(ParamList& pl, Everything& e)
    {
		pl.get(e.pertInfo.wfnsFilename, string(), "filename", true);
		auto conversion = std::make_shared<ElecInfo::ColumnBundleReadConversion>();
		conversion->realSpace = false;
		pl.get(conversion->EcutOld, 0.0, "EcutOld");
		if(conversion->EcutOld > 0.0) e.pertInfo.readConversion = conversion;
	}

	void printStatus(Everything& e, int iRep)
	{
		if (!e.pertInfo.readConversion)
			logPrintf("%s", e.pertInfo.wfnsFilename.c_str());
		else
			logPrintf("%s %g", e.pertInfo.wfnsFilename.c_str(), e.pertInfo.readConversion->EcutOld);
	}
}
commandPerturbIncommensurateWavefunctions;

struct CommandPerturbTest : public Command
{
	CommandPerturbTest() : Command("perturb-test", "jdftx/Perturb")
	{	comments = "Perform finite difference tests of perturbation operators.\n";
	}
	void process(ParamList& pl, Everything& e) { e.pertInfo.testing = true; }
	void printStatus(Everything& e, int iRep) { return; }
}
commandPerturbTest;


struct CommandPhononDFPT : public Command
{
	CommandPhononDFPT() : Command("phonon-dfpt", "jdftx/Perturb")
	{
		format="<Nq0> <Nq1> <Nq2>";
		comments="Compute primitive-cell DFPT phonons on a Gamma-centered uniform q grid, then Fourier transform\n"
			"to phononCellMap, phononOmegaSq and phononBasis for interpolation. Also writes phononForceMatrixQ.\n"
			"Equivalent q points are reconstructed using space-group symmetry and time reversal by default.\n"
			"Requires symmetries none, periodic or slab Coulomb (including embedding), nonmagnetic scalar states, analytic LDA/GGA,\n"
			"and nonrelativistic norm-conserving pseudopotentials without nonlinear core corrections.\n"
			"Use fully occupied bands for insulators; metals require Fermi or Gauss smearing at fixed electron number,\n"
			"with enough empty bands that the highest-band occupation is below 1e-6 at k and k+/-q.\n"
			"For slab Coulomb, set q-grid dimension 1 along the truncated direction.\n"
			"Set perturb-minimize nIterations and residualTol explicitly. No offset wavefunction files are needed.";
		forbid("perturb-ion"); forbid("perturb-Vexternal"); forbid("perturb-rhoExternal");
		forbid("perturb-electric-field"); forbid("perturb-wavevector"); forbid("perturb-incommensurate-wavefunctions");
	}
	void process(ParamList& pl,Everything& e)
	{
		for(int j=0;j<3;j++) {
			pl.get(e.pertInfo.phononGrid[j],0,"q-grid dimension",true);
			if(e.pertInfo.phononGrid[j]<=0) throw string("q-grid dimensions must be positive");
		}
		long long count=1;
		for(int j=0;j<3;j++) {
			count*=e.pertInfo.phononGrid[j];
			if(count>INT_MAX) throw string("q-grid is too large");
		}
	}
	void printStatus(Everything& e,int iRep)
	{ for(int j=0;j<3;j++) logPrintf("%d ",e.pertInfo.phononGrid[j]); }
} commandPhononDFPT;

struct CommandPhononDFPTElectronPhonon : public Command
{
	CommandPhononDFPTElectronPhonon() : Command("phonon-dfpt-electron-phonon", "jdftx/Perturb")
	{
		format="<yes|no>";
		comments="Export screened electron-phonon matrices in the native phonon / Wannier format (default no).\n"
			"Writes phononHsub and phononKpts, plus matching wfns, eigenvals and fillings.\n"
			"Requires a Gamma-centered electronic mesh with kpoint-folding divisible by the phonon q grid.\n"
			"Use wannier phononSupercell with this q grid and wannier-initial-state with the DFPT dump prefix.\n"
			"Q/displacement symmetry reductions reconstruct screened potentials; band contractions retain the full k mesh.\n"
			"Offset-state reuse, symmetry-reduced SCF restarts and MPI q groups remain available.\n"
			"To include conduction bands, use Fermi/Gauss smearing and enough elec-n-bands in SCF and DFPT.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{ pl.get(e.pertInfo.phononElectronPhonon,false,boolMap,"export electron-phonon matrices",true); }
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%s",boolMap.getString(e.pertInfo.phononElectronPhonon)); }
} commandPhononDFPTElectronPhonon;

struct CommandPhononDFPTOptimize : public Command
{
	CommandPhononDFPTOptimize() : Command("phonon-dfpt-optimize", "jdftx/Perturb")
	{
		format="<yes|no>";
		comments="Use direct complex response operators and fused electron-phonon export (default yes).\n"
			"Disable to compare with separate bare/induced reference contractions.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{ pl.get(e.pertInfo.phononOptimize,true,boolMap,"optimize phonon operators",true); }
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%s",boolMap.getString(e.pertInfo.phononOptimize)); }
} commandPhononDFPTOptimize;

struct CommandPhononDFPTEphCache : public Command
{
	CommandPhononDFPTEphCache() : Command("phonon-dfpt-eph-cache-size", "jdftx/Perturb")
	{
		format="<MiB>";
		comments="Maximum cached real-space export wavefunctions per rank in MiB (default 512).\n"
			"Zero disables this cache. Storage is released before the next response solve.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{
		pl.get(e.pertInfo.phononEphCacheMiB,512.,"export cache size",true);
		if(!std::isfinite(e.pertInfo.phononEphCacheMiB) || e.pertInfo.phononEphCacheMiB<0.
			|| e.pertInfo.phononEphCacheMiB>=double(SIZE_MAX)/(1024.*1024.))
			throw string("cache size must be finite, nonnegative and fit in the address space");
	}
	void printStatus(Everything& e,int iRep) { logPrintf("%g",e.pertInfo.phononEphCacheMiB); }
} commandPhononDFPTEphCache;

struct CommandPhononDFPTEphReciprocity : public Command
{
	CommandPhononDFPTEphReciprocity() : Command("phonon-dfpt-eph-reciprocity", "jdftx/Perturb")
	{
		format="<yes|no>";
		comments="Contract one Hermitian k pair and write its conjugate-transpose partner (default yes).\n"
			"Requires paired q and -q in the same symmetry orbit; otherwise falls back to full contractions.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{ pl.get(e.pertInfo.phononEphReciprocity,true,boolMap,"use export reciprocity",true); }
	void printStatus(Everything& e,int iRep) { logPrintf("%s",boolMap.getString(e.pertInfo.phononEphReciprocity)); }
} commandPhononDFPTEphReciprocity;

struct CommandPhononDFPTSymmetries : public Command
{
	CommandPhononDFPTSymmetries() : Command("phonon-dfpt-symmetries", "jdftx/Perturb")
	{
		format="<yes|no>";
		comments="Use space-group symmetry and time reversal to reduce phonon q points (default yes).\n"
			"Only operations preserving the q grid, full electronic k mesh, FFT mesh and ground-state potential are used.\n"
			"Electronic symmetries must still be none. Displacement reduction uses these operations too.\n"
			"Set phonon-dfpt-displacement-symmetries no to retain all Cartesian displacement solves.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{ pl.get(e.pertInfo.phononUseSymmetry,true,boolMap,"use q symmetry",true); }
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%s",boolMap.getString(e.pertInfo.phononUseSymmetry)); }
} commandPhononDFPTSymmetries;

struct CommandPhononDFPTQGroups : public Command
{
	CommandPhononDFPTQGroups() : Command("phonon-dfpt-q-groups", "jdftx/Perturb")
	{
		format="<nGroups>";
		comments="Split MPI ranks into independent groups which solve different representative phonon q points.\n"
			"Within each group, electronic k points remain distributed over its ranks. Default: 1 group.\n"
			"Each group initializes and polishes its own full ground state; final phonon outputs are written once.\n"
			"Use no more groups than ranks, and preferably no more than the number of representative q points.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{
		pl.get(e.pertInfo.phononQGroups,1,"number of q groups",true);
		if(e.pertInfo.phononQGroups<1) throw string("number of q groups must be positive");
	}
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%d",e.pertInfo.phononQGroups); }
} commandPhononDFPTQGroups;

struct CommandPhononDFPTDisplacements : public Command
{
	CommandPhononDFPTDisplacements() : Command("phonon-dfpt-displacement-symmetries", "jdftx/Perturb")
	{
		format="<yes|no>";
		comments="Reduce independent atomic displacement responses using the little group of each q (default yes).\n"
			"Includes Cartesian rotations, atom permutations, cell phases and time reversal.\n"
			"Set no to calculate every Cartesian displacement explicitly for validation.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{ pl.get(e.pertInfo.phononReduceDisplacements,true,boolMap,"reduce displacements",true); }
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%s",boolMap.getString(e.pertInfo.phononReduceDisplacements)); }
} commandPhononDFPTDisplacements;

struct CommandPhononDFPTReuseStates : public Command
{
	CommandPhononDFPTReuseStates() : Command("phonon-dfpt-reuse-offset-states", "jdftx/Perturb")
	{
		format="<yes|no>";
		comments="Reuse ground-state occupied wavefunctions for k+q and k-q on the electronic mesh (default yes).\n"
			"Checks the Hamiltonian residual and refines states when necessary; off-mesh states use the offset eigensolver.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{ pl.get(e.pertInfo.phononReuseOffsetStates,true,boolMap,"reuse offset states",true); }
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%s",boolMap.getString(e.pertInfo.phononReuseOffsetStates)); }
} commandPhononDFPTReuseStates;

struct CommandPhononDFPTCache : public Command
{
	CommandPhononDFPTCache() : Command("phonon-dfpt-cache", "jdftx/Perturb")
	{
		format="<yes|no>";
		comments="Cache frozen-ground-state XC data and bare displacement potentials within each q (default yes).\n"
			"Also retain displacement source vectors up to phonon-dfpt-source-cache-size MiB per rank.\n"
			"Set no to recompute these quantities for numerical validation.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{ pl.get(e.pertInfo.phononCache,true,boolMap,"cache phonon data",true); }
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%s",boolMap.getString(e.pertInfo.phononCache)); }
} commandPhononDFPTCache;

struct CommandPhononDFPTSourceCacheSize : public Command
{
	CommandPhononDFPTSourceCacheSize() : Command("phonon-dfpt-source-cache-size", "jdftx/Perturb")
	{
		format="<MiB>";
		comments="Maximum additional source-vector cache storage per MPI rank (default 128 MiB).\n"
			"Zero disables source-vector storage while retaining the small potential and XC caches.\n"
			"Cache entries are released after each representative q point.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{
		pl.get(e.pertInfo.phononSourceCacheMiB,128.,"source cache size in MiB",true);
		if(!std::isfinite(e.pertInfo.phononSourceCacheMiB) || e.pertInfo.phononSourceCacheMiB<0.)
			throw string("source cache size must be finite and nonnegative");
	}
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%g",e.pertInfo.phononSourceCacheMiB); }
} commandPhononDFPTSourceCacheSize;

struct CommandPhononDFPTBalanceQGroups : public Command
{
	CommandPhononDFPTBalanceQGroups() : Command("phonon-dfpt-balance-q-groups", "jdftx/Perturb")
	{
		format="<yes|no>";
		comments="Balance q groups using displacement counts, offset-state work and export orbit sizes (default yes).\n"
			"Accounts for unequal group sizes and keeps all displacements at a q together.\n"
			"Set no to assign representatives round-robin.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{ pl.get(e.pertInfo.phononBalanceQGroups,true,boolMap,"balance q groups",true); }
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%s",boolMap.getString(e.pertInfo.phononBalanceQGroups)); }
} commandPhononDFPTBalanceQGroups;

struct CommandPhononDFPTKSymmetries : public Command
{
	CommandPhononDFPTKSymmetries() : Command("phonon-dfpt-k-symmetries", "jdftx/Perturb")
	{
		format="<yes|no>";
		comments="Reduce electronic response k points using the symmetry subgroup of each displacement (default no).\n"
			"Uses translation-free operations preserving q exactly and the displaced atom, with direction character +/-1.\n"
			"Symmetrizes response densities and reconstructs wavefunction responses with band sewing matrices.\n"
			"Unsupported operations retain the full response mesh; electron-phonon contractions always use full saved states.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{ pl.get(e.pertInfo.phononReduceK,false,boolMap,"reduce response k mesh",true); }
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%s",boolMap.getString(e.pertInfo.phononReduceK)); }
} commandPhononDFPTKSymmetries;

struct CommandPhononDFPTSolver : public Command
{
	CommandPhononDFPTSolver() : Command("phonon-dfpt-solver", "jdftx/Perturb")
	{
		format="<coupled|nested>";
		comments="Choose coupled response minimization or fixed-potential inner solves with outer potential mixing.\n"
			"Default coupled; nested speed depends on the material and mesh.\n"
			"Nested solves adapt inner accuracy to the outer residual and reuse the previous response.\n"
			"Both paths verify the same final coupled response residual.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{
		string solver; pl.get(solver,string("coupled"),"phonon response solver",true);
		if(solver!="coupled" && solver!="nested") throw string("solver must be coupled or nested");
		e.pertInfo.phononNestedSolver=solver=="nested";
	}
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%s",e.pertInfo.phononNestedSolver ? "nested" : "coupled"); }
} commandPhononDFPTSolver;

struct CommandPhononDFPTMixing : public Command
{
	CommandPhononDFPTMixing() : Command("phonon-dfpt-mixing", "jdftx/Perturb")
	{
		format="<fraction> <history> <iterations>";
		comments="Outer response potential mixing: fraction (0,1], history length, and maximum iterations.\n"
			"Defaults: 0.4 8 80. Used only by phonon-dfpt-solver nested.";
		require("phonon-dfpt");
	}
	void process(ParamList& pl,Everything& e)
	{
		auto& p=e.pertInfo;
		pl.get(p.phononMixFraction,0.4,"mixing fraction",true);
		pl.get(p.phononMixHistory,8,"mixing history",true);
		pl.get(p.phononMixIterations,80,"outer iterations",true);
		if(!std::isfinite(p.phononMixFraction) || p.phononMixFraction<=0. || p.phononMixFraction>1.)
			throw string("mixing fraction must be finite and in (0,1]");
		if(p.phononMixHistory<1 || p.phononMixIterations<1) throw string("history and iterations must be positive");
	}
	void printStatus(Everything& e,int iRep)
	{ auto& p=e.pertInfo; logPrintf("%g %d %d",p.phononMixFraction,p.phononMixHistory,p.phononMixIterations); }
} commandPhononDFPTMixing;

struct CommandPhononDFPTInitialState : public Command
{
	CommandPhononDFPTInitialState() : Command("phonon-dfpt-initial-state", "jdftx/Perturb")
	{
		format="<scf-input> <state-pattern>";
		comments="Expand a symmetry-reduced ground-state checkpoint onto the full DFPT electronic mesh.\n"
			"scf-input specifies the geometry, pseudopotentials, basis and symmetries used to write the checkpoint.\n"
			"state-pattern must contain $VAR, e.g. scf.$VAR; wavefunctions and fillings are read.\n"
			"Both calculations must have identical geometry, cutoff, bands, smearing and compatible k meshes.\n"
			"The DFPT input still uses symmetries none. No full-BZ SCF calculation is required.";
		require("phonon-dfpt"); forbid("initial-state"); forbid("wavefunction"); forbid("elec-initial-fillings");
	}
	void process(ParamList& pl,Everything& e)
	{
		pl.get(e.pertInfo.phononInitialInput,string(),"SCF input",true);
		pl.get(e.pertInfo.phononInitialState,string(),"state pattern",true);
		if(e.pertInfo.phononInitialState.find("$VAR")==string::npos) throw string("state pattern must contain $VAR");
	}
	void printStatus(Everything& e,int iRep)
	{ logPrintf("%s %s",e.pertInfo.phononInitialInput.c_str(),e.pertInfo.phononInitialState.c_str()); }
} commandPhononDFPTInitialState;
