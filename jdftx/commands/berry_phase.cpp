#include <commands/command.h>
#include <electronic/Everything.h>
#include <electronic/BerryPhase.h>

struct CommandBerryPhaseField : public Command
{
	CommandBerryPhaseField() : Command("berry-phase-field","jdftx/Electronic/Parameters")
	{	format = "<Ex> <Ey> <Ez>";
		comments = "Minimize the bulk electric enthalpy EDFT - volume E.P using occupied-state Berry phases.\n"
			"E is the total macroscopic Cartesian field in atomic units, with the physical positive-ion sign convention.\n"
			"Requires a neutral insulator, only filled bands, norm-conserving pseudopotentials, periodic Coulomb,\n"
			"and a uniform kpoint-folding mesh. Disables spatial and k-inversion symmetry reduction.\n"
			"Use electronic-minimize (no electronic-scf at nonzero E); Hartree and XC update at every step.\n"
			"Initialize weak-field runs from the same converged zero-field wavefunctions.\n"
			"Ionic and lattice relaxation and stress are supported at fixed Cartesian E.\n"
			"For cell relaxation, select a physical polarization branch with berry-phase-branch.\n"
			"Writes $VAR=polarization at End, including electronic/ionic polarization and enthalpy.\n"
			"Use E=0 for zero-field Berry polarization. Wannier analysis is independent postprocessing.";
		forbid("electric-field");
	}
	void process(ParamList& pl, Everything& e)
	{	e.berryPhase = std::make_shared<BerryPhase>();
		for(int a=0; a<3; a++)
		{	pl.get(e.berryPhase->field[a],0.,a==0 ? "Ex" : a==1 ? "Ey" : "Ez",true);
			if(!std::isfinite(e.berryPhase->field[a])) throw string("Berry electric field must be finite");
		}
	}
	void printStatus(Everything& e,int)
	{	for(int a=0; a<3; a++) logPrintf("%g ",e.berryPhase->field[a]); }
} commandBerryPhaseField;

struct CommandBerryPhaseBranch : public Command
{
	CommandBerryPhaseBranch() : Command("berry-phase-branch","jdftx/Electronic/Parameters")
	{	format = "<n1> <n2> <n3>";
		comments = "Shift the total polarization by sum_i n_i a_i / volume, in integer electronic-charge quanta.\n"
			"Default: 0 0 0. The offset follows the lattice during strain at fixed Cartesian field.\n"
			"Select the branch corresponding to the physical surface-charge boundary condition before relaxing the cell.\n"
			"For the supplied diamond-silicon geometry, -1 -1 -1 gives the nonpolar branch.";
		require("berry-phase-field");
	}
	void process(ParamList& pl, Everything& e)
	{	for(int d=0; d<3; d++) pl.get(e.berryPhase->polarizationBranch[d],0,d==0 ? "n1" : d==1 ? "n2" : "n3",true); }
	void printStatus(Everything& e,int)
	{	for(int d=0; d<3; d++) logPrintf("%d ",e.berryPhase->polarizationBranch[d]); }
} commandBerryPhaseBranch;
