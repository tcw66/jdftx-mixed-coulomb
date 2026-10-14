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
			"Internal-ion relaxation at fixed strain is supported; stress and lattice relaxation are not.\n"
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
