// Numerical variational checks, including reciprocal-boundary links and MPI ownership.
#include <electronic/Everything.h>
#include <electronic/BerryPhase.h>
#include <electronic/ElecMinimizer.h>
#include <electronic/SpeciesInfo.h>
#include <core/Coulomb.h>
#include <commands/parser.h>
#include <cmath>

namespace {
void close(double actual,double expected,double tolerance,const char* label)
{
	logPrintf("%s: actual %.12e expected %.12e error %.3e\n",label,actual,expected,std::abs(actual-expected));
	if(std::abs(actual-expected)>tolerance) die("Berry regression failed: %s\n",label);
}
void check(Everything& e)
{
	matrix exchange = zeroes(2,2); exchange.set(0,1,1.); exchange.set(1,0,1.);
	close(det(exchange).real(),-1.,1e-15,"Odd determinant row permutation");
	close(det(exchange*exchange).real(),1.,1e-15,"Even determinant row permutation");
	auto original = e.eVars.C;
	std::vector<ColumnBundle> berryGradient;
	double coupling = e.berryPhase->evaluate(e,&berryGradient);
	ElecGradient grad, kg; grad.init(e); kg.init(e);
	e.eVars.elecEnergyAndGrad(e.ener,&grad,&kg);
	std::vector<ColumnBundle> direction(e.eInfo.nStates);
	double norm = 0.;
	for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++)
	{	direction[q] = original[q].similar(); randomize(direction[q]);
		direction[q] -= original[q]*(original[q]^O(direction[q]));
		norm += dotc(direction[q],direction[q]).real();
	}
	mpiWorld->allReduce(norm,MPIUtil::ReduceSum);
	double derivative = 0., fullDerivative = 0.;
	for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++)
	{	direction[q] *= 1./sqrt(norm);
		derivative += 2*dotc(berryGradient[q],direction[q]).real();
		fullDerivative += 2*dotc(grad.C[q],direction[q]).real();
	}
	mpiWorld->allReduce(derivative,MPIUtil::ReduceSum);
	mpiWorld->allReduce(fullDerivative,MPIUtil::ReduceSum);
	const double h = 1e-5;
	double energy[2], fullEnergy[2];
	for(int sign=0; sign<2; sign++)
	{	e.eVars.C = original;
		for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++)
		{	e.eVars.C[q] += (sign ? h : -h)*direction[q]; e.eVars.orthonormalize(q); }
		energy[sign] = e.berryPhase->evaluate(e);
		fullEnergy[sign] = e.eVars.elecEnergyAndGrad(e.ener);
	}
	close((energy[1]-energy[0])/(2*h),derivative,1e-8,"Berry directional derivative");
	close((fullEnergy[1]-fullEnergy[0])/(2*h),fullDerivative,2e-6,"Self-consistent DFT + field directional derivative");
	e.eVars.C = original;
	for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++)
	{	matrix generator = zeroes(e.eInfo.nBands,e.eInfo.nBands); randomize(generator);
		generator = dagger_symmetrize(generator);
		e.eVars.C[q] = original[q]*cis(generator);
	}
	close(e.berryPhase->evaluate(e),coupling,1e-12,"Independent occupied gauge rotations");
	e.eVars.C = original;
	vector3<> displacement(0.00013,-0.00021,0.00017);
	for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++) e.eVars.C[q] = translate(original[q],displacement);
	// Translate moves the electron density forward. Its negative physical charge
	// increases the electric enthalpy by N E.dr, including across BZ boundary links.
	close(e.berryPhase->evaluate(e)-coupling,e.eInfo.nElectrons*dot(e.berryPhase->field,e.gInfo.R*displacement),1e-12,"Electronic translation / boundary phase / physical sign");
	for(auto sp:e.iInfo.species) for(auto& pos:sp->atpos) pos += displacement;
	close(e.berryPhase->evaluate(e),coupling,1e-12,"Neutral joint electronic and ionic translation");
	for(auto sp:e.iInfo.species)
	{	for(auto& pos:sp->atpos) pos -= displacement; sp->sync_atpos(); }
	e.eVars.C = original;
	// An oversized trial must not reset the branch or terminate the process.
	vector3<> trialShift(0.3/e.eInfo.nBands,0.,0.);
	for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++) e.eVars.C[q] = translate(original[q],trialShift);
	if(std::isfinite(e.berryPhase->evaluate(e))) die("Berry regression failed: oversized phase trial accepted\n");
	e.eVars.C = original;
	close(e.berryPhase->evaluate(e),coupling,1e-12,"Rejected trial leaves accepted branch unchanged");
	// Walk a branch forward and backward through several polarization quanta.
	// Keeping the initial reference forever would produce a sawtooth enthalpy.
	vector3<> increment(0.1/e.eInfo.nBands,0.,0.);
	for(int sign: {1,-1}) for(int j=0; j<30; j++)
	{	for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++) e.eVars.C[q] = translate(e.eVars.C[q],double(sign)*increment);
		double distance = (sign==1 ? j+1 : 29-j)*increment[0];
		close(e.berryPhase->evaluate(e),coupling+e.eInfo.nElectrons*dot(e.berryPhase->field,e.gInfo.R*vector3<>(distance,0.,0.)),1e-11,"Continuous multi-quantum polarization branch");
		e.berryPhase->acceptPhase();
	}
	e.eVars.C = original;
	e.iInfo.update(e.ener);
	for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++) e.iInfo.project(e.eVars.C[q],e.eVars.VdagC[q]);
	e.eVars.elecEnergyAndGrad(e.ener,0,0,true);
	e.iInfo.ionicEnergyAndGrad();
	vector3<> force = e.gInfo.invRT*e.iInfo.forces[0][0];
	vector3<> position = e.iInfo.species[0]->atpos[0];
	vector3<> delta = e.gInfo.invR*vector3<>(h,0.,0.);
	for(int sign=0; sign<2; sign++)
	{	e.iInfo.species[0]->atpos[0] = position+(sign ? 1. : -1.)*delta;
		e.iInfo.species[0]->sync_atpos(); e.iInfo.update(e.ener);
		for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++) e.iInfo.project(e.eVars.C[q],e.eVars.VdagC[q]);
		fullEnergy[sign] = e.eVars.elecEnergyAndGrad(e.ener,0,0,true);
	}
	close(-(fullEnergy[1]-fullEnergy[0])/(2*h),force[0],2e-6,"Finite-field ionic force");
	e.iInfo.species[0]->atpos[0] = position; e.iInfo.species[0]->sync_atpos();
	// Differentiate all six symmetric strains at fixed normalized reduced
	// coefficients. This checks the field term and the complete DFT stress.
	const matrix3<> lattice = e.gInfo.R;
	const double volume = e.gInfo.detR;
	e.iInfo.computeStress = true;
	auto strainEnergy = [&](const matrix3<>& strain)
	{	e.gInfo.R = (matrix3<>(1,1,1)+strain)*lattice;
		e.gInfo.update(); e.updateSupercell();
		e.coulombParams.recreateCoulomb(e.gInfo,e.gInfoWfns,e.coulomb,e.coulombWfns);
		e.iInfo.update(e.ener);
		e.eVars.C = original;
		for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++)
		{	e.eVars.C[q] *= sqrt(volume/e.gInfo.detR);
			e.iInfo.project(e.eVars.C[q],e.eVars.VdagC[q]);
		}
		return e.eVars.elecEnergyAndGrad(e.ener,0,0,true);
	};
	for(vector3<int> branch: {vector3<int>(),vector3<int>(-1,2,-3)})
	{	e.berryPhase->polarizationBranch = branch;
		strainEnergy(matrix3<>());
		e.iInfo.ionicEnergyAndGrad();
		matrix3<> fullStrainGradient = volume*e.iInfo.stress;
		matrix3<> berryStrainGradient = e.berryPhase->latticeGradient(e);
		for(int a=0; a<3; a++) for(int b=a; b<3; b++)
		{	matrix3<> direction;
			direction(a,b) = direction(b,a) = a==b ? 1. : 0.5;
			for(int sign=0; sign<2; sign++)
			{	fullEnergy[sign] = strainEnergy((sign ? h : -h)*direction);
				energy[sign] = e.berryPhase->evaluate(e);
			}
			close((energy[1]-energy[0])/(2*h),berryStrainGradient(a,b),1e-8,"Fixed-Cartesian-field Berry stress / branch");
			close((fullEnergy[1]-fullEnergy[0])/(2*h),fullStrainGradient(a,b),1e-5,"Complete electric-enthalpy strain derivative");
		}
	}
	e.berryPhase->polarizationBranch = vector3<int>();
	strainEnergy(matrix3<>());
	logPrintf("Berry-phase variational regressions passed.\n");
}
}

int main(int argc,char** argv)
{
	InitParams params("Validate Berry-phase occupied-state electric enthalpy.");
	initSystemCmdline(argc,argv,params);
	{	Everything e; parse(readInputFile(params.inputFilename),e); e.setup(); check(e); }
	finalizeSystem();
}
