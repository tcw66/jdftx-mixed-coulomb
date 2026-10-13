/* Charge/magnetic finite-q kernels versus full XC potential differences. */
#include <electronic/Everything.h>
#include <perturb/PerturbationSolver.h>
#include <core/CoulombPeriodic.h>
#include <core/Operators.h>
#include <core/Util.h>
#include <core/ScalarFieldIO.h>

namespace {
void check(ExCorrType functional,int profile,int fold,bool magnetic)
{
	Everything e;
	GridInfo& g=e.gInfo;
	g.R=matrix3<>(8.,0.,0.,0.,7.,0.,0.,0.,6.);
	if(profile==3) g.R=matrix3<>(8.,0.,0.,0.,8.,0.,0.,0.,8.);
	g.S=profile==3 ? vector3<int>(24,24,24) : vector3<int>(32,32,32); g.initialize();
	e.exCorr=ExCorr(functional); e.exCorr.setup(e);
	e.coulombParams.geometry=CoulombParams::Periodic;
	e.coulomb=std::make_shared<CoulombPeriodic>(g,e.coulombParams);
	auto n=ScalarFieldData::alloc(g);
	auto source=complexScalarFieldData::alloc(g);
	for(int x=0;x<g.S[0];x++) for(int y=0;y<g.S[1];y++) for(int z=0;z<g.S[2];z++) {
		int i=(x*g.S[1]+y)*g.S[2]+z;
		double a=2*M_PI*x/g.S[0],b=2*M_PI*y/g.S[1],c=2*M_PI*z/g.S[2];
		double amplitude=profile==1 ? 0. : profile==2 ? 1e-8 : 1.;
		n->data()[i]=.08*(1.+amplitude*(.45*cos(a)+.2*sin(b)+.15*cos(c)));
		source->data()[i]=.01*complex(cos(a),fold==1 ? 0. : .4*sin(b))*(1.+.2*cos(c));
	}
	if(profile==3) {
		loadRawBinary(n,getenv("JDFTX_TEST_KERNEL_DENSITY"));
		auto rho=ScalarFieldData::alloc(g);
		loadRawBinary(rho,getenv("JDFTX_TEST_KERNEL_SOURCE"));
		source=Complex(rho);
	}
	auto& p=e.pertInfo;
	p.commensurate=false; p.fullMeshResponse=true; p.kiSpinResponse=true; p.magneticResponse=magnetic;
	auto nG=J(n);
	nullToZero(p.sigma_cached,g); initZero(p.sigma_cached);
	for(int d=0;d<3;d++) {
		p.IDJn_cached[d]=I(D(nG,d));
		p.sigma_cached+=p.IDJn_cached[d]*p.IDJn_cached[d];
	}
	e.exCorr.getSpinSecondDerivatives(n,p.sigma_cached,
		p.spin_e_mm_cached,p.spin_e_mu_cached,p.spin_e_uu_cached,p.spin_e_v_cached);
	e.exCorr.getSecondDerivatives(n,p.e_nn_cached,p.e_sigma_cached,
		p.e_nsigma_cached,p.e_sigmasigma_cached,1e-9,&p.sigma_cached);
	for(const auto& field:{p.e_nn_cached,p.e_sigma_cached,p.e_nsigma_cached,p.e_sigmasigma_cached})
		if(field && !std::isfinite(nrm2(field))) die("Nonfinite charge XC Hessian.\n");
	vector3<> q(0.,0.,fold==1 ? 0. : 1./fold);
	// A real modulation in a commensurate supercell supplies an independent
	// finite-q reference using the complete spin-polarized XC evaluator.
	GridInfo super; super.R=g.R; super.R(2,2)*=fold;
	super.S=g.S; super.S[2]*=fold; super.initialize();
	auto reference=complexScalarFieldData::alloc(g); initZero(reference);
	const double h=profile==3 ? (getenv("JDFTX_TEST_KERNEL_STEP") ? atof(getenv("JDFTX_TEST_KERNEL_STEP")) : 1e-6) : 1e-4;
	ScalarFieldArray potentials[2];
	for(int sign=0;sign<2;sign++) {
		ScalarFieldArray density(magnetic ? 2 : 1);
		nullToZero(density,super);
		for(int x=0;x<super.S[0];x++) for(int y=0;y<super.S[1];y++) for(int z=0;z<super.S[2];z++) {
			int index=(x*super.S[1]+y)*super.S[2]+z;
			int i=(x*g.S[1]+y)*g.S[2]+z%g.S[2];
			complex phase=cis(2*M_PI*q[2]*z/g.S[2]);
			double dm=(sign ? h : -h)*(phase*source->data()[i]).real();
			if(magnetic) {
				density[0]->data()[index]=.5*(n->data()[i]+dm);
				density[1]->data()[index]=.5*(n->data()[i]-dm);
			} else density[0]->data()[index]=n->data()[i]+dm;
		}
		e.exCorr(density,&potentials[sign]);
	}
	for(int x=0;x<super.S[0];x++) for(int y=0;y<super.S[1];y++) for(int z=0;z<super.S[2];z++) {
		int index=(x*super.S[1]+y)*super.S[2]+z;
		int i=(x*g.S[1]+y)*g.S[2]+z%g.S[2];
		double vm=magnetic ? (potentials[1][0]->data()[index]-potentials[1][1]->data()[index]
			-potentials[0][0]->data()[index]+potentials[0][1]->data()[index])/(4*h)
			: (potentials[1][0]->data()[index]-potentials[0][0]->data()[index])/(2*h);
		reference->data()[i]+=(fold==1 ? 1. : 2./fold)*cis(-2*M_PI*q[2]*z/g.S[2])*vm;
	}
	PerturbationSolver solver(e);
	for(bool optimized:{false,true}) {
		p.optimizeFrozenResponse=optimized;
		complexScalarFieldArray actual(1);
		solver.getdVsclocPsi(complexScalarFieldArray(1,source),actual,&q);
		actual[0]*=(1./g.dV);
		if(!magnetic) actual[0]-=I(e.coulomb->applyBloch(J(source),q));
		double error=nrm2(actual[0]-reference)/nrm2(reference);
		if(profile==3) logPrintf("Orbital XC curvature: actual %.15g, finite difference %.15g, h %.6g.\n",
			g.dV*dot(source,actual[0]).real(),g.dV*dot(source,reference).real(),h);
		logPrintf("%s XC kernel: %s, %s density, qz=%g, optimized=%d: relative error %.3g\n",
			magnetic ? "Magnetic" : "Charge",functional==ExCorrGGA_PBE ? "PBE" : "LDA-PZ",
			profile==1 ? "uniform" : profile==2 ? "nearly uniform" : "inhomogeneous",q[2],optimized,error);
		if(!std::isfinite(error) || error>3e-6) die("XC kernel finite-difference regression failed.\n");
	}

}
}

int main(int argc,char** argv)
{
	initSystem(argc,argv);
	if(getenv("JDFTX_TEST_KERNEL_DENSITY") && getenv("JDFTX_TEST_KERNEL_SOURCE")) {
		for(bool magnetic:{false,true}) check(ExCorrGGA_PBE,3,1,magnetic);
		finalizeSystem(); return 0;
	}
	for(auto functional:{ExCorrLDA_PZ,ExCorrGGA_PBE})
		for(int profile:{0,1,2}) for(int fold:{1,3})
			for(bool magnetic:{false,true}) check(functional,profile,fold,magnetic);
	logPrintf("Charge and magnetic XC kernel finite-difference regression passed.\n");
	finalizeSystem(); return 0;
}
