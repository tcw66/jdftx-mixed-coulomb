/* Primitive-cell DFPT phonons and validation. GPL v3 or later; part of JDFTx. */
#include <perturb/DFPTPhonon.h>
#include <perturb/LinearSolver.h>
#include <core/Util.h>
#include <core/GridInfo.h>
#include <core/Operators.h>
#include <core/WignerSeitz.h>

namespace {
void require(bool condition,const char* message) { if(!condition) die("DFPT test failed: %s\n",message); }
// Exercise the actual templated solver on a nontrivial symmetric indefinite system.
struct Vec { double x=0.,y=0.; Vec& operator*=(double a) { x*=a;y*=a;return *this; } };
Vec clone(const Vec& a) { return a; }
void axpy(double a,const Vec& x,Vec& y) { y.x+=a*x.x;y.y+=a*x.y; }
double dot(const Vec& a,const Vec& b) { return a.x*b.x+a.y*b.y; }
struct Solver : LinearSolvableIndefinite<Vec> {
	void hessian(Vec& Av,const Vec& v) override { double x=v.x,y=v.y;Av.x=2*x+y;Av.y=x-y; }
	void precondition(Vec&) override {} void Ksqrt(Vec&) override {}
};
}

int main(int argc,char** argv)
{
	initSystem(argc,argv);
	{
		// Alternating cheap/expensive q points defeat round-robin assignment.
		std::vector<double> costs{1.,9.,1.,9.};
		auto groups=balanceDFPTQGroups(costs,{1,1});
		std::vector<double> load(2,0.);
		for(size_t i=0;i<costs.size();i++) load[groups[i]]+=costs[i];
		require(load[0]==10. && load[1]==10.,"q group cost balancing");
		require(groups==balanceDFPTQGroups(costs,{1,1}),"deterministic q group schedule");
		groups=balanceDFPTQGroups({9.,1.},{1,2,1});
		require(groups[0]==1,"q scheduler respects unequal group sizes");
		require(balanceDFPTQGroups({}, {1,1}).empty(),"empty q schedule");
		require(balanceDFPTQGroups({1.}, {1,1,1}).size()==1,"idle q groups");
		// An anisotropic FCC cell welds coincident vertices into self-loop
		// edges. Repeat with varying allocation order to catch stale iterators.
		for(int repeat=0;repeat<24;repeat++) {
			matrix3<> R(0.,3.825,7.65,3.825,0.,7.65,3.825,3.825,0.);
			WignerSeitz ws(R);
			ws.checkGraph();
			require(ws.inRadius()>0.,"anisotropic FCC Wigner-Seitz geometry");
		}
		Solver s; SolverParams p; p.nIterations=10;p.residualTol=1e-12;p.residualDiffThreshold=0.;
		s.solve(Vec(),p); require(std::isfinite(s.state.x) && s.state.x==0.,"zero MINRES source");
	p.algorithm=SolverParams::CG;
	s.solve(Vec(),p); require(std::isfinite(s.state.x) && s.state.x==0.,"zero CG source");
	p.algorithm=SolverParams::MINRES;
		Vec b;b.x=1.;b.y=2.;s.solve(b,p);
		Vec r;s.hessian(r,s.state);axpy(1.,b,r);
		require(dot(r,r)<1e-22,"indefinite MINRES residual / Lanczos termination");
		GridInfo g;g.R=matrix3<>(8.,9.,10.);g.S=vector3<int>(8,8,8);g.initialize();
		CoulombParams cp;cp.geometry=CoulombParams::Periodic;auto coulomb=cp.createCoulomb(g);
		std::vector<Atom> atoms {Atom(1.,vector3<>(0.12,0.21,0.31)),Atom(2.,vector3<>(0.43,0.55,0.69))};
		const matrix analytic=ionicForceMatrix(g.R,atoms,vector3<>());
		matrix numerical=zeroes(6,6);
		const double h=1e-4;
		for(int a=0;a<6;a++) {
			auto plus=atoms,minus=atoms;vector3<> d;d[a%3]=h;
			plus[a/3].pos+=g.invR*d;minus[a/3].pos-=g.invR*d;
			coulomb->energyAndGrad(plus);coulomb->energyAndGrad(minus);
			for(int b=0;b<6;b++) numerical.set(b,a,(-(g.invRT*(plus[b/3].force-minus[b/3].force))/(2*h))[b%3]);
		}
		require(nrm2(analytic-numerical)/nrm2(numerical)<1e-7,"Gamma Ewald Hessian versus force differences");
		matrix translation=zeroes(6,3);
		for(int a=0;a<6;a++)translation.set(a,a%3,1.);
		require(nrm2(analytic*translation)<1e-12,"ionic acoustic sum rule");
		vector3<> q(0.31,-0.23,0.17);
		matrix finite=ionicForceMatrix(g.R,atoms,q),negative=ionicForceMatrix(g.R,atoms,-q);
		require(nrm2(finite-dagger(finite))<1e-12,"finite-q Hermiticity");
		require(nrm2(finite-conj(negative))<1e-12,"finite-q reality");
		require(nrm2(finite-ionicForceMatrix(g.R,atoms,q+vector3<>(1,0,0)))<1e-12,"reciprocal periodicity");
		// A half-cell translation swaps atoms but shifts only the second image
		// into the next cell. Omitting its Bloch phase must fail at finite q.
		std::vector<Atom> chain {Atom(1.,vector3<>(0,0,0)),Atom(1.,vector3<>(0.5,0,0))};
		matrix chainF=ionicForceMatrix(g.R,chain,q);
		std::vector<int> swap {1,0},identity {0,1};
		std::vector<vector3<int>> shift {vector3<int>(),vector3<int>(1,0,0)};
		matrix transformed=transformDFPTForceMatrix(chainF,matrix3<>(1,1,1),q,swap,shift);
		require(nrm2(chainF-transformed)<1e-12,"atom permutation with cell-translation phases");
		shift[1]=vector3<int>(-1,0,0);
		vector3<> reflectedQ(-q[0],q[1],q[2]);
		transformed=transformDFPTForceMatrix(chainF,matrix3<>(-1,1,1),reflectedQ,identity,shift);
		require(nrm2(transformed-ionicForceMatrix(g.R,chain,reflectedQ))<1e-12,"rotated finite-q matrix with cell phases");
		// Rational q agrees with a replicated-cell Ewald force derivative.
		vector3<> boundary(0.5,0,0);
		matrix F=ionicForceMatrix(g.R,atoms,boundary);
		GridInfo sup;sup.R=g.R*Diag(vector3<>(2,1,1));sup.S=vector3<int>(16,8,8);sup.initialize();
		auto coulombSup=cp.createCoulomb(sup);
		std::vector<Atom> extended;
		for(int cell=0;cell<2;cell++)for(const Atom& a:atoms)
			extended.emplace_back(a.Z,inv(sup.R)*g.R*(a.pos+vector3<>(cell,0,0)));
		numerical=zeroes(6,6);
		for(int a=0;a<6;a++) {
			auto plus=extended,minus=extended;
			for(int cell=0;cell<2;cell++) {
				vector3<> d;d[a%3]=h*(cell ? -1.:1.);
				plus[2*cell+a/3].pos+=sup.invR*d;minus[2*cell+a/3].pos-=sup.invR*d;
			}
			coulombSup->energyAndGrad(plus);coulombSup->energyAndGrad(minus);
			for(int b=0;b<6;b++) numerical.set(b,a,(-(sup.invRT*(plus[b/3].force-minus[b/3].force))/(2*h))[b%3]);
		}
		require(nrm2(F-numerical)/nrm2(numerical)<1e-7,"finite-q Ewald Hessian versus replicated-cell forces");
		// Compare analytic slab Hessians against the independent 2D Ewald
		// forces for every truncation direction, including buckled atoms.
		for(int dir=0;dir<3;dir++) for(bool embedded:{false,true}) {
			cp.geometry=CoulombParams::Slab; cp.iDir=dir; cp.ionMargin=0.5; cp.embed=embedded; cp.embedCenter=vector3<>(); if(embedded) cp.embedCenter[dir]=0.13;
			auto slab=cp.createCoulomb(g);
			matrix slabGamma=ionicForceMatrix(g.R,atoms,vector3<>(),cp);
			numerical=zeroes(6,6);
			for(int a=0;a<6;a++) {
				auto plus=atoms,minus=atoms; vector3<> d; d[a%3]=h;
				plus[a/3].pos+=g.invR*d; minus[a/3].pos-=g.invR*d;
				slab->energyAndGrad(plus); slab->energyAndGrad(minus);
				for(int b=0;b<6;b++) numerical.set(b,a,(-(g.invRT*(plus[b/3].force-minus[b/3].force))/(2*h))[b%3]);
			}
			require(nrm2(slabGamma-numerical)/nrm2(numerical)<1e-7,"slab Gamma Hessian versus Ewald forces");
			require(nrm2(slabGamma*translation)<1e-12,"slab ionic acoustic sum rule");
			vector3<> planar=q; planar[dir]=0.;
			matrix slabQ=ionicForceMatrix(g.R,atoms,planar,cp);
			require(nrm2(slabQ-dagger(slabQ))<1e-12,"slab finite-q Hermiticity");
			require(nrm2(slabQ-conj(ionicForceMatrix(g.R,atoms,-planar,cp)))<1e-12,"slab finite-q reality");
			int axis=(dir+1)%3;
			vector3<> boundary2; boundary2[axis]=0.5;
			vector3<> scale(1,1,1); scale[axis]=2.;
			GridInfo slabSup; slabSup.R=g.R*Diag(scale); slabSup.S=g.S; slabSup.S[axis]*=2; slabSup.initialize();
			auto slabCoulombSup=cp.createCoulomb(slabSup);
			extended.clear();
			for(int cell=0;cell<2;cell++) for(const Atom& atom:atoms) {
				vector3<> offset; offset[axis]=cell;
				extended.emplace_back(atom.Z,slabSup.invR*g.R*(atom.pos+offset));
			}
			numerical=zeroes(6,6);
			for(int a=0;a<6;a++) {
				auto plus=extended,minus=extended;
				for(int cell=0;cell<2;cell++) {
					vector3<> d; d[a%3]=h*(cell ? -1.:1.);
					plus[2*cell+a/3].pos+=slabSup.invR*d; minus[2*cell+a/3].pos-=slabSup.invR*d;
				}
				slabCoulombSup->energyAndGrad(plus); slabCoulombSup->energyAndGrad(minus);
				for(int b=0;b<6;b++) numerical.set(b,a,(-(slabSup.invRT*(plus[b/3].force-minus[b/3].force))/(2*h))[b%3]);
			}
			matrix slabBoundary=ionicForceMatrix(g.R,atoms,boundary2,cp);
			require(nrm2(slabBoundary-numerical)/nrm2(numerical)<1e-7,"slab finite-q Hessian versus replicated Ewald forces");
			ScalarField density=ScalarFieldData::alloc(g);
			for(int i=0;i<g.nr;i++) density->data()[i]=sin(0.31*i)+0.1;
			complexScalarFieldTilde full=J(Complex(density));
			for(auto mode:{Coulomb::PointChargeNone,Coulomb::PointChargeRight,Coulomb::PointChargeLeft})
				require(nrm2(I(slab->applyBloch(full,vector3<>(),mode))-Complex(I((*slab)(J(density),mode))))<1e-10,
					"slab Bloch Gamma kernel and point-charge range separation agree with ground-state Coulomb");
		}
		logPrintf("DFPT solver and Ewald checks passed.\n");
	}
	finalizeSystem();
	return 0;
}
