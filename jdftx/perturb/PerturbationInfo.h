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

#ifndef PERTURB_PERTURBATIONINFO_H_
#define PERTURB_PERTURBATIONINFO_H_

#include <vector>
#include <string>
#include <core/ScalarField.h>
#include <core/Coulomb.h>
#include <core/matrix.h>
#include <electronic/ColumnBundle.h>
#include <electronic/ElecInfo.h>
#include <perturb/LinearSolver.h>
#include <perturb/PerturbationSolver.h>

struct AtomicMode
{	unsigned int sp, at; //!< species and atom number
	vector3<> dirLattice; //!< Perturbation in lattice coords
	vector3<> dirCartesian; //!< Perturbation in Cartesian coords
};

class Perturbation {
public:
	Perturbation(const Everything& e);
private:
	const Everything& e;
	const class ElecVars& eVars;
	const ElecInfo& eInfo;
};

class AtomPerturbation : public Perturbation {
public:
	AtomPerturbation(unsigned int sp, unsigned int at, int iDir, const Everything& e);
	AtomPerturbation(unsigned int sp, unsigned int at, vector3<> dirCartesian, const Everything& e);
	
	bool sameAtom(const std::shared_ptr<AtomPerturbation> pert) { return pert->mode.sp == mode.sp && pert->mode.at == mode.at; };
	void init(const Everything &e, const class ElecVars& eVars, const ElecInfo& eInfo); //!< Initialize variables
	bool isUltrasoft(const class IonInfo& iInfo); //!< Does this atom use an ultrasoft potential
	
	AtomicMode mode; //!< Contains info about the atom and perturbation
	std::vector<ColumnBundle> Vatom, dVatom; //!< Cached single atom projectors and derivatives
	std::vector<matrix> VdagCatom, dVdagCatom; //!< Cached single atom projections and derivatives
	matrix E_nAug_datom; //!< Derivative of augmentation density w.r.t. atom perturbation
	ScalarFieldTilde Vlocps; //!< Local single atom pseudopotential contribution
	ScalarFieldArray dnatom; //!< Derivative of density w.r.t. atom perturbation
	std::vector<ColumnBundle> dCatom; //!< Derivative of wavefunctions
};

class VextPerturbation : public Perturbation {
public:
	VextPerturbation(const Everything& e) : Perturbation(e) {};
	std::vector<string> dVexternalFilename; //!< External potential filename (read in real space)
	ScalarFieldArray dVext; //!< Change in external potential
	complexScalarFieldArray dVextpq, dVextmq; //!< Change in external potential (incommensurate)
};

class RhoPerturbation : public Perturbation {
public:
	RhoPerturbation(const Everything& e) : Perturbation(e) {};
	std::vector<string> drhoExtFilename; //!< External charge filename (read in real space)
	ScalarFieldTilde drhoExt; //!< Change in charge potential
	complexScalarFieldTilde drhoExtpq, drhoExtmq; //!< Change in charge potential (incommensurate)
};

class ElectricFieldPerturbation : public Perturbation {
public:
	ElectricFieldPerturbation(const Everything &e) : Perturbation(e) {};
	vector3<> Efield;
	std::shared_ptr<Coulomb> C;
	ScalarField dVext;
};

class PerturbationInfo {
public:
	PerturbationInfo() : dVext(0), datom(0) {};
	~PerturbationInfo() {};
	void setup(const Everything &e, const class ElecVars &eVars);
	void read(const ElecInfo &eInfo, const Everything &e, std::vector<ColumnBundle>& C, const char *fname, const ElecInfo::ColumnBundleReadConversion* conversion=0) const;
	void setupkpoints(const Everything &e, const ElecInfo &eInfo);
	void initInc(std::vector<ColumnBundle>& Y, int nbundles, int ncols, const ElecInfo* eInfo); //!< Setup incommensurate ColumnBundles
	void checkSupportedFeatures(const Everything &e, const ElecInfo &eInfo);
	bool densityAugRequired(const Everything &e);
	
	void sampleCB (ColumnBundle C, std::string name);
	void sampleMat (matrix C, std::string name);
	void sampleField (ScalarField V, std::string name);
	void sampleField (ScalarFieldTilde V, std::string name);

	bool commensurate = true; //!< Is the perturbation lattice periodic
	bool offsetStatesReady = false; //!< Reset when offset bases are allocated
	bool fullMeshResponse = false; //!< Driver expands all electronic k points; orbital sources must not be crystal-symmetrized
	bool optimizeFrozenResponse = false; //!< KI driver guarantees frozen density throughout the solver lifetime
	bool oneSidedResponse = false; //!< Verified nonmagnetic full-mesh response uses time reversal to eliminate the minus branch
	bool kiSpinResponse = false; //!< Nonmagnetic KI resolves charge and longitudinal magnetization response
	bool kiVerbose = false; //!< Detailed logging for full-mesh KI; other perturbations keep their usual output
	bool magneticResponse = false; //!< Active scalar response represents m=n_up-n_down and V_m=(V_up-V_down)/2
	double frozenOrbitalCacheMiB = 0.; //!< Per-rank real-space occupied-orbital cache budget
	bool nestedResponse = false; //!< Enable nested screening for the full-mesh orbital driver
	bool independentBandCG = true; //!< Use band CG, with MINRES fallback, inside nested screening
	bool occupiedProjectorSupplied = false; //!< Driver supplies occupied-only Cinc projectors with fixed empty source bands
	bool groundStateReady = false; //!< Ground state has been minimized for the current response sequence
	vector3<int> phononGrid; //!< Nonzero entries enable primitive-cell DFPT phonons
	bool phononUseSymmetry = true; //!< Reduce q by space-group symmetry and time reversal
	bool phononReduceDisplacements = true; //!< Reconstruct displacement responses using the little group of q
	bool phononReuseOffsetStates = true; //!< Reuse existing occupied states when k +/- q lies on the k mesh
	bool phononElectronPhonon = false; //!< Export screened Bloch Hamiltonian derivatives at every q and displacement
	bool phononOptimize = true; //!< Fused phonon operators and bounded export caches (disable for reference comparisons)
	double phononEphCacheMiB = 512.; //!< Maximum cached real-space export wavefunctions per rank
	bool phononEphReciprocity = true; //!< Contract one member of each Hermitian k pair
	string phononInitialInput, phononInitialState; //!< Input and state pattern of a symmetry-reduced SCF restart
	bool phononIBZStorage = false; //!< Retain only reduced ground states; reconstruct active k batches
	std::shared_ptr<class DFPTGroundState> phononGroundState;
	int phononQGroups = 1; //!< Independent MPI groups for representative q points
	string phononCheckpoint; //!< Directory of atomic, completed-q response checkpoints
	string phononCheckpointInput; //!< Parsed physical input used to validate response checkpoints
	int phononQBatch = 0, phononQBatches = 1; //!< Optional 1-based representative batch
	bool phononMerge = false; //!< Require all checkpoints; export without response solves
	int phononKBatchSize = 0; //!< Stream fixed-potential response work by local electronic k points
	int phononKBatchStart = -1, phononKBatchStop = -1; //!< Internal active local k range
	bool inPhononKBatch(int k) const { return phononKBatchStart<0 || (k>=phononKBatchStart && k<phononKBatchStop); }
	bool phononCache = true; //!< Reuse frozen-ground-state and bare perturbation data within each q
	double phononSourceCacheMiB = 128.; //!< Maximum retained source-vector storage per MPI rank
	bool phononBalanceQGroups = true; //!< Assign representative q points by estimated work
	bool phononReduceK = false; //!< Reduce electronic response using symmetries of each displacement
	bool phononNestedSolver = false; //!< Fixed-potential inner response solves with outer potential mixing
	double phononMixFraction = 0.4;
	int phononMixHistory = 8, phononMixIterations = 80;
	double lastResponseResidual = 0.; //!< Residual of the most recent completed response
	bool testing = false; //!< Whether or not a FD test is being conducted
	SolverParams solverParams; //!< solver parameters

	std::shared_ptr<VextPerturbation> dVext;
	std::shared_ptr<AtomPerturbation> datom;
	std::shared_ptr<RhoPerturbation> drhoExt;
	std::shared_ptr<ElectricFieldPerturbation> dElectricField;
	
	vector3<> qvec; //!< Bloch wavevector of perturbation. Equal to zero if perturbation is commensurate
	
	std::shared_ptr<struct ElecInfo::ColumnBundleReadConversion> readConversion;
	std::vector<QuantumNumber> kplusq_vectors; //!< List of k+q vectors
	std::vector<QuantumNumber> kminusq_vectors; //!< List of k-q vectors
	std::vector<Basis> kplusq_basis; //!< List of k+q basis elements
	std::vector<Basis> kminusq_basis; //!< List of k-q basis elements
	
	string wfnsFilename; //!< Name of ground state wavefunctions

	/* Gradients and wavefunction shifts */
	PerturbationGradient dGradTau, dGradPsi;
	std::vector<ColumnBundle> dC, Cinc;
	std::vector<matrix> dU, dUmhalfatom, dHsub, dHsubatom, CdagdHC, CdagdHCatom;
	
	/* Intermediate scalar field derivs */
	ScalarFieldArray dn;
	ScalarField dnCoreA;
	complexScalarFieldArray dnpq, dnmq;
	ScalarFieldArray dVscloc, dVsclocTau;
	complexScalarFieldArray dVsclocpq, dVsclocmq;
	
	/* Cached quantities */
	std::vector<ColumnBundle> grad, HC, OC;
	ScalarField sigma_cached, e_nn_cached, e_sigma_cached, e_nsigma_cached, e_sigmasigma_cached; //LDA and GGA second derivs
	VectorField IDJn_cached;
	ScalarField spin_e_mm_cached, spin_e_mu_cached, spin_e_uu_cached, spin_e_v_cached;
	
	std::vector<matrix> E_nAug_cached, E_nAug_dVsclocpsi, E_nAug_dVscloctau;
};

#endif /* PERTURB_PERTURBATIONINFO_H_ */
