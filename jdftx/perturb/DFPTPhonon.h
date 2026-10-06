/* Primitive-cell DFPT phonons. GPL v3 or later; part of JDFTx. */
#ifndef JDFTX_PERTURB_DFPTPHONON_H
#define JDFTX_PERTURB_DFPTPHONON_H

#include <core/matrix.h>
#include <core/Coulomb.h>

class Everything;
class MPIUtil;
// Cell-phase convention: u_s(R) = u_s(q) exp(2*pi*i*q.R).
// Returns the unweighted ionic force matrix in Cartesian coordinates.
matrix ionicForceMatrix(const matrix3<>& R, const std::vector<Atom>& atoms, const vector3<>& q);
// Rotate and permute Cartesian displacement components in the cell-phase gauge.
// cellShift[s] is the integer cell containing the symmetry image of atom s.
matrix transformDFPTForceMatrix(const matrix& F,const matrix3<>& rotation,const vector3<>& targetQ,
	const std::vector<int>& atomMap,const std::vector<vector3<int>>& cellShift);
void calculateDFPTPhonons(Everything& e,const MPIUtil* parent=nullptr);
void loadSymmetryReducedDFPTState(Everything& e,bool dryRun);
// Deterministic longest-job-first scheduling, accounting for group capacities.
std::vector<int> balanceDFPTQGroups(const std::vector<double>& costs,const std::vector<int>& capacities);

#endif
