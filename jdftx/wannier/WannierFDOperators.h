#ifndef JDFTX_WANNIER_FDOPERATORS_H
#define JDFTX_WANNIER_FDOPERATORS_H
#include <core/matrix.h>

// Pack overlap diagonals without moving the overlap off the device. Read the
// packed matrix on the host only after all edges have been queued.
void wannierDiagonal(const matrix& overlap, matrix& packed, size_t edge);

// Accumulate both endpoint derivatives directly, without scaled/sliced matrix
// temporaries. left is nIn_i x nCenters and right is nCenters x nIn_j;
// the same full overlap matrix may be supplied for both arguments.
void wannierOverlapGradient(const matrix& left, const matrix& right,
    const matrix& derivatives, size_t edge, matrix& gi, matrix& gj);
// Pack dM_nn / i in the U1 frame for a matrix-free center Jacobian product.
void wannierCenterVariation(const matrix& left, const matrix& right,
    const matrix& ai, const matrix& aj, matrix& packed, size_t edge);
#endif
