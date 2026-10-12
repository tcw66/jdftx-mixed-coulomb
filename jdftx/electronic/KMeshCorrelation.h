/* Exact uniform-k-mesh correlations. GPL v3 or later; part of JDFTx. */
#ifndef JDFTX_KMESHCORRELATION_H
#define JDFTX_KMESHCORRELATION_H
#include <core/ScalarField.h>
#include <core/matrix.h>
#include <memory>

// Evaluate orbital sources and bare conduction contractions by cyclic FFTs
// across the k mesh. Real-space grid tiles bound the temporary workspace.
class KMeshCorrelation
{
	struct Impl;
	std::unique_ptr<Impl> impl;
public:
	KMeshCorrelation(const GridInfo& grid,const vector3<int>& folding,
		const std::vector<vector3<>>& mesh,const std::vector<vector3<>>& qmesh,
		const std::vector<std::vector<complexScalarField>>& orbital,double cacheMiB);
	~KMeshCorrelation();
	bool enabled() const { return bool(impl); }
	complexScalarField source(int n,int iq) const;
	// A null potential denotes an unowned q point in a response group.
	void accumulateConduction(int n,const std::vector<complexScalarField>& potential,
		const std::vector<int>& emptyCenters,std::vector<matrix>& correction);
};

#ifdef GPU_ENABLED
void kMeshPackOrbital_gpu(int nk,int centers,int n,int start,int count,int tile,
	const vector3<int>& shape,const int* slotToK,const vector3<>* mesh,
	const complex* const* orbital,complex* values);
void kMeshPower_gpu(size_t count,complex* values);
void kMeshStoreSource_gpu(int nk,int nr,int start,int count,const vector3<int>& shape,
	const int* qToSlot,const vector3<>* qmesh,const complex* values,complex* source);
void kMeshPackPotential_gpu(int nk,int start,int count,int tile,const vector3<int>& shape,
	const int* slotToQ,const vector3<>* qmesh,const complex* const* potential,complex* values);
void kMeshMultiply_gpu(size_t count,const complex* a,complex* b);
void kMeshConduction_gpu(int nk,int centers,int n,int start,int count,const vector3<int>& shape,
	const int* kToSlot,const vector3<>* mesh,const complex* const* orbital,
	int emptyCount,const int* emptyCenters,const complex* weighted,complex* const* correction,double scale);
#endif
#endif
