/* Batched scalar-wavefunction FFT kernels. GPL v3 or later; part of JDFTx. */
#ifndef JDFTX_COLUMNBUNDLEFFT_H
#define JDFTX_COLUMNBUNDLEFFT_H
#include <electronic/ColumnBundle.h>
// Pair density with a frozen real-space operand, using the same quadrature as diagouterI.
complexScalarFieldArray diagouterI_cached(const ColumnBundle& response,
	const ManagedArray<complex>& ground,const ManagedArray<double>& fillings,bool groundFirst,
	int nDensities=1,const GridInfo* gInfoOut=nullptr);
// Reduce all columns with a single GPU-to-host transfer; CPU uses diagDot.
std::vector<diagMatrix> batchedDiagDot(const std::vector<const ColumnBundle*>& x,
	const std::vector<const ColumnBundle*>& y);
// Independent-band CG updates, fused across all local response blocks.
void batchedCGStep(const std::vector<ColumnBundle*>& solution,const std::vector<ColumnBundle*>& residual,
	const std::vector<const ColumnBundle*>& direction,const std::vector<const ColumnBundle*>& applied,
	const std::vector<diagMatrix>& alpha);
void batchedCGDirection(const std::vector<ColumnBundle*>& direction,
	const std::vector<const ColumnBundle*>& preconditioned,const std::vector<diagMatrix>& beta);
// In-place X -= C(C^O(X)) for norm-conserving scalar bundles. Group equal
// dimensions for CUDA GEMM; coefficient workspace and pointer tables are reused.
void projectOutBatched(const std::vector<const ColumnBundle*>& projectors,
	const std::vector<ColumnBundle*>& values);
#ifdef GPU_ENABLED
struct ColumnDotDescriptor { const complex *x,*y; size_t count; };
struct ColumnFFTDescriptor {
	const complex* input; complex* output;
	const int *inputIndices,*outputIndices;
	int inputBasis,outputBasis;
};
struct ColumnCGStepDescriptor {
	complex *solution,*residual; const complex *direction,*applied;
	size_t count; double alpha;
};
struct ColumnUpdateDescriptor {
	complex* output; const complex* input; size_t count;
	double outputScale,inputScale;
};
void columnDot_gpu(int count,const ColumnDotDescriptor* pairs,double* partial,double* result);
void columnDiagDot_gpu(int count,const ColumnDotDescriptor* pairs,double* result);
void columnCGStep_gpu(int count,size_t maximum,const ColumnCGStepDescriptor* descriptors);
void columnUpdate_gpu(int count,size_t maximum,const ColumnUpdateDescriptor* descriptors);
void columnFFTScatterBatch_gpu(int count,int maximum,int nr,const ColumnFFTDescriptor* descriptors,complex* values);
void columnFFTGatherBatch_gpu(int count,int maximum,int nr,const ColumnFFTDescriptor* descriptors,const complex* values);
double batchedColumnDot(const std::vector<ColumnBundle>& x,const std::vector<ColumnBundle>& y,
	int start,int stop,int nStates,bool commensurate);
void columnPairDensity_gpu(int nr,int columns,const complex* ground,const complex* response,
	const double* fillings,complex* output,bool groundFirst);
void matrixColumnAccumulate_gpu(int count,const int* rows,int nr,int column,
	const complex* values,complex* output);
bool useBatchedColumnFFT();
void columnFFTScatter_gpu(size_t count,int nbasis,int nr,const int* indices,const complex* input,complex* output);
void columnFFTMultiply_gpu(size_t count,int nr,const double* realPotential,const complex* complexPotential,complex* values);
void columnFFTGather_gpu(size_t count,int nbasis,int nr,const int* indices,const complex* input,complex* output);
void fieldSymmetryAccumulate_gpu(int nr,const int* indices,const complex* phases,const complex* input,
	complex* output,complex weight,bool conjugate);
ColumnBundle batchedLocalPotential(const ColumnBundle& C,const double* realPotential,const complex* complexPotential,
	const ColumnBundle& Cout,const ManagedArray<complex>* cached=nullptr);
// Mix columns from different k points/bases in each FFT plan. All share one grid/potential.
void batchedLocalPotentials(const std::vector<const ColumnBundle*>& input,
	const std::vector<ColumnBundle*>& output,const double* realPotential,const complex* complexPotential);
void updateColumnBundleGPU(ColumnBundle& output,const ColumnBundle* input,const diagMatrix& scales,
	double outputScale,double inputScale,bool scaleOutput);
ManagedArray<complex> batchedRealSpaceColumns(const ColumnBundle& C);
#endif
#endif
