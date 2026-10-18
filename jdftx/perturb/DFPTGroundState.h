/* On-demand reconstruction of IBZ ground states. GPL v3 or later; part of JDFTx. */
#ifndef JDFTX_PERTURB_DFPTGROUNDSTATE_H
#define JDFTX_PERTURB_DFPTGROUNDSTATE_H
#include <electronic/ColumnBundle.h>
#include <memory>

class Everything;
class MPIUtil;

// The reduced calculation owns its bases and symmetry operations for the entire
// response sequence. Only requested full-mesh states are ever reconstructed.
class DFPTGroundState {
	Everything& e;
	std::shared_ptr<Everything> source;
	std::vector<int> owners,inverts,operations;
	bool ready=false;
public:
	DFPTGroundState(Everything& e,std::shared_ptr<Everything> source,
		std::vector<int> owners,std::vector<int> inverts,std::vector<int> operations);
	void prepare(const MPIUtil* parent);
	int batches() const;
	void loadBatch(int batch);
	void release();
	// Collective over the q group; requests and destinations are local to each
	// rank. A full-mesh index may target a reciprocal-equivalent shifted basis.
	void reconstruct(const std::vector<std::pair<int,ColumnBundle*>>& requests);
	void writeWavefunctions(const char* filename);
};
#endif
