/* Regression for ordinary / derivative projector cache separation and lifetime. */
#include <electronic/Everything.h>
#include <electronic/ColumnBundle.h>
#include <electronic/SpeciesInfo.h>
#include <commands/parser.h>
#include <core/Util.h>
#include <algorithm>

namespace {
void require(bool condition,const char* message)
{ if(!condition) die("Projector cache regression failed: %s\n",message); }

void same(const ColumnBundle& actual,const ColumnBundle& expected,const char* message)
{
	ColumnBundle difference=actual-expected;
	double error=sqrt(dotc(difference,difference).real());
	double reference=sqrt(dotc(expected,expected).real());
	require(error<=1e-13*std::max(1.,reference),message);
}

void check(Everything& e,SpeciesInfo& species)
{
	const Basis& basis=e.basis.front();
	QuantumNumber qnum=e.eInfo.qnums.front();
	ColumnBundle state(1,basis.nbasis,&basis,&qnum,isGpuEnabled());
	const size_t previousLimit=species.projectorCacheLimit();
	const bool previousSetting=e.cntrl.cacheProjectors;
	e.cntrl.cacheProjectors=true;
	species.setProjectorCacheLimit(size_t(-1)); species.clearProjectorCache();
	auto uncached=[&](const ColumnBundle& input,const vector3<>* direction=0,int stress=-1) {
		e.cntrl.cacheProjectors=false;
		auto result=species.getV(input,direction,stress);
		e.cntrl.cacheProjectors=true;
		return result;
	};
	auto ordinary=species.getV(state);
	require(bool(ordinary),"input must contain nonlocal projectors");
	const size_t bytes=ordinary->nData()*sizeof(complex);
	require(species.getV(state).get()==ordinary.get(),"ordinary projectors are reused");
	require(species.projectorCacheBytes()==bytes,"ordinary retained-byte count");
	same(*ordinary,*uncached(state),"ordinary cached/uncached values");
	for(int stress : {0,1,8}) {
		auto derivative=species.getV(state,0,stress);
		require(derivative.get()!=ordinary.get(),"stress derivatives cannot return ordinary projectors");
		same(*derivative,*uncached(state,0,stress),"stress cached/uncached values");
		require(species.getV(state).get()==ordinary.get(),"stress derivatives do not replace ordinary cache");
	}
	vector3<> direction(1.,0.,0.);
	same(*species.getV(state,&direction),*uncached(state,&direction),"Cartesian derivative values");
	require(species.getV(state).get()==ordinary.get(),"Cartesian derivatives do not replace ordinary cache");

	// A cache key may be requested through a temporary quantum-number object.
	// Returned projectors own that metadata, including after cache eviction.
	std::shared_ptr<ColumnBundle> temporary;
	ColumnBundle slice, copied, assigned, similar;
	{
		QuantumNumber temporaryQnum=qnum; temporaryQnum.weight=0.123;
		ColumnBundle temporaryState(1,basis.nbasis,&basis,&temporaryQnum,isGpuEnabled());
		temporary=species.getV(temporaryState);
		slice=temporary->getSub(0,1);
		copied=ColumnBundle(slice);
		assigned=slice;
		similar=slice.similar();
	}
	species.clearProjectorCache();
	require(temporary->qnum->weight==0.123 && temporary->qnum->k==qnum.k,"quantum-number lifetime");
	temporary.reset();
	slice.free();
	require(copied.qnum->weight==0.123 && assigned.qnum->weight==0.123
		&& similar.qnum->weight==0.123,"derived-bundle quantum-number lifetime");
	ColumnBundle moved(std::move(copied));
	ColumnBundle moveAssigned; moveAssigned=std::move(assigned);
	similar.init(similar.nCols(),similar.colLength(),similar.basis,similar.qnum,similar.isOnGpu());
	require(moved.qnum->weight==0.123 && moveAssigned.qnum->weight==0.123
		&& similar.qnum->weight==0.123,"move and reinitialization preserve metadata ownership");

	// Only the most recent k can be retained at one basis address.
	for(int step=0;step<12;step++) {
		qnum.k[0]+=0.03125;
		auto shifted=species.getV(state);
		require(species.projectorCacheBytes()==bytes,"latest-k retention stays bounded");
		same(*shifted,*uncached(state),"changed-k cached/uncached values");
	}

	// A one-entry budget must evict another basis, even while a caller holds
	// its returned shared pointer. The limit counts cache ownership only.
	species.setProjectorCacheLimit(bytes);
	auto first=species.getV(state);
	Basis secondBasis;
	secondBasis.setup(*basis.gInfo,e.iInfo,e.cntrl.Ecut,e.eInfo.qnums.front().k);
	ColumnBundle secondState(1,secondBasis.nbasis,&secondBasis,&qnum,isGpuEnabled());
	auto second=species.getV(secondState);
	require(species.projectorCacheBytes()<=bytes,"finite byte limit");
	require(species.getV(state).get()!=first.get(),"least-recently-used eviction");
	species.clearProjectorCache(&basis);
	require(species.projectorCacheBytes()==0,"per-basis invalidation");

	// Setup may reuse the same basis address, length and allocation while
	// changing the index order. It must invalidate automatically.
	auto beforeSetup=species.getV(secondState);
	std::vector<int> reversed(secondBasis.index.data(),secondBasis.index.data()+secondBasis.nbasis);
	std::reverse(reversed.begin(),reversed.end());
	secondBasis.setup(*basis.gInfo,e.iInfo,reversed);
	require(species.projectorCacheBytes()==0,"automatic setup invalidation");
	auto afterSetup=species.getV(secondState);
	require(beforeSetup.get()!=afterSetup.get(),"rebuilt-basis invalidation");
	same(*afterSetup,*uncached(secondState),"rebuilt-basis values");

	species.setProjectorCacheLimit(bytes-1);
	auto tooLarge=species.getV(state);
	require(species.projectorCacheBytes()==0,"oversized entries are not retained");
	species.setProjectorCacheLimit(0);
	auto disabled=species.getV(state);
	require(species.getV(state).get()!=disabled.get(),"zero budget disables retention");
	require(species.projectorCacheBytes()==0,"disabled retention byte count");

	species.setProjectorCacheLimit(size_t(-1));
	species.getV(state);
	species.atpos.front()[0]+=0.01; species.sync_atpos();
	require(species.projectorCacheBytes()==0,"atom motion invalidates projectors");
	same(*species.getV(state),*uncached(state),"moved-atom values");
	species.atpos.front()[0]-=0.01; species.sync_atpos();
	species.setProjectorCacheLimit(previousLimit);
	e.cntrl.cacheProjectors=previousSetting;
}
}

int main(int argc,char** argv)
{
	InitParams params("Validate nonlocal projector caching with a native JDFTx input.");
	initSystemCmdline(argc,argv,params);
	{
		Everything e;
		parse(readInputFile(params.inputFilename),e);
		e.eVars.skipWfnsInit=true;
		e.setup();
		require(!e.basis.empty(),"input must contain a k point");
		bool checked=false;
		for(const auto& species:e.iInfo.species) if(species->nProjectors()) {
			check(e,*species); checked=true;
		}
		require(checked,"input must contain nonlocal projectors");
		logPrintf("Projector cache regression passed.\n");
	}
	finalizeSystem();
	return 0;
}
