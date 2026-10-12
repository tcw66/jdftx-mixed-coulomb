/* Direct complex-field regression for exact k-mesh source/conduction FFTs. */
#include <electronic/KMeshCorrelation.h>
#include <core/Operators.h>
#include <core/Util.h>
#include <algorithm>
#include <cstdlib>

namespace
{
void require(bool condition,const char* message)
{ if(!condition) die("K-mesh correlation regression failed: %s\n",message); }

complex phase(const GridInfo& grid,int r,const vector3<>& k)
{
	int z=r%grid.S[2],xy=r/grid.S[2],y=xy%grid.S[1],x=xy/grid.S[1];
	return cis((2*M_PI)*(k[0]*x/grid.S[0]+k[1]*y/grid.S[1]+k[2]*z/grid.S[2]));
}
struct Mapped { int ik; vector3<> wrap; };
Mapped mapped(const std::vector<vector3<>>& mesh,const vector3<>& target)
{
	for(size_t ik=0;ik<mesh.size();ik++) {
		vector3<> delta=target-mesh[ik],wrap;
		for(int d=0;d<3;d++) wrap[d]=round(delta[d]);
		if((delta-wrap).length_squared()<1e-16) return {int(ik),wrap};
	}
	die("Synthetic k mesh is incomplete.\n"); return {-1,vector3<>()};
}
vector3<> point(int slot,const vector3<int>& folding)
{
	int z=slot%folding[2],xy=slot/folding[2],y=xy%folding[1],x=xy/folding[1];
	vector3<> result(double(x)/folding[0],double(y)/folding[1],double(z)/folding[2]);
	for(int d=0;d<3;d++) result[d]-=floor(result[d]+0.5);
	return result;
}

void check(const vector3<int>& folding,bool partial)
{
	GridInfo grid; grid.R=matrix3<>(8.,0.,0.,0.,7.,0.,0.,0.,6.);
	// 1295 grid entries give a partial final 1024-entry FFT tile.
	grid.S=vector3<int>(7,5,37); grid.initialize();
	const int nk=folding[0]*folding[1]*folding[2],centers=3;
	std::vector<vector3<>> mesh,qmesh;
	for(int ik=0;ik<nk;ik++) {
		vector3<> k=point((5*ik+7)%nk,folding);
		k[0]+=ik%3-1; k[2]+=ik%2; // independently represented reciprocal wraps
		mesh.push_back(k); qmesh.push_back(point((7*ik+3)%nk,folding));
	}
	std::vector<std::vector<complexScalarField>> orbital(nk);
	// The correlation object retains device pointers to the original fields.
	// Keep independent host copies for the direct reference instead of calling
	// data() on those fields after construction and moving their allocations.
	std::vector<std::vector<std::vector<complex>>> orbitalReference(nk,
		std::vector<std::vector<complex>>(centers,std::vector<complex>(grid.nr)));
	for(int ik=0;ik<nk;ik++) for(int n=0;n<centers;n++) {
		auto field=complexScalarFieldData::alloc(grid);
		complex* values=field->data();
		for(int r=0;r<grid.nr;r++) values[r]=orbitalReference[ik][n][r]
			=complex(sin(.071*r+.23*ik+.31*n),cos(.049*r-.17*ik+.41*n))*(1.+.03*n);
		orbital[ik].push_back(field);
	}
	KMeshCorrelation disabled(grid,folding,mesh,qmesh,orbital,0.);
	require(!disabled.enabled(),"zero cache budget retains direct fallback");
	const double cacheMiB=16.;
	KMeshCorrelation fft(grid,folding,mesh,qmesh,orbital,cacheMiB);
	require(fft.enabled(),"synthetic regular mesh must use FFT correlations");
	double sourceError=0.,sourceNorm=0.;
	for(int n=0;n<centers;n++) for(int iq=0;iq<nk;iq++) {
		auto rho=fft.source(n,iq); const complex* actual=rho->data();
		std::vector<Mapped> plus;
		for(const auto& k:mesh) plus.push_back(mapped(mesh,k+qmesh[iq]));
		for(int r=0;r<grid.nr;r++) {
			complex expected;
			for(int ik=0;ik<nk;ik++) expected+=orbitalReference[ik][n][r].conj()
				*orbitalReference[plus[ik].ik][n][r]*phase(grid,r,-plus[ik].wrap)/nk;
			sourceError=std::max(sourceError,abs(actual[r]-expected)); sourceNorm=std::max(sourceNorm,abs(expected));
		}
	}
	require(sourceError<=2e-12*std::max(1.,sourceNorm),"complex source densities, mesh order and reciprocal wrap phases");
	const std::vector<int> empty={0,2};
	std::vector<matrix> actual(nk),expected(nk);
	for(int ik=0;ik<nk;ik++) { actual[ik]=zeroes(centers,centers); expected[ik]=zeroes(centers,centers); }
	for(int n:empty) {
		std::vector<complexScalarField> potential(nk);
		std::vector<std::vector<complex>> potentialReference(nk);
		for(int iq=0;iq<nk;iq++) {
			if(partial && iq%3==1) continue;
			auto& field=potential[iq]; field=complexScalarFieldData::alloc(grid);
			potentialReference[iq].resize(grid.nr);
			complex* values=field->data();
			for(int r=0;r<grid.nr;r++) {
				values[r]=complex(cos(.013*r+.33*iq+.19*n),sin(.027*r-.21*iq+.37*n));
				// A distinctive Gamma-only term checks the same linear path used
				// to add the point-charge Ewald potential in the physical driver.
				if(qmesh[iq].length_squared()<1e-16) values[r]+=complex(.173,.047)*cis(.037*r);
				potentialReference[iq][r]=values[r];
			}
		}
		fft.accumulateConduction(n,potential,empty,actual);
		for(int ik=0;ik<nk;ik++) for(int m:empty) {
			complex value;
			for(int iq=0;iq<nk;iq++) if(potential[iq]) {
				Mapped plus=mapped(mesh,mesh[ik]+qmesh[iq]);
				const complex *v=potentialReference[iq].data(),*left=orbitalReference[ik][m].data(),*right=orbitalReference[plus.ik][n].data();
				for(int r=0;r<grid.nr;r++) value+=v[r].conj()*left[r].conj()*right[r]*phase(grid,r,-plus.wrap)*(grid.dV/nk);
			}
			expected[ik].set(m,n,expected[ik](m,n)+value);
		}
	}
	double correctionError=0.,correctionNorm=0.;
	for(int ik=0;ik<nk;ik++) {
		correctionError=std::max(correctionError,nrm2(actual[ik]-expected[ik]));
		correctionNorm=std::max(correctionNorm,nrm2(expected[ik]));
	}
	require(correctionError<=2e-11*std::max(1.,correctionNorm),"conduction convolution, Gamma term and partial q ownership");
	logPrintf("K-mesh correlation regression: %d x %d x %d, %s q mask; maximum source %.3g, contraction %.3g.\n",
		folding[0],folding[1],folding[2],partial ? "partial" : "complete",sourceError,correctionError);
}
}

int main(int argc,char** argv)
{
	initSystem(argc,argv);
	setenv("JDFTX_KI_FFT_CORRELATION","1",1); unsetenv("JDFTX_KI_CORRELATION_CACHE_MIB");
	check(vector3<int>(1,1,1),false);
	check(vector3<int>(3,2,2),false);
	check(vector3<int>(3,2,2),true);
	check(vector3<int>(2,3,2),true);
	check(vector3<int>(2,3,1),true);
	logPrintf("K-mesh correlation regression passed.\n");
	finalizeSystem(); return 0;
}
