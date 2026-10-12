/* Numerical check of mixed-shape occupied projections, GPL v3 or later. */
#include <electronic/ColumnBundleFFT.h>
#include <core/matrix.h>
#include <core/Util.h>
#include <core/GridInfo.h>
#include <cmath>

void check(int count)
{
	std::vector<ColumnBundle> projectors(count),values(count);
	std::vector<GridInfo> grids(count);
	std::vector<Basis> bases(count);
	std::vector<std::vector<complex>> reference(count),bra(count);
	std::vector<const ColumnBundle*> inputs;
	std::vector<ColumnBundle*> outputs;
	for(int b=0;b<count;b++) {
		int length=31+(b%3)*17,occupied=1+(b%3),bands=1+(b%4);
		grids[b].detR=1.+37.*(b%2); bases[b].gInfo=&grids[b]; bases[b].nbasis=length;
		projectors[b].init(occupied,length,&bases[b],nullptr,isGpuEnabled());
		values[b].init(bands,length,&bases[b],nullptr,isGpuEnabled());
		complex* c=projectors[b].data();
		for(int j=0;j<occupied;j++) for(int i=0;i<length;i++)
			c[j*length+i]=complex(sin(.27*i+.71*j+.23*b),cos(.16*i-.35*j+.09*b));
		projectors[b]=projectors[b]*invsqrt(grids[b].detR*(projectors[b]^projectors[b]));
		const auto& projector=projectors[b];
		bra[b].assign(projector.data(),projector.data()+projector.nData());
		complex* x=values[b].data();
		for(int j=0;j<bands;j++) for(int i=0;i<length;i++)
			x[j*length+i]=complex(cos(.13*i+.31*j+.17*b),sin(.23*i-.43*j+.11*b));
		reference[b].assign(x,x+values[b].nData());
		inputs.push_back(&projectors[b]); outputs.push_back(&values[b]);
	}
	for(int pass=0;pass<2;pass++) {
		for(int b=0;b<count;b++) {
			int length=values[b].colLength(),occupied=projectors[b].nCols(),bands=values[b].nCols();
			std::vector<complex> overlaps(occupied*bands);
			for(int j=0;j<bands;j++) for(int n=0;n<occupied;n++) for(int i=0;i<length;i++)
				overlaps[j*occupied+n]+=grids[b].detR*conj(bra[b][n*length+i])*reference[b][j*length+i];
			for(int j=0;j<bands;j++) for(int i=0;i<length;i++) for(int n=0;n<occupied;n++)
				reference[b][j*length+i]-=bra[b][n*length+i]*overlaps[j*occupied+n];
		}
		projectOutBatched(inputs,outputs);
		double maximum=0.;
		for(int b=0;b<count;b++) {
			const auto& value=values[b]; const complex* actual=value.data();
			for(size_t i=0;i<value.nData();i++) maximum=std::max(maximum,abs(actual[i]-reference[b][i]));
		}
		if(maximum>2e-11) die("Batched projection mismatch: %.3g.\n",maximum);
		logPrintf("Batched projections: %d blocks, pass %d, error %.3g.\n",count,pass+1,maximum);
	}
}

int main(int argc,char** argv)
{
	initSystem(argc,argv);
	check(0); check(17); check(137); check(9); check(0);
	logPrintf("Batched projector regression passed.\n");
	finalizeSystem(); return 0;
}
