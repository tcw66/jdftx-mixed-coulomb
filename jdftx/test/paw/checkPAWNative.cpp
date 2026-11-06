// Read/write AEWfns using the existing native ColumnBundle implementation.
#include <electronic/Everything.h>
#include <commands/parser.h>

int main(int argc, char **argv)
{
	initSystem(argc, argv);
	if (argc != 5)
		die("Usage: checkPAWNative input.in AE-cutoff AE.wfns roundtrip.wfns\n");
	{
		Everything e;
		parse(readInputFile(argv[1]), e, false);
		e.setup();
		GridInfo grid;
		grid.R = e.gInfo.R;
		grid.Gmax = sqrt(2 * atof(argv[2]));
		grid.initialize(true, e.symm.getMatrices());
		std::vector<Basis> bases(e.eInfo.nStates);
		std::vector<ColumnBundle> ae(e.eInfo.nStates);
		for (int q = e.eInfo.qStart; q < e.eInfo.qStop; q++)
		{
			bases[q].setup(grid, e.iInfo, atof(argv[2]), e.eInfo.qnums[q].k);
			ae[q].init(e.eInfo.nBands, bases[q].nbasis, &bases[q], &e.eInfo.qnums[q]);
		}
		e.eInfo.read(ae, argv[3]);
		e.eInfo.write(ae, argv[4]);
		logPrintf("AE ColumnBundles read and written using the native I/O routines.\n");
	}
	finalizeSystem();
}
