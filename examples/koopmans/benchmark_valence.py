#!/usr/bin/env python3
"""Time KI solvers at identical valence rotations in isolated output folders."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--screening-mesh', type=int, nargs=3, metavar=('NQ0','NQ1','NQ2'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.ranks < 1 or args.threads < 1:
        parser.error('ranks and threads must be positive')
    root = args.example_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    build = repo/'build'
    suffix = '' if args.cpu else '_gpu'
    binary = Path(os.environ.get('WANNIER_BINARY',build/('wannier'+suffix))).expanduser().resolve()
    if not binary.is_file():
        parser.error('missing executable: '+str(binary))
    output = args.output.resolve() if args.output else root/'archive'/('valence-benchmark-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True, exist_ok=False)

    def expand(path):
        lines = []
        for line in path.read_text().splitlines():
            words = line.split()
            if words and words[0] == 'include':
                if words[1] == 'ki-response.in':
                    continue
                lines.append(expand(root/words[1]))
            else:
                lines.append(line)
        return '\n'.join(lines)+'\n'

    base = expand(root/'02-valence.in')
    base = re.sub(r'^wannier-initial-state .*$',f'wannier-initial-state {root}/state/scf.$VAR',base,flags=re.M)
    base = re.sub(r'^ion-species .*$',f'ion-species {repo}/build/pseudopotentials/SG15/$ID_ONCV_PBE.upf',base,flags=re.M)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS='1',
               JDFTX_FFT_BATCH_SIZE=os.environ.get('JDFTX_FFT_BATCH_SIZE','32'))
    launch = [] if args.ranks == 1 else [os.environ.get('MPIEXEC','mpirun'),'-n',str(args.ranks),'--bind-to','none']

    def run(folder,text,label):
        folder.mkdir(exist_ok=True)
        text = re.sub(r'^wannier-dump-name .*$',f'wannier-dump-name {folder}/valence.$VAR',text,flags=re.M)
        inp, log = folder/(label+'.in'), folder/(label+'.out')
        inp.write_text(text)
        start = time.monotonic()
        with (folder/(label+'.launcher.log')).open('w') as stream:
            result = subprocess.run(launch+[str(binary),'-c',str(args.threads),'-i',str(inp),'-o',str(log)],
                                    cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT)
        seconds = time.monotonic()-start
        contents = log.read_text()
        if result.returncode or not contents.rstrip().endswith('Done!'):
            raise RuntimeError(f'{label} failed; see {log}')
        return seconds,contents

    localized = output/'localized'
    localized.mkdir()
    for extension in ('mlwfU','mlwfM0','mlwfState'):
        source = root/'results'/('valence.'+extension)
        if source.exists():
            shutil.copy2(source,localized/source.name)
    run(localized,base,'localize')
    minimize_start = base.index('wannier-minimize')
    frozen = base[:minimize_start]+re.sub(r'nIterations\s+\d+', 'nIterations 0',base[minimize_start:],count=1)
    response = f'wannier-koopmans 0 400 1e-7 yes 1e-6 yes {min(2,args.ranks)} yes\nwannier-koopmans-spin yes\n'
    if args.screening_mesh:
        response += 'wannier-koopmans-screening-mesh '+' '.join(map(str,args.screening_mesh))+' 0\n'
    reference = None
    timings = []
    for label,config in [('coupled','no yes'),('nested-cg','yes yes'),('nested-minres','yes no')]:
        folder = output/label
        folder.mkdir()
        for extension in ('mlwfU','mlwfM0','mlwfState'):
            source = localized/('valence.'+extension)
            if source.exists():
                shutil.copy2(source,folder/source.name)
        seconds,log = run(folder,frozen+response+f'wannier-koopmans-response {config} 0.4 8 80 2048 512 4096\n',label)
        screening = np.loadtxt(folder/'valence.koopmansScreening',ndmin=2)
        bands = np.loadtxt(folder/'valence.koopmansBands',ndmin=2)
        error = 0.
        if reference is None:
            reference = (screening,bands)
        else:
            for actual,expected in zip((screening,bands),reference):
                np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
            error = float(np.max(abs(bands[:,3:]-reference[1][:,3:])))
        solves,total = map(int,re.search(r'KI response solves: (\d+)/(\d+)',log).groups())
        timings.append(dict(solver=label,seconds=seconds,independent_pairs=solves,total_pairs=total,
                            max_band_difference_Ha=error))
        print(json.dumps(timings[-1]),flush=True)
        (output/'timings.json').write_text(json.dumps(dict(backend='cpu' if args.cpu else 'gpu',ranks=args.ranks,
            threads=args.threads,screening_mesh=args.screening_mesh,results=timings),indent=2)+'\n')
    print('Results: '+str(output))


if __name__ == '__main__':
    main()
