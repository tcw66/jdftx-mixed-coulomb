#!/usr/bin/env python3
"""Configure the Nb post-processing example, optionally refreshing Wannier data."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/home/tyler/Nb_DFPT'))
    parser.add_argument('--output', type=Path, default=here/'results-nb')
    parser.add_argument('--stage', choices=('run', 'spectrum', 'solve', 'bdg'), default='run')
    parser.add_argument('--refresh-wannier', action='store_true', help='run the existing 03-wannier.in with an isolated output prefix')
    parser.add_argument('--wannier-binary', type=Path, default=repo/'build/wannier_gpu')
    parser.add_argument('--wannier-prefix', type=Path, help='use a separately prepared Wannier export prefix, including trailing dot')
    parser.add_argument('--wannier-log', type=Path)
    parser.add_argument('--kmesh', nargs=3, type=int)
    parser.add_argument('--qmesh', nargs=3, type=int)
    parser.add_argument('--sigma-ev', type=float)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'auto'))
    parser.add_argument('--gpu-device', type=int)
    parser.add_argument('--chunk-size', type=int)
    args = parser.parse_args()
    data, out = args.data.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((here/'nb.json').read_text())
    saved = out/'configuration.json'
    if args.stage in ('solve', 'bdg') and saved.exists():
        config = json.loads(saved.read_text())
    else:
        config.update(wannier_prefix=str(data/'wannier.'), wannier_log=str(data/'wannier.out'),
                      phonon_prefix=str(data/'phonon.'), metadata_log=str(data/'dfpt.out'), output=str(out))
    if args.refresh_wannier:
        dest = out/'wannier-input'
        dest.mkdir(exist_ok=True)
        source = (data/'03-wannier.in').read_text()
        source, count = re.subn(r'^wannier-dump-name\s+\S+',
                               lambda _: 'wannier-dump-name '+str(dest/'wannier.$VAR'), source, flags=re.M)
        if count != 1:
            parser.error('03-wannier.in must contain exactly one wannier-dump-name')
        (dest/'wannier.in').write_text(source)
        env = dict(os.environ, OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '2'),
                   OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS', '1'))
        subprocess.run([str(args.wannier_binary.resolve()), '-i', str(dest/'wannier.in'),
                        '-o', str(dest/'wannier.out')], cwd=data, env=env, check=True)
        config.update(wannier_prefix=str(dest/'wannier.'), wannier_log=str(dest/'wannier.out'))
    if args.wannier_prefix:
        config['wannier_prefix'] = str(args.wannier_prefix.resolve())
    if args.wannier_log:
        config['wannier_log'] = str(args.wannier_log.resolve())
    for name in ('kmesh', 'qmesh', 'sigma_ev', 'backend', 'gpu_device', 'chunk_size'):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    saved.write_text(json.dumps(config, indent=2)+'\n')
    subprocess.run([sys.executable, str(repo/'jdftx/scripts/migdalEliashberg'),
                    args.stage, str(saved)], check=True)


if __name__ == '__main__':
    main()
