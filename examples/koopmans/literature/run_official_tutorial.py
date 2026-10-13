#!/usr/bin/env python3
"""Prepare/run the official silicon tutorial through its full-KI QE/KCP backend."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=int, nargs=3, help='Override the downloaded 2x2x2 mesh')
    parser.add_argument('--task', choices=('singlepoint', 'wannierize'), default='singlepoint')
    parser.add_argument('--npool', type=int, help='Override the downloaded pw.x pool count of four')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Allow an existing matching input directory')
    args = parser.parse_args()
    if args.mesh and min(args.mesh) < 1:
        parser.error('Mesh dimensions must be positive')
    if args.npool is not None and args.npool < 1:
        parser.error('Pool count must be positive')
    config = json.loads((ROOT/'si-official-tutorial.json').read_text())
    config['workflow']['task'] = args.task
    if args.mesh:
        config['kpoints']['grid'] = args.mesh
    if args.npool is not None:
        config['engine']['commands']['pw']['npool'] = args.npool
    mesh = 'x'.join(map(str, config['kpoints']['grid']))
    folder = (args.directory or ROOT/'runs'/f'official-tutorial-{args.task}-{mesh}').resolve()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'si.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Existing tutorial input differs; use a new --directory')
        if not (args.resume or args.prepare_only):
            raise ValueError('Input already exists; use --resume or a new --directory')
    else:
        path.write_text(json.dumps(config, indent=2)+'\n')
    print(f'Prepared official silicon {args.task} workflow: {path}', flush=True)
    if args.prepare_only:
        return
    try:
        from koopmans.io import read
    except ImportError:
        raise ValueError('Activate the official koopmans environment with its KCP/wan2odd backend, '
                         'then rerun with --resume. Stock QE pw.x/ph.x/kcw.x alone cannot execute this workflow. '
                         'See https://koopmans-functionals.org/en/latest/installation.html')
    workflow = read(path)
    workflow.directory = folder
    workflow.run()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
