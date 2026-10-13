#!/usr/bin/env python3
"""Prepare/run full KI@PBE with the official koopmans DSCF backend."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=int, nargs=3, default=(4,4,4))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.mesh) < 1:
        parser.error('Mesh dimensions must be positive')
    folder = (args.directory or ROOT/'runs'/('full-ki-pbe-'+'x'.join(map(str,args.mesh)))).resolve()
    config = json.loads((ROOT/'si-full-ki-pbe.json').read_text())
    config['kpoints']['grid'] = list(args.mesh)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'si.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Existing full-KI input differs; use a new --directory')
        if not (args.resume or args.prepare_only):
            raise ValueError('This run already exists; use --resume or a new --directory')
    else:
        path.write_text(json.dumps(config,indent=2)+'\n')
    print(f'Prepared full KI@PBE with finite-occupation DSCF screening: {path}',flush=True)
    if args.prepare_only:
        return
    try:
        from koopmans.io import read
    except ImportError:
        raise ValueError('The official koopmans Python package and its KCP/wan2odd backend are required. '
                         'Activate that environment, then rerun this script with --resume. '
                         'Setup: https://koopmans-functionals.org/en/stable/installation.html')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    # Python API avoids differences between the old/new koopmans CLI syntax.
    workflow = read(path)
    workflow.directory = folder
    workflow.run()


if __name__ == '__main__':
    try:
        main()
    except (ValueError,OSError) as error:
        raise SystemExit(str(error))
