#!/usr/bin/env python3
"""Run silicon SCF -> primitive-cell DFPT -> Fourier interpolation and plotting."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(items):
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode())
        h.update(b'\0')
    return h.hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def find_pseudo(requested):
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'pseudopotential does not exist: {path}')
        return path
    roots = [REPO/'build/pseudopotentials']
    if os.environ.get('JDFTX_PSEUDO_DIR'):
        roots.insert(0, Path(os.environ['JDFTX_PSEUDO_DIR']))
    roots += sorted(REPO.parent.glob('build*/pseudopotentials'))
    for root in roots:
        path = root/'SG15/Si_ONCV_PBE.upf'
        if path.is_file():
            return path.resolve()
    raise ValueError('SG15 silicon not found; pass --pseudo /path/to/Si_ONCV_PBE.upf')


def find_binary(requested):
    candidate = requested or os.environ.get('JDFTX_BIN') or str(REPO/'build/jdftx')
    located = shutil.which(candidate)
    if not located:
        raise ValueError(f'JDFTx executable not found: {candidate}; pass --jdftx-bin /path/to/jdftx')
    return Path(located).resolve()


def render(name, values):
    text = (HERE/name).read_text()
    # Only expand braced variables. Preserve literal JDFTx $VAR dump patterns.
    return re.sub(r'\$\{([A-Z0-9_]+)\}', lambda m: str(values[m[1]]), text)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def require_marker(path, expected):
    if not path.is_file():
        raise ValueError(f'{path.name} missing: run the preceding stage first')
    marker = json.loads(path.read_text())
    if marker['signature'] != expected:
        raise ValueError(f'{path.name} does not match these inputs; rerun the preceding stage or use a new --out directory')
    for name, checksum in marker['files'].items():
        artifact = path.parent/name
        if not artifact.is_file() or file_digest(artifact) != checksum:
            raise ValueError(f'{name} is missing or changed; rerun the preceding stage')


def mark(path, signature, filenames):
    for name in filenames:
        if not (path.parent/name).is_file() or (path.parent/name).stat().st_size == 0:
            raise ValueError(f'expected output missing or empty: {name}')
    write_json(path, {'signature': signature,
                      'files': {name: file_digest(path.parent/name) for name in filenames}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'scf', 'dfpt', 'interpolate', 'check'], default='all')
    parser.add_argument('--out', type=Path, help='calculation directory; defaults to results (results-quick with --quick)')
    parser.add_argument('--quick', action='store_true', help='8 Ha cutoff, 2x2x2 k mesh and 2x1x1 q mesh: workflow test only')
    parser.add_argument('--ecut', type=float, help='wavefunction cutoff in Hartree; default 30')
    parser.add_argument('--lattice', type=float, default=10.26, help='conventional cubic lattice parameter in bohr')
    parser.add_argument('--kgrid', type=int, nargs=3, metavar=('K0', 'K1', 'K2'))
    parser.add_argument('--qgrid', type=int, nargs=3, metavar=('Q0', 'Q1', 'Q2'))
    parser.add_argument('--pseudo', help='SG15 silicon norm-conserving UPF path')
    parser.add_argument('--jdftx-bin', help='the rebuilt jdftx executable')
    parser.add_argument('--launcher', default='', help='MPI launcher prefix, e.g. "mpirun -n 4" or "srun -n 4"')
    parser.add_argument('--threads', type=int, default=1, help='CPU threads per MPI rank')
    parser.add_argument('--q-groups', type=int, default=1, help='independent MPI groups for phonon q points; default 1')
    parser.add_argument('--reduced-scf', action='store_true', help='converge the SCF on the irreducible k mesh and expand its checkpoint for DFPT')
    parser.add_argument('--points-per-segment', type=int, default=60)
    parser.add_argument('--no-plot', action='store_true', help='write numerical results and checks without requiring Matplotlib')
    args = parser.parse_args()
    try:
        ecut = args.ecut if args.ecut is not None else (8. if args.quick else 30.)
        kgrid = args.kgrid or ([2, 2, 2] if args.quick else [6, 6, 6])
        qgrid = args.qgrid or ([2, 1, 1] if args.quick else [4, 4, 4])
        if not math.isfinite(ecut) or not math.isfinite(args.lattice) or min(ecut, args.lattice, args.threads, args.q_groups, args.points_per_segment, *kgrid, *qgrid) <= 0:
            raise ValueError('cutoff, lattice, grids, threads and path sampling must be finite and positive')
        if math.prod(qgrid) > 2147483647:
            raise ValueError('q grid is too large')
        directory = (args.out or HERE/('results-quick' if args.quick else 'results')).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        pseudo = find_pseudo(args.pseudo)
        binary = find_binary(args.jdftx_bin)
        interpolator = REPO/'jdftx/scripts/phononDFPT'
        if not interpolator.is_file():
            raise ValueError(f'interpolation utility missing: {interpolator}')
        if importlib.util.find_spec('numpy') is None:
            raise ValueError('NumPy is required; use a Python environment containing numpy')
        if args.stage in ('all', 'interpolate') and not args.no_plot and importlib.util.find_spec('matplotlib') is None:
            raise ValueError('Matplotlib is required for plots; use --no-plot for numerical output only')
        values = {'SI_LATTICE': args.lattice, 'SI_ECUT': ecut, 'SI_PSEUDO': pseudo}
        values.update({f'K{i}': kgrid[i] for i in range(3)})
        values.update({f'Q{i}': qgrid[i] for i in range(3)})
        inputs = {name: render(name, values) for name in ('system.in', 'common.in', '01-scf.in', '02-dfpt.in')}
        if args.reduced_scf:
            inputs['common-scf.in'] = re.sub(r'(?m)^\s*symmetries\s+[^\n]*(?:\n|$)', '', inputs['common.in'])
            inputs['common-scf.in'] += '\nsymmetries automatic\n'
            inputs['01-scf.in'] = re.sub(r'(?m)^[ \t]*include[ \t]+common\.in[ \t]*$', 'include common-scf.in', inputs['01-scf.in'])
            inputs['02-dfpt.in'] = re.sub(r'(?m)^\s*initial-state\s+[^\n]*(?:\n|$)', '', inputs['02-dfpt.in'])
            inputs['02-dfpt.in'] += '\nphonon-dfpt-initial-state 01-scf.in scf.$VAR\n'
        inputs['02-dfpt.in'] = re.sub(r'(?m)^\s*phonon-dfpt-q-groups\s+[^\n]*(?:\n|$)', '', inputs['02-dfpt.in'])
        inputs['02-dfpt.in'] += f'\nphonon-dfpt-q-groups {args.q_groups}\n'
        for name, text in inputs.items():
            (directory/name).write_text(text)
        pseudo_checksum = file_digest(pseudo)
        scf_signature = digest([inputs[n] for n in ('system.in', 'common.in', '01-scf.in')]
                               + ([inputs['common-scf.in']] if args.reduced_scf else []) + [pseudo_checksum])
        dfpt_signature = digest([scf_signature, inputs['02-dfpt.in']])
        write_json(directory/'settings.json', {'lattice_bohr': args.lattice, 'ecut_Hartree': ecut,
                   'kgrid': kgrid, 'qgrid': qgrid, 'pseudo': str(pseudo), 'pseudo_sha256': pseudo_checksum,
                   'jdftx_bin': str(binary), 'launcher': args.launcher, 'threads': args.threads,
                   'q_groups': args.q_groups,
                   'reduced_scf': args.reduced_scf,
                   'quick_test': args.quick, 'scf_signature': scf_signature, 'dfpt_signature': dfpt_signature})
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
        prefix = shlex.split(args.launcher)+[str(binary), '-c', str(args.threads), '-d']

        def run_jdftx(input_name, log_name, dry=False):
            command = prefix+(['-n'] if dry else [])+['-i', input_name, '-o', log_name]
            print('+ '+shlex.join(command), flush=True)
            subprocess.run(command, cwd=directory, env=env, check=True)
            text = (directory/log_name).read_text()
            if not re.search(r'^Done!$', text, re.MULTILINE):
                raise ValueError(f'JDFTx did not finish successfully; see {directory/log_name}')
            return text

        print(f'Calculation directory: {directory}', flush=True)
        print(f'Cutoff {ecut:g} Ha; k mesh {kgrid}; phonon q mesh {qgrid}', flush=True)
        scf_marker, dfpt_marker = directory/'scf.complete.json', directory/'dfpt.complete.json'
        if args.stage == 'check':
            run_jdftx('01-scf.in', 'scf-check.out', dry=True)
            run_jdftx('02-dfpt.in', 'dfpt-check.out', dry=True)
            print('Both JDFTx inputs passed initialization. No SCF or phonon solve was performed.')
            return
        if args.stage in ('all', 'scf'):
            scf_marker.unlink(missing_ok=True)
            dfpt_marker.unlink(missing_ok=True)
            text = run_jdftx('01-scf.in', 'scf.out')
            if not re.search(r'^SCF: Converged', text, re.MULTILINE):
                raise ValueError('ground-state SCF did not converge; see scf.out')
            mark(scf_marker, scf_signature, ['scf.wfns', 'scf.fillings', 'scf.eigenvals'])
        if args.stage in ('all', 'dfpt'):
            require_marker(scf_marker, scf_signature)
            dfpt_marker.unlink(missing_ok=True)
            text = run_jdftx('02-dfpt.in', 'dfpt.out')
            # DFPT includes a fresh ground-state polish before solving responses.
            if not re.search(r'^ElecMinimize: Converged', text, re.MULTILINE):
                raise ValueError('DFPT ground-state polish did not converge; see dfpt.out')
            mark(dfpt_marker, dfpt_signature, ['phonon.'+suffix for suffix in (
                'phononForceMatrixQ', 'phononCellMap', 'phononCellWeights', 'phononOmegaSq', 'phononBasis')])
        if args.stage in ('all', 'interpolate'):
            require_marker(scf_marker, scf_signature)
            require_marker(dfpt_marker, dfpt_signature)
            subprocess.run([sys.executable, str(HERE/'make_qpath.py'), '--lattice', str(args.lattice),
                            '--points-per-segment', str(args.points_per_segment), '--out', str(directory)], check=True)
            subprocess.run([sys.executable, str(interpolator), str(directory/'phonon.'), str(directory/'qpath.dat'),
                            '-o', str(directory/'frequencies.dat'), '--matrices', str(directory/'phonons.npz')], check=True)
            subprocess.run([sys.executable, str(HERE/'analyze.py'), str(directory),
                            '--interpolator', str(interpolator)]+(['--no-plot'] if args.no_plot else []), check=True)
        print(f'Completed {args.stage} stage. Results: {directory}', flush=True)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'run.py: {error}\n')


if __name__ == '__main__':
    main()
