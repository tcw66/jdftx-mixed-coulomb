#!/usr/bin/env python3
"""Validate DFPT accelerations against explicit displacement/offset solves."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('binary', type=Path)
parser.add_argument('pseudo', type=Path)
parser.add_argument('--out', type=Path)
parser.add_argument('--launcher')
parser.add_argument('--rank-flag', default='-n')
parser.add_argument('--ranks', type=int, default=4)
args = parser.parse_args()
binary, pseudo = args.binary.resolve(), args.pseudo.resolve()
environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
common = f'''lattice face-centered Cubic 10.26
coords-type lattice
ion-species {pseudo}
ion Si 0 0 0 1
ion Si 0.25 0.25 0.25 1
elec-cutoff 8
kpoint-folding 2 2 2
elec-n-bands 4
electronic-minimize nIterations 500 energyDiffThreshold 1e-13
'''
scf = ('electronic-scf nIterations 200 nEigSteps 0 energyDiffThreshold 0 '
       'eigDiffThreshold 0 residualThreshold 1e-10\n')
response = 'perturb-minimize nIterations 500 residualTol 1e-7 residualDiffThreshold 0\n'
timings = {}


def run(directory, name, text, success=True, dry=False):
    (directory / f'{name}.in').write_text(text)
    command = [str(binary), '-c', '1', '-d', '-i', f'{name}.in', '-o', f'{name}.out']
    if dry:
        command.append('-n')
    if args.launcher:
        command = [args.launcher] + shlex.split(os.environ.get('DFPT_MPI_PREFLAGS', '')) + [args.rank_flag, str(args.ranks)] + command
    start = time.monotonic()
    result = subprocess.run(command, cwd=directory, env=environment, capture_output=True, text=True, timeout=240)
    timings[name] = time.monotonic() - start
    log = (directory / f'{name}.out').read_text()
    if success:
        assert result.returncode == 0 and 'Done!' in log, log + result.stderr
    else:
        assert result.returncode != 0, log
    return log + result.stderr


def matrices(directory, prefix):
    lines = (directory / f'{prefix}.phononForceMatrixQ').read_text().splitlines()
    nq, nm = map(int, lines[1].split())
    data = np.fromstring(' '.join(lines[2:]), sep=' ').reshape(nq, 3 + 2*nm*nm)
    z = data[:, 3:].reshape(nq, nm, nm, 2)
    return z[..., 0] + 1j*z[..., 1]


context = tempfile.TemporaryDirectory(prefix='jdftx-acceleration-') if args.out is None else nullcontext(str(args.out.resolve()))
with context as name:
    directory = Path(name)
    directory.mkdir(parents=True, exist_ok=True)
    for kind, symmetry in [('full', 'none'), ('reduced', 'automatic')]:
        log = run(directory, kind, common + f'symmetries {symmetry}\n' + scf +
                  f'dump-name {kind}.$VAR\ndump End State Fillings\n')
        assert 'SCF: Converged' in log
    results = {}
    solves = {}
    # Isolate each acceleration, then exercise the reduced checkpoint with all enabled.
    for name, displacement, reuse, reduced in [
        ('baseline', 'no', 'no', False),
        ('displacements', 'yes', 'no', False),
        ('reuse', 'no', 'yes', False),
        ('accelerated', 'yes', 'yes', True),
    ]:
        initial = 'phonon-dfpt-initial-state reduced.in reduced.$VAR\n' if reduced else 'initial-state full.$VAR\n'
        text = common + 'symmetries none\n' + initial + response + 'phonon-dfpt 2 2 2\n'
        text += f'phonon-dfpt-displacement-symmetries {displacement}\nphonon-dfpt-reuse-offset-states {reuse}\n'
        if args.launcher and reduced:
            text += 'phonon-dfpt-q-groups 2\n'
        text += f'dump-name {name}.$VAR\ndump End State Fillings\n'
        log = run(directory, name, text)
        results[name] = matrices(directory, name)
        match = re.search(r'representatives \((\d+) displacement solves\)', log)
        solves[name] = int(match[1])
        if displacement == 'yes':
            assert solves[name] < solves['baseline'], log
        reuse_rows = re.findall(r'DFPT offset-state reuse: (\d+)/(\d+) k/sign pairs \((\d+) required refinement\)', log)
        if not (args.launcher and reduced):
            assert reuse_rows and all(int(a) == (int(b) if reuse == 'yes' else 0) for a, b, c in reuse_rows), log
        if reduced:
            assert 'DFPT ground-state expansion: 3 reduced states -> 8 full-mesh states.' in log
            assert (directory / f'{name}.wfns').stat().st_size == (directory / 'full.wfns').stat().st_size
            assert (directory / f'{name}.fillings').stat().st_size == 8*4*8
            initial_energy = float(re.search(r'ElecMinimize: Iter:\s*0\s+Etot:\s*(\S+)', log)[1])
            source_energy = float(re.findall(r'SCF: Cycle:.*?Etot:\s*(\S+)', (directory / 'reduced.out').read_text())[-1])
            assert abs(initial_energy - source_energy) < 1e-9, 'unfolded checkpoint changed the initial energy'
        error = np.linalg.norm(results[name] - results['baseline']) / np.linalg.norm(results['baseline'])
        assert error < 2e-5, f'{name} force-matrix error {error:g}'
        print(f'{name}: {solves[name]} response solves, {timings[name]:.3f} s, force-matrix relative error {error:.3g}', flush=True)

    # Noncommensurate q still uses the eigensolver; reconstruction includes complex cell phases.
    for enabled in ('no', 'yes'):
        name = f'offmesh-{enabled}'
        log = run(directory, name, common + 'symmetries none\ninitial-state full.$VAR\n' + response +
                  f'phonon-dfpt 4 1 1\nphonon-dfpt-displacement-symmetries {enabled}\n'
                  f'phonon-dfpt-reuse-offset-states {enabled}\ndump-name {name}.$VAR\ndump End None\n')
        results[name] = matrices(directory, name)
        if enabled == 'yes':
            assert 'DFPT offset-state reuse: 0/16 k/sign pairs' in log
            error = np.linalg.norm(results[name] - results['offmesh-no']) / np.linalg.norm(results['offmesh-no'])
            assert error < 2e-5, f'off-mesh force-matrix error {error:g}'

    # A manual identity-only space group forces explicit time reversal in the
    # restart map. Odd k meshes include distinct k/-k states and reciprocal wraps.
    odd = common.replace('kpoint-folding 2 2 2', 'kpoint-folding 3 3 3')
    for name, symmetry in [('odd-full', 'symmetries none\n'),
                           ('odd-reduced', 'symmetries manual\nsymmetry-matrix 1 0 0 0 1 0 0 0 1 0 0 0\n')]:
        run(directory, name, odd + symmetry + scf + f'dump-name {name}.$VAR\ndump End State Fillings\n')
    for name, initial, displacement in [
        ('odd-baseline', 'initial-state odd-full.$VAR\n', 'no'),
        ('odd-accelerated', 'phonon-dfpt-initial-state odd-reduced.in odd-reduced.$VAR\n', 'yes'),
    ]:
        log = run(directory, name, odd + 'symmetries none\n' + initial + response +
                  f'phonon-dfpt 1 1 1\nphonon-dfpt-displacement-symmetries {displacement}\n'
                  f'dump-name {name}.$VAR\ndump End None\n')
        results[name] = matrices(directory, name)
        if name == 'odd-accelerated':
            assert 'DFPT ground-state expansion: 14 reduced states -> 27 full-mesh states.' in log
            error = np.linalg.norm(results[name] - results['odd-baseline']) / np.linalg.norm(results['odd-baseline'])
            assert error < 2e-5, f'time-reversed checkpoint error {error:g}'
            initial_energy = float(re.search(r'ElecMinimize: Iter:\s*0\s+Etot:\s*(\S+)', log)[1])
            source_energy = float(re.findall(r'SCF: Cycle:.*?Etot:\s*(\S+)', (directory / 'odd-reduced.out').read_text())[-1])
            assert abs(initial_energy - source_energy) < 1e-9, 'time-reversed checkpoint changed the initial energy'
            print(f'Time-reversed odd-mesh checkpoint: force-matrix relative error {error:.3g}', flush=True)

    # Initialization and mismatch checks must work without running any responses.
    text = (directory / 'accelerated.in').read_text()
    log = run(directory, 'dry', text, dry=True)
    assert 'DFPT ground-state expansion:' in log and 'Dry run successful' in log
    log = run(directory, 'bad-mesh', text.replace('kpoint-folding 2 2 2', 'kpoint-folding 3 3 3'), success=False)
    assert 'incompatible electronic k mesh' in log or 'different k points or weights' in log
    log = run(directory, 'bad-cutoff', text.replace('elec-cutoff 8', 'elec-cutoff 9'), success=False)
    assert 'must match the lattice, cutoff and species' in log
    log = run(directory, 'missing', text.replace('reduced.in reduced.$VAR', 'reduced.in missing.$VAR'), success=False)
    assert 'requires both wavefunctions and fillings' in log
    (directory / 'acceleration.json').write_text(json.dumps({'seconds': timings, 'response_solves': solves}, indent=2) + '\n')
