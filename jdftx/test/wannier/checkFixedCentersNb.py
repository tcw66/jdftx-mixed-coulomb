#!/usr/bin/env python3
"""Compare hard-center CG/LBFGS/SQP using an existing bulk-Nb 12^3 SCF checkpoint.

All outputs and writable restart copies live in --work. Source files are read only.
The inner/outer windows are relative to the chemical potential in dfpt.out.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('build', type=Path)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--cache', type=Path, help='Optional same-dimension Wannier prefix for overlap reuse (rotations are initialized from trials)')
p.add_argument('--restart', type=Path, help='Compatible same-window Wannier prefix to resume')
p.add_argument('--method', choices=['CG', 'LBFGS', 'SQP', 'Both'], default='Both')
p.add_argument('--allow-unconverged-cg', action='store_true',
               help='Record a CG iteration-limit reference without failing; other methods must still converge')
p.add_argument('--measure', choices=['FiniteDifference', 'TruncatedDC'], default='TruncatedDC')
p.add_argument('--krylov', type=int, default=200)
p.add_argument('--history', type=int, default=15, help='LBFGS curvature history length')
p.add_argument('--orbitals', choices=['sd', 'spd'], default='sd', help='Six s/d or nine s/p/d atomic trials; spd requires the augmented 5p UPF')
p.add_argument('--inner-min', type=float)
p.add_argument('--inner-max', type=float)
p.add_argument('--outer-min', type=float)
p.add_argument('--outer-max', type=float)
p.add_argument('--radius', type=float, default=.5)
p.add_argument('--np', type=int, default=1)
p.add_argument('--iterations', type=int, default=500)
p.add_argument('--threshold', type=float, default=1e-6)
a = p.parse_args()
build, data, root = a.build.resolve(), a.data.resolve(), a.work.resolve()
root.mkdir(parents=True, exist_ok=False)
mu = float(re.findall(r'FillingsUpdate:\s+mu:\s*(\S+)', (data/'dfpt.out').read_text())[-1])
eigenvalues = np.fromfile(data/'phonon.eigenvals').reshape(-1, 20)
assert eigenvalues.shape == (1728, 20), eigenvalues.shape
inner = (mu-.15 if a.inner_min is None else a.inner_min, mu+.05 if a.inner_max is None else a.inner_max)
outer = (mu-.30 if a.outer_min is None else a.outer_min, mu+.45 if a.outer_max is None else a.outer_max)
orbitals = ['2s', 'dz2', 'dxy', 'dxz', 'dx2-y2', 'dyz']
if a.orbitals == 'spd':
    orbitals += ['2px', '2py', '2pz']
nw = len(orbitals)
inner_mask = (eigenvalues > inner[0]) & (eigenvalues < inner[1])
outer_counts = ((eigenvalues > outer[0]) & (eigenvalues < outer[1])).sum(axis=1)
assert inner_mask.sum(axis=1).max() <= nw and outer_counts.min() >= nw, (
    'Windows cannot support the chosen number of orbitals', nw,
    int(inner_mask.sum(axis=1).max()), int(outer_counts.min()))
common = (data/'common.in').read_text()
# Make the source's relative include paths absolute in the isolated run directory.
common = re.sub(r'^include\s+(\S+)', lambda m: 'include '+str(data/m[1]), common, flags=re.M)
(root/'common.in').write_text(common)
launcher = ['mpirun', '-np', str(a.np)] if a.np > 1 else []
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
results = {}
restart_hash = (hashlib.sha256(Path(str(a.restart.resolve())+'.mlwfU').read_bytes()).hexdigest()
                if a.restart else None)
for method in (['SQP', 'CG'] if a.method == 'Both' else [a.method]):
    directory = root/method.lower()
    directory.mkdir()
    prefix = directory/'wannier'
    if a.cache:
        for suffix in ('.mlwfM0', '.mlwfState'):
            shutil.copy2(str(a.cache.resolve())+suffix, str(prefix)+suffix)
    if a.restart:
        for suffix in ('.mlwfM0', '.mlwfU', '.mlwfState'):
            shutil.copy2(str(a.restart.resolve())+suffix, str(prefix)+suffix)
    text = f'''include {root}/common.in
symmetries none
wannier-initial-state {data}/phonon.$VAR
wannier-dump-name {prefix}.$VAR
wannier localizationMeasure {a.measure} fixCenters yes innerWindow {inner[0]} {inner[1]} outerWindow {outer[0]} {outer[1]} loadRotations yes saveWfns no
wannier-center-constraint-method {method}
wannier-center-constraint-sqp-krylov {a.krylov}
wannier-center-constraint-sqp-radius {a.radius}
wannier-center-constraint-solver MatrixFree
wannier-minimize nIterations {a.iterations} energyDiffThreshold 0 knormThreshold {a.threshold} history {a.history} fdTest yes
'''
    text += ''.join(f'wannier-center Nb 1 {orbital}\n' for orbital in orbitals)
    (directory/'wannier.in').write_text(text)
    start = time.monotonic()
    run = subprocess.run(launcher+[str(build/'wannier'), '-i', 'wannier.in', '-o', 'wannier.out'],
                         cwd=directory, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    wall = time.monotonic()-start
    out = (directory/'wannier.out').read_text()
    if run.returncode:
        raise RuntimeError(method+' failed\n'+out[-6000:]+'\n'+run.stderr)
    if a.restart:
        assert 'Reading initial rotations from' in out, 'Requested restart was not loaded'
    rows = re.findall(r'WannierMinimize: Iter:\s*(\d+)\s+Omega:\s*(\S+)\s+\|grad\|_K:\s*(\S+)\s+centerError:\s*(\S+)\s+t\[s\]:\s*(\S+)', out)
    assert rows
    objectives = np.array([float(r[1]) for r in rows])
    assert np.all(np.diff(objectives) <= 1e-10) and objectives[-1] < objectives[0]
    centers = np.loadtxt(str(prefix)+'.mlwfCenters', ndmin=2)
    center_error = float(np.max(np.linalg.norm(centers[:, :3], axis=1)))
    U = np.fromfile(str(prefix)+'.mlwfU', dtype=np.complex128, count=1728*20*nw).reshape(1728, nw, 20).swapaxes(1, 2)
    orth = float(np.max(np.linalg.norm(U.conj().swapaxes(1, 2)@U-np.eye(nw), axis=(1, 2))))
    inclusion = 0.
    for k, W in enumerate(U):
        F = np.eye(20)[:, inner_mask[k]]
        inclusion = max(inclusion, float(np.linalg.norm(F-W@(W.conj().T@F))))
    assert center_error < 1e-8 and max(orth, inclusion) < 1e-8
    assert max(float(r[3]) for r in rows) < 1e-8
    assert 'fdTest feasible derivative error' in out
    diagnostics = {}
    if method == 'SQP':
        assert 'fdTest SQP Hessian symmetry error' in out
        assert 'fdTest global phase projection' in out
        stats = re.search(r'SQP Hessian products (\d+) \((\S+) s\); trust trials (\d+); rejected (\d+); negative-curvature boundaries (\d+)', out)
        assert stats
        diagnostics = dict(hessian_products=int(stats[1]), hessian_seconds=float(stats[2]),
                           trust_trials=int(stats[3]), rejected_trials=int(stats[4]),
                           negative_curvature_boundaries=int(stats[5]))
    results[method] = dict(iterations=int(rows[-1][0]), objective_initial=float(rows[0][1]),
                           objective=float(rows[-1][1]), gradient=float(rows[-1][2]),
                           center_error_bohr=center_error, orthonormality_error=orth,
                           inner_window_error=inclusion, wall_seconds=wall,
                           iteration_seconds=float(rows[-1][4])-float(rows[0][4]),
                           converged='WannierMinimize: Converged' in out,
                           trajectory=[dict(iteration=int(r[0]), objective=float(r[1]),
                                            gradient=float(r[2]), center_error_bohr=float(r[3]),
                                            seconds=float(r[4])) for r in rows])
    results[method].update(diagnostics)
    summary = dict(mu_hartree=mu, inner_window_hartree=inner, outer_window_hartree=outer,
                   source=str(data), cache=str(a.cache.resolve()) if a.cache else None, kmesh=[12, 12, 12], bands=20,
                   centers=nw, orbitals=orbitals, ranks=a.np, krylov=a.krylov, history=a.history, initial_radius=a.radius, restart=str(a.restart.resolve()) if a.restart else None, restart_rotations_sha256=restart_hash, measure=a.measure, threshold=a.threshold, results=results)
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(method, json.dumps({k: v for k, v in results[method].items() if k != 'trajectory'}), flush=True)
    if method != 'CG' or not a.allow_unconverged_cg:
        assert results[method]['converged'] and results[method]['gradient'] < a.threshold
if a.method == 'Both':
    assert abs(results['CG']['objective_initial']-results['SQP']['objective_initial']) < 1e-9
    # Nonconvex localization can reach different stationary points. Report the
    # objective difference rather than assuming the methods choose the same basin.
    print('SQP - CG objective:', results['SQP']['objective']-results['CG']['objective'], flush=True)
print('Bulk Nb fixed-center regression passed in', root, flush=True)
