#!/usr/bin/env python3
"""Validate separate electronic-grid screening and imported KI Hamiltonians."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('build', type=Path)
    p.add_argument('pseudo', type=Path)
    p.add_argument('--launcher')
    p.add_argument('--work-dir', type=Path)
    a = p.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix='jdftx-coarse-workflow-') if not a.work_dir else None
    root = a.work_dir.resolve() if a.work_dir else Path(temporary.name)
    root.mkdir(parents=True, exist_ok=True)
    build, pseudo = a.build.resolve(), a.pseudo.resolve()
    suffix = os.environ.get('JDFTX_TEST_EXECUTABLE_SUFFIX', '')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               JDFTX_BINARY=str(build/('jdftx'+suffix)), WANNIER_BINARY=str(build/('wannier'+suffix)))
    if a.launcher: env['MPIEXEC'] = a.launcher
    driver = Path(__file__).resolve().parents[3]/'examples/koopmans/run_coarse_screening.py'

    def run(executable, text, label, error=None):
        inp, log = root/(label+'.in'), root/(label+'.out')
        inp.write_text(text)
        result = subprocess.run([str(build/(executable+suffix)), '-c', '1', '-i', str(inp), '-o', str(log)],
                                cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=180)
        output = log.read_text() if log.exists() else ''
        if error:
            assert result.returncode and error in output+result.stderr, output[-3000:]+result.stderr
        else:
            assert result.returncode == 0 and output.rstrip().endswith('Done!'), output[-3000:]+result.stderr
        return output

    common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
ion H 4 4 3.3 1
ion H 4 4 4.7 1
spintype no-spin
symmetries none
kpoint-folding 2 2 1
fftbox 24 24 24
core-overlap-check none
elec-cutoff 6
elec-ex-corr gga-PBE
elec-n-bands 4
converge-empty-states yes
electronic-scf nIterations 200 nEigSteps 40 energyDiffThreshold 1e-13 residualThreshold 1e-10
electronic-minimize nIterations 300 energyDiffThreshold 1e-13
'''
    (root/'state').mkdir(exist_ok=True); (root/'results').mkdir(exist_ok=True)
    (root/'common.in').write_text(common)
    run('jdftx', common+'dump-name state/scf.$VAR\ndump End State BandEigs ElecDensity\n', 'scf')
    (root/'ki-response.in').write_text('wannier-koopmans 2 400 1e-9 no 1e-6 yes 1 yes\nwannier-koopmans-spin yes\n')
    references = {}
    for name, number, band in [('valence', 2, 0), ('conduction', 3, 1)]:
        prefix = root/'results'/name
        text = f'''include common.in
wannier-initial-state state/scf.$VAR
wannier-dump-name {prefix}.$VAR
wannier bStart {band} saveWfns no loadRotations yes
wannier-center Gaussian 4 4 3.3
wannier-minimize nIterations 100 knormThreshold 1e-8
'''
        (root/f'{number:02d}-{name}-localize.in').write_text(text)
        run('wannier', text, name+'-localize')
        frozen = text.replace('nIterations 100', 'nIterations 0')+'include ki-response.in\n'
        run('wannier', frozen, name+'-full')
        kind = Path(str(prefix)+'.koopmansFormat').read_text().split()[0]
        dtype = np.float64 if kind == 'real64' else np.complex128
        screening = np.loadtxt(str(prefix)+'.koopmansScreening', ndmin=2)
        references[name] = (screening, np.fromfile(str(prefix)+'.mlwfKI', dtype=dtype), dtype)
        # Importing a same-grid result must reproduce its exact Hamiltonian.
        imported = frozen+f'wannier-koopmans-alpha-file {prefix}.koopmansScreening\n'
        log = run('wannier', imported, name+'-import')
        assert 'KI response solves: 0/0' in log and 'no screening response solves' in log
        np.testing.assert_allclose(np.fromfile(str(prefix)+'.mlwfKI', dtype=dtype), references[name][1], atol=3e-10, rtol=3e-9)
        # Zero and negative alphas are honored, including off-diagonal hopping.
        original = Path(str(prefix)+'.koopmansScreening').read_text()
        for scale in (0., -1.):
            table = screening.copy(); table[:, 3] *= scale; table[:, 4] *= scale
            source = root/(name+'-alpha.txt')
            source.write_text('\n'.join(line for line in original.splitlines() if line.startswith('#'))+'\n')
            with source.open('a') as stream: np.savetxt(stream, table, fmt='%.17g')
            run('wannier', frozen+f'wannier-koopmans-alpha-file {source}\n', name+'-scaled-'+str(scale))
            np.testing.assert_allclose(np.fromfile(str(prefix)+'.mlwfKI', dtype=dtype), scale*references[name][1], atol=3e-10, rtol=3e-9)
        for label, changed, error in [
            ('epsilon', original.replace('# epsilon_infinity 2;', '# epsilon_infinity 3;'), 'matching functional, spin, epsilon'),
            ('functional', original.replace('KI_second_order', 'KI'), 'matching functional, spin, epsilon'),
            ('spin', original.replace('single_spin_charge_and_magnetic', 'restricted_charge_only'), 'matching functional, spin, epsilon'),
            ('missing', '\n'.join(line for line in original.splitlines() if line.startswith('#')), 'complete orbital rows'),
            ('duplicate', original+next(line for line in original.splitlines() if line and not line.startswith('#'))+'\n', 'duplicate or out-of-range'),
            ('occupation', original.replace('\n0 '+str(1-band)+' ', '\n0 '+str(band)+' '), 'inconsistent occupation'),
            ('q-only', original.replace('screening_q_mesh 2 2 1', 'screening_q_mesh 1 1 1'), 'full coarse k/q mesh'),
        ]:
            source = root/(name+'-bad-'+label+'.txt'); source.write_text(changed)
            run('wannier', frozen+f'wannier-koopmans-alpha-file {source}\n', name+'-bad-'+label, error=error)
        # Exact reference-density loading gives the same fine-grid kernel.
        source = root/(name+'-original-alpha.txt'); source.write_text(original)
        run('wannier', frozen+f'wannier-koopmans-alpha-file {source}\nwannier-koopmans-reference-density {root}/state/scf.n\n', name+'-reference-density')
        np.testing.assert_allclose(np.loadtxt(str(prefix)+'.koopmansScreening', ndmin=2), screening, atol=3e-8, rtol=3e-7)

    # Full workflow: independent Gamma NSCF/Wannier screening from a 2x2x1
    # SCF density, then fine Hamiltonians without any fine DFPT responses.
    target = root/'workflow'
    command = [os.sys.executable, str(driver), '--example-dir', str(root), '--coarse-grid', '1', '1', '1',
               '--directory', str(target), '--minimum-density-overlap', '0.8', '--ranks', '2' if a.launcher else '1', '--threads', '1']
    if not suffix: command += ['--cpu']
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    assert not result.returncode, result.stdout
    for name in references:
        coarse = np.loadtxt(target/'coarse'/(name+'.koopmansScreening'), ndmin=2)
        fine = np.loadtxt(target/'fine'/(name+'.koopmansScreening'), ndmin=2)
        np.testing.assert_allclose(coarse[:, 4], fine[:, 4], atol=0, rtol=0)
        old, correction, dtype = references[name]
        # A one-orbital model's entire fine correction scales with alpha.
        np.testing.assert_allclose(np.fromfile(target/'fine'/(name+'.mlwfKI'), dtype=dtype),
                                   correction*fine[0, 4]/old[0, 4], atol=3e-8, rtol=3e-7)
        log = (target/'logs'/(name+'-fine-hamiltonian.out')).read_text()
        assert 'KI response solves: 0/0' in log
        assert not (target/'coarse'/(name+'.mlwfHki')).exists()
        assert 'source k/q mesh 1 1 1' in log
    # A matching resume must reuse all completed calculations.
    result = subprocess.run(command+['--resume'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    assert not result.returncode and 'Running ' not in result.stdout, result.stdout
    independent = command.copy()
    independent[independent.index('--coarse-grid')+1:independent.index('--coarse-grid')+4] = ['3', '3', '1']
    independent[independent.index('--directory')+1] = str(root/'independent-grid')
    result = subprocess.run(independent+['--prepare-only'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert result.returncode == 0, result.stdout

    spec = importlib.util.spec_from_file_location('coarse_driver', driver)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    match = root/'mapping'; match.mkdir(exist_ok=True)
    def densities(prefix, data):
        Path(str(prefix)+'.koopmansDensityFormat').write_text('float64 2 2 1 2 1\n1 0 0\n0 2 0\n0 0 1\n')
        np.asarray(data, dtype='<f8').tofile(str(prefix)+'.koopmansOrbitalDensities')
    def table(alpha):
        Path(str(match/'coarse')+'.koopmansScreening').write_text(
            '# functional KI_second_order\n# epsilon_infinity 0; electrostatics neutralizing_background\n'
            '# spin_treatment single_spin_charge_and_magnetic\n# screening_q_mesh 1 1 1; full_k_mesh 1 1 1\n'
            + ''.join(f'{i} 1 1 {x} {x} 0\n' for i, x in enumerate(alpha)))
    densities(match/'fine', [[1, 0], [0, 1]]); densities(match/'coarse', [[0, 1], [1, 0]]); table([.1, .8])
    assert module.transfer_alpha(match/'fine', match/'coarse', match/'alpha')['fine_to_coarse'] == [1, 0]
    densities(match/'fine', [[.5, .5], [.5, .5]]); densities(match/'coarse', [[.5, .5], [.5, .5]])
    try: module.transfer_alpha(match/'fine', match/'coarse', match/'alpha')
    except ValueError as exc: assert 'Ambiguous' in str(exc)
    else: raise AssertionError('Ambiguous unequal alphas accepted')
    table([.1, .1]); module.transfer_alpha(match/'fine', match/'coarse', match/'alpha')
    quality_log = match/'quality.out'
    quality_log.write_text('WannierMinimize: Iter: 8 |grad|_K: 2.0e-7\nWannierMinimize: Probably at roundoff error limit. (Stopping)\n')
    assert module.localization_quality(quality_log)['termination'] == 'roundoff_floor'
    quality_log.write_text(quality_log.read_text().replace('2.0e-7', '5.8'))
    try: module.localization_quality(quality_log)
    except ValueError: pass
    else: raise AssertionError('Large-gradient localization accepted')
    print('Separate coarse NSCF/Wannier screening, alpha import, density retention, mapping and resume passed.', flush=True)
    if temporary: temporary.cleanup()


if __name__ == '__main__': main()
