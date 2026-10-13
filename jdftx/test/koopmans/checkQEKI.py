#!/usr/bin/env python3
"""Compare native KI(2) with QE KCW on identical Gamma-point H2 fixtures.

Usage: checkQEKI.py JDFTX_BUILD QE_BIN H_NC_UPF [--work-dir DIRECTORY]
Requires pw.x and kcw.x. Checks LDA-PZ/PBE, single-spin orbital screening,
occupied/empty Hamiltonian shifts and background/Ewald electrostatics.
This deliberately small numerical fixture is not a physical gap benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

EV = 27.211386245988


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('qe_bin', type=Path)
    parser.add_argument('pseudo', type=Path)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--scf-mesh', type=int, nargs=3, default=(1, 1, 1),
                        help='SCF mesh; screening remains Gamma with NSCF states at the retained SCF density')
    parser.add_argument('--collect-all', action='store_true',
                        help='Write all comparisons before failing if any original tolerance is exceeded')
    args = parser.parse_args()
    if min(args.scf_mesh) < 1:
        parser.error('SCF mesh must be positive')
    scf_grid = ' '.join(map(str, args.scf_mesh))
    separate_nscf = list(args.scf_mesh) != [1, 1, 1]
    build, qe_bin, pseudo = (p.resolve() for p in (args.build, args.qe_bin, args.pseudo))
    suffix = os.environ.get('JDFTX_TEST_EXECUTABLE_SUFFIX', '')
    for binary in (build/('jdftx'+suffix), build/('wannier'+suffix), qe_bin/'pw.x', qe_bin/'kcw.x'):
        if not binary.is_file():
            parser.error(f'Missing executable: {binary}')
    if not pseudo.is_file():
        parser.error(f'Missing norm-conserving hydrogen UPF: {pseudo}')
    temporary = tempfile.TemporaryDirectory(prefix='jdftx-qe-ki-') if not args.work_dir else None
    root = args.work_dir.resolve() if args.work_dir else Path(temporary.name)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error('--work-dir must be empty; comparison never reuses previous calculations')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    report = []

    def run(folder, label, executable, text, native=False):
        inp, out = folder/(label+'.in'), folder/(label+'.out')
        inp.write_text(text)
        if native:
            subprocess.run([str(executable), '-c', '1', '-i', inp.name, '-o', out.name],
                           cwd=folder, env=env, check=True, stdout=subprocess.DEVNULL, timeout=180)
        else:
            with out.open('w') as fp:
                subprocess.run([str(executable), '-in', inp.name], cwd=folder, env=env,
                               check=True, stdout=fp, stderr=subprocess.STDOUT, timeout=180)
        log = out.read_text()
        assert ('Done!' if native else 'JOB DONE.') in log, log[-4000:]
        return log

    for functional, qe_xc in [('lda-PZ', 'PZ'), ('gga-PBE', 'PBE')]:
        folder = root/functional
        folder.mkdir()
        common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
ion H 4 4 3.3 1
ion H 4 4 4.7 1
spintype no-spin
symmetries none
kpoint-reduce-inversion no
kpoint-folding {scf_grid}
fftbox 24 24 24
core-overlap-check none
elec-cutoff 6
elec-ex-corr {functional}
elec-n-bands 2
converge-empty-states yes
electronic-scf nIterations 200 nEigSteps 40 energyDiffThreshold 1e-13 residualThreshold 1e-10
electronic-minimize nIterations 300 energyDiffThreshold 1e-13
'''
        run(folder, 'scf', build/('jdftx'+suffix), common+
            'dump-name scf.$VAR\ndump End State BandEigs ElecDensity\n', native=True)
        qe_input = f'''&control
 calculation='scf', prefix='h2', outdir='./qe-tmp', pseudo_dir='{pseudo.parent}', verbosity='high'
/
&system
 ibrav=1, celldm(1)=8, nat=2, ntyp=1, ecutwfc=12, ecutrho=48,
 nr1=24, nr2=24, nr3=24, nr1s=24, nr2s=24, nr3s=24,
 nbnd=2, nspin=2, tot_magnetization=0, input_dft='{qe_xc}', nosym=.true., noinv=.true.
/
&electrons
 conv_thr=1d-13, mixing_beta=0.5, electron_maxstep=200, diago_thr_init=1d-10, diago_full_acc=.true.
/
ATOMIC_SPECIES
H 1.0 {pseudo.name}
ATOMIC_POSITIONS bohr
H 4 4 3.3
H 4 4 4.7
K_POINTS automatic
{scf_grid} 0 0 0
'''
        run(folder, 'pw', qe_bin/'pw.x', qe_input)
        state = 'scf'
        ki_common = common
        reference_density = ''
        if separate_nscf:
            ki_common = re.sub(r'^electronic-scf[^\n]*\n', '', common, flags=re.M).replace(
                f'kpoint-folding {scf_grid}', 'kpoint-folding 1 1 1')
            run(folder, 'nscf', build/('jdftx'+suffix), ki_common+
                'fix-electron-density scf.$VAR\ndump-name nscf.$VAR\ndump End State BandEigs\n', native=True)
            run(folder, 'pw-nscf', qe_bin/'pw.x', qe_input.replace("calculation='scf'", "calculation='nscf'").replace(
                f'{scf_grid} 0 0 0', '1 1 1 0 0 0'))
            state = 'nscf'
            reference_density = f'wannier-koopmans-reference-density {folder}/scf.n\n'
        native_dft = np.fromfile(folder/(state+'.eigenvals'), dtype=np.float64)
        xml = ET.parse(folder/'qe-tmp/h2.save/data-file-schema.xml')
        qe_dft = np.fromstring(xml.find('.//output/band_structure/ks_energies/eigenvalues').text, sep=' ')[:2]
        np.testing.assert_allclose(native_dft, qe_dft, atol=2e-6, rtol=0)

        def control(calculation, epsilon):
            return f'''&control
 calculation='{calculation}', prefix='h2', outdir='./qe-tmp', kcw_at_ks=.true.,
 read_unitary_matrix=.false., l_vcut={'.true.' if epsilon else '.false.'},
 spin_component=1, mp1=1, mp2=1, mp3=1
/
'''
        run(folder, 'wann2kcw', qe_bin/'kcw.x', control('wann2kcw', 0))
        for epsilon in (0, 2):
            log = run(folder, f'screen-{epsilon}', qe_bin/'kcw.x', control('screen', epsilon)+
                      f'&screen\n eps_inf={epsilon or 1}, tr2=1d-18, niter=100\n/\n')
            pattern = (r'iwann\s+=\s+(\d+)\s+relaxed =\s+([\d.eE+-]+)\s+'
                       r'unrelaxed =\s+([\d.eE+-]+)\s+alpha =\s+([\d.eE+-]+)')
            rows = re.findall(pattern, log)
            assert len(rows) == 2, log[-4000:]
            # QE curvatures are Ry. Alpha is dimensionless.
            qe = np.array([[float(bare)/2, float(relaxed)/2, float(alpha)]
                           for _, relaxed, bare, alpha in rows])
            ham = run(folder, f'ham-{epsilon}', qe_bin/'kcw.x', control('ham', epsilon)+
                      '&ham\n write_hr=.false., l_alpha_corr=.false.\n/\n')
            # Compare the active Hamiltonian output. The nspin=2 KCW XML
            # serialization is not used as the reference for band energies.
            ks = np.fromstring(re.search(r'^\s+KS  (.+)$', ham, re.M)[1], sep=' ')/EV
            ki = np.fromstring(re.search(r'^\s+KI  (.+)$', ham, re.M)[1], sep=' ')/EV
            for b in (0, 1):
                prefix = folder/f'ki-{epsilon}-{b}'
                run(folder, prefix.name, build/('wannier'+suffix), ki_common+f'''wannier-initial-state {folder}/{state}.$VAR
wannier-dump-name {prefix}.$VAR
wannier bStart {b}
wannier-center Gaussian 4 4 3.3
wannier-minimize nIterations 100 knormThreshold 1e-8
wannier-koopmans {epsilon} 400 1e-10
wannier-koopmans-spin yes
'''+reference_density, native=True)
                screen = np.loadtxt(str(prefix)+'.koopmansScreening')
                checks = [np.max(abs(screen[2:4]-qe[b, :2])) <= 8e-5, abs(screen[4]-qe[b, 2]) <= 8e-5]
                if not args.collect_all:
                    np.testing.assert_allclose(screen[2:4], qe[b, :2], atol=8e-5, rtol=0)
                    np.testing.assert_allclose(screen[4], qe[b, 2], atol=8e-5, rtol=0)
                kind = Path(str(prefix)+'.koopmansFormat').read_text().split()[0]
                delta = np.fromfile(str(prefix)+'.mlwfKI', dtype=np.float64 if kind == 'real64' else np.complex128)
                assert delta.size == 1
                checks.append(abs(delta[0].real-(ki[b]-ks[b])) <= 4e-5)
                if not args.collect_all:
                    np.testing.assert_allclose(delta[0].real, ki[b]-ks[b], atol=4e-5, rtol=0)
                expected_image = 2.837297479480619/8 if epsilon else 0.
                np.testing.assert_allclose(screen[5], expected_image, atol=2e-10, rtol=0)
                report.append(dict(functional=functional, epsilon=epsilon, orbital=b, scf_mesh=list(args.scf_mesh),
                                   native_curvatures_Ha=screen[2:4].tolist(), qe_curvatures_Ha=qe[b, :2].tolist(),
                                   native_alpha=float(screen[4]), qe_alpha=float(qe[b, 2]),
                                   native_shift_Ha=float(delta[0].real), qe_shift_Ha=float(ki[b]-ks[b]),
                                   passes_original_tolerances=bool(all(checks))))
            passed = all(row['passes_original_tolerances'] for row in report[-2:])
            print(f'{functional}, epsilon={epsilon}: QE/native comparison '+('passed' if passed else 'FAILED original tolerances'), flush=True)
    (root/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    if not all(row['passes_original_tolerances'] for row in report):
        raise AssertionError('QE/native differences exceed original tolerances; see comparison.json')
    print('QE KCW Gamma comparison passed. Finite-q gauge/matrix equivalence and full-KI DSCF are outside this fixture.')
    if temporary:
        temporary.cleanup()


if __name__ == '__main__':
    main()
