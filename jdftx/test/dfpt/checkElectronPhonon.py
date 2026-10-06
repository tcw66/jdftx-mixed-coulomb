#!/usr/bin/env python3
"""Physical DFPT electron-phonon regression: screened band derivatives and reciprocity."""
from contextlib import nullcontext
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

binary = Path(os.environ.get('JDFTX_BINARY', sys.argv[1])).resolve()
pseudo = Path(sys.argv[2]).resolve()
script = Path(__file__).resolve().parents[2]/'scripts'/'electronPhononDFPT'
loader = importlib.machinery.SourceFileLoader('electronPhononDFPT', str(script))
spec = importlib.util.spec_from_loader(loader.name, loader)
eph = importlib.util.module_from_spec(spec)
loader.exec_module(eph)
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
symmetries none
core-overlap-check none
elec-cutoff 8
elec-ex-corr gga-PBE
kpoint-folding 1 1 3
electronic-minimize nIterations 500 energyDiffThreshold 1e-14
electronic-scf nIterations 150 nEigSteps 30 energyDiffThreshold 0 eigDiffThreshold 0 residualThreshold 1e-10 mixFraction 0.4
dump-name result.$VAR
'''
response = '''perturb-minimize nIterations 500 residualTol 1e-9 residualDiffThreshold 0
phonon-dfpt 1 1 3
phonon-dfpt-electron-phonon yes
dump End None
'''


def run(directory, text, launch=()):
    directory.mkdir()
    (directory/'input.in').write_text(text)
    subprocess.run([*launch, str(binary), '-i', 'input.in', '-o', 'output.out'],
                   cwd=directory, env=env, check=True, stdout=subprocess.DEVNULL)


with (nullcontext(os.environ['DFPT_TEST_DIR']) if 'DFPT_TEST_DIR' in os.environ
      else tempfile.TemporaryDirectory(prefix='jdftx-eph-')) as name:
    root = Path(name)
    root.mkdir(parents=True, exist_ok=True)
    ions = 'ion H 0 0 3.3 1\nion H 0 0 4.7 1\n'
    for case, bands in [('insulator', 1), ('smeared', 10)]:
        settings = f'elec-n-bands {bands}\n'
        if case == 'smeared':
            settings += 'elec-smearing Fermi 0.025\n'
        directory = root/case
        run(directory, common + settings + ions + response)
        prefix = str(directory/'result.')
        qs, dimensions = eph.manifest(prefix)
        assert dimensions == (3, bands, 6)
        assert '3/3 representatives (18 displacement solves)' in (directory/'output.out').read_text()
        matrices = [eph.load_q(prefix, i, dimensions)[-1].copy() for i in range(3)]
        gamma = matrices[0]
        np.testing.assert_allclose(gamma, gamma.conj().transpose(0, 1, 3, 2), atol=2e-7)
        # Ground eigenvectors supply a common gauge at all commensurate k+q.
        # Inverting scattering swaps initial/final electronic bands.
        plus = eph.load_q(prefix, 1, dimensions)
        minus = eph.load_q(prefix, 2, dimensions)
        target_indices = plus[1][:, 4].astype(int)
        assert np.all(target_indices >= 0)
        np.testing.assert_allclose(matrices[1], matrices[2][:, target_indices].conj().transpose(0, 1, 3, 2),
                                   atol=3e-6, rtol=2e-4)
        # A diagonal screened Hamiltonian derivative is the derivative of the
        # corresponding nondegenerate band energy. SCF is repeated independently.
        step = 1e-3
        eigs = []
        for sign in (-1, 1):
            displaced = f'ion H 0 0 {3.3+sign*step:.16g} 1\nion H 0 0 4.7 1\n'
            scf = root/f'{case}-fd-{sign}'
            run(scf, common + settings + displaced + 'dump End BandEigs\n')
            eigs.append(np.fromfile(scf/'result.eigenvals', dtype='<f8').reshape(3, bands))
        measured = (eigs[1]-eigs[0])/(2*step)
        ncompare = 1 if case == 'insulator' else 2
        diagonal = np.diagonal(gamma[2], axis1=1, axis2=2)[:, :ncompare]
        error = np.linalg.norm(diagonal-measured[:, :ncompare])/np.linalg.norm(measured[:, :ncompare])
        assert error < 3e-3, f'{case}: screened band derivative relative error {error:g}'
        print(f'{case}: screened DFPT band derivative versus independent SCF finite differences: {error:.3g}')
        subprocess.run([sys.executable, str(script), prefix, '-o', str(directory/'couplings'), '--bands', '0', str(ncompare)], check=True)
        converted = np.load(directory/'couplings'/'q000001.npz')
        phonon = eph.phonon_module()
        cells, omega = phonon.load(prefix)
        dyn = phonon.interpolate(cells, omega, qs[1])[0]
        masses = phonon.read_text_array(prefix+'phononBasis', dtype=str)[:, -1].astype(float)
        # Undo zero-point normalization and phonon rotation: recover dH/du.
        valid = converted['valid_modes']
        factors = np.sqrt(2*np.sqrt(converted['omega_squared'][valid]))
        rotated = converted['g'][:, valid]/eph.HARTREE_MEV*factors[None, :, None, None]
        recovered = np.einsum('av,kvmn->akmn', converted['eigenvectors'][:, valid].conj(), rotated)
        recovered *= np.sqrt(masses*eph.AMU)[:, None, None, None]
        expected = matrices[1][:, :, :ncompare, :ncompare]
        # Compare the same retained-mode projection if the structure has unstable modes.
        projection = converted['eigenvectors'][:, valid].conj() @ converted['eigenvectors'][:, valid].T
        expected = np.einsum('ab,bkmn->akmn', projection,
                             expected/np.sqrt(masses*eph.AMU)[:, None, None, None])
        expected *= np.sqrt(masses*eph.AMU)[:, None, None, None]
        np.testing.assert_allclose(recovered, expected, atol=1e-12)
        assert np.isnan(converted['g'][:, ~valid]).all()
        if case == 'insulator':
            subprocess.run([sys.executable, str(script), prefix, '-o', str(directory/'ev'),
                            '--q-index', '1', '--units', 'eV'], check=True)
            ev = np.load(directory/'ev'/'q000001.npz')
            np.testing.assert_allclose(ev['g'], converted['g']/1000, equal_nan=True)
            np.testing.assert_allclose(ev['phonon_energies'], converted['phonon_energies']/1000)
            assert len(list((directory/'ev').glob('*.npz'))) == 1
        print(f'{case}: q/-q reciprocity and mass/frequency normalization passed')
        # Exercise off-mesh target diagonalization and its energy metadata.
        if case == 'smeared':
            offmesh = root/'offmesh'
            run(offmesh, common + settings + ions + response.replace('phonon-dfpt 1 1 3', 'phonon-dfpt 1 1 2'))
            q, ks, source, target, derivative = eph.load_q(str(offmesh/'result.'), 1)
            assert np.all(ks[:, 4] == -1) and np.isfinite(derivative).all()
            assert np.all(np.diff(target, axis=1) >= -1e-10)
            print('Off-mesh target band export passed')
            if len(sys.argv) > 3:
                mpi = root/'mpi'
                run(mpi, common + settings + ions + response + 'phonon-dfpt-q-groups 2\n',
                    [sys.argv[3], sys.argv[4], '4'])
                for iq in range(3):
                    _, _, mpi_source, mpi_target, other = eph.load_q(str(mpi/'result.'), iq, dimensions)
                    _, _, serial_source, serial_target, _ = eph.load_q(prefix, iq, dimensions)
                    # The highest stored band cuts a degenerate empty-state
                    # multiplet at Gamma. Compare the lowest seven bands, whose
                    # spectral boundary is separated at every source/target k,
                    # so rotations stay inside the compared subspace.
                    window = 7
                    for energies in (mpi_source, mpi_target, serial_source, serial_target):
                        assert np.min(energies[:, window]-energies[:, window-1]) > 1e-3
                    np.testing.assert_allclose(mpi_source[:, :window], serial_source[:, :window], atol=2e-7)
                    np.testing.assert_allclose(mpi_target[:, :window], serial_target[:, :window], atol=2e-7)
                    np.testing.assert_allclose(np.linalg.norm(other[:, :, :window, :window], axis=(2, 3)),
                                               np.linalg.norm(matrices[iq][:, :, :window, :window], axis=(2, 3)),
                                               rtol=2e-4, atol=3e-6)
                print('MPI electronic distribution and q groups preserve coupling norms')
