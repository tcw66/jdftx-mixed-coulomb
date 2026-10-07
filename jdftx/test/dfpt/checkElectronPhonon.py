#!/usr/bin/env python3
"""Native DFPT el-ph format, physical derivatives, and Wannier regression."""
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

binary = Path(os.environ.get('JDFTX_BINARY', sys.argv[1])).resolve()
pseudo = Path(sys.argv[2]).resolve()
build = Path(sys.argv[1]).resolve().parent
wannier_binary = Path(os.environ.get('WANNIER_BINARY', build/'wannier')).resolve()
embedded = '--embedded' in sys.argv
mpi_args = [a for a in sys.argv[3:] if a != '--embedded']
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
symmetries none
core-overlap-check none
elec-cutoff 8
elec-ex-corr gga-PBE
electronic-minimize nIterations 500 energyDiffThreshold 1e-14
electronic-scf nIterations 150 nEigSteps 30 energyDiffThreshold 0 eigDiffThreshold 0 residualThreshold 1e-10 mixFraction 0.4
dump-name result.$VAR
'''
if embedded:
    common = common.replace('lattice Cubic 8', 'lattice Orthorhombic 8 8 16')
    common += '''coulomb-interaction Slab 001
coulomb-truncation-embed 0.3 0.4 8.137
coulomb-truncation-ion-margin 1
'''
response = '''perturb-minimize nIterations 500 residualTol 1e-9 residualDiffThreshold 0
phonon-dfpt-electron-phonon yes
dump End None
'''


def run(directory, text, launch=(), executable=binary, success=True):
    directory.mkdir()
    (directory/'input.in').write_text(text)
    result = subprocess.run([*launch, str(executable), '-i', 'input.in', '-o', 'output.out'],
                            cwd=directory, env=env, stdout=subprocess.DEVNULL)
    if success and result.returncode:
        raise RuntimeError((directory/'output.out').read_text()[-3000:])
    if not success:
        assert result.returncode, 'incompatible electronic mesh unexpectedly accepted'


def load(directory, nq, bands, states=True):
    assert not list(directory.glob('result.phononEph*'))
    kpoints = np.loadtxt(directory/'result.phononKpts', ndmin=2)
    assert kpoints.shape == (nq, 3)
    basis = np.loadtxt(directory/'result.phononBasis', dtype=str, ndmin=2, skiprows=1)
    scale = np.linalg.norm(basis[:, 2:5].astype(float), axis=1)
    assert (directory/'result.phononHsub').stat().st_size == 16*6*nq*nq*bands*bands
    # Band matrices are column-major; all outer dimensions are C ordered.
    h = np.fromfile(directory/'result.phononHsub', dtype=np.complex128).reshape(6, nq, nq, bands, bands).swapaxes(-1, -2)
    assert np.isfinite(h).all()
    np.testing.assert_allclose(h, h.conj().transpose(0, 2, 1, 4, 3), atol=3e-6, rtol=2e-4)
    for variable in (['wfns', 'eigenvals', 'fillings'] if states else []):
        assert (directory/f'result.{variable}').stat().st_size > 0
    return kpoints, scale, h


def wannier(directory, settings, ions, nq, source, dump='w'):
    grid = f'{nq} 1 1' if embedded else f'1 1 {nq}'
    center = '0 0 8' if embedded else '0 0 4'
    # The bond-centered H2 trial is already localized. Test the complete
    # native operator transform without optimizing its nearly flat spread.
    text = common+settings+ions+f'''wannier phononSupercell {grid}
wannier-initial-state {source}/result.$VAR
wannier-dump-name {dump}.$VAR
wannier-center Gaussian {center}
wannier-minimize nIterations 0
'''
    run(directory, text, executable=wannier_binary)
    matrix = np.fromfile(directory/f'{dump}.mlwfHePh', dtype=np.float64)
    assert matrix.size and np.isfinite(matrix).all()
    summed = np.fromfile(directory/f'{dump}.mlwfHePhSum', dtype=np.float64)
    assert summed.size and np.isfinite(summed).all()
    assert (directory/f'{dump}.mlwfCellMapPh').exists()
    return matrix


with (nullcontext(os.environ['DFPT_TEST_DIR']) if 'DFPT_TEST_DIR' in os.environ
      else tempfile.TemporaryDirectory(prefix='jdftx-eph-')) as name:
    root = Path(name)
    root.mkdir(parents=True, exist_ok=True)
    z = 7.3 if embedded else 3.3
    ions = f'ion H 0 0 {z} 1\nion H 0 0 {z+1.4} 1\n'
    cases = [('insulator', 1, 6, 2)] if embedded else [('insulator', 1, 6, 2), ('smeared', 10, 3, 3)]
    for case, bands, nk, nq in cases:
        settings = f'elec-n-bands {bands}\nkpoint-folding '+(f'{nk} 1 1\n' if embedded else f'1 1 {nk}\n')
        if bands > 1:
            settings += 'elec-smearing Fermi 0.025\n'
        qgrid = f'{nq} 1 1' if embedded else f'1 1 {nq}'
        directory = root/case
        run(directory, common+settings+ions+response+f'phonon-dfpt {qgrid}\n')
        ks, scale, h = load(directory, nq, bands)
        # Nk > Nq must store only the native subset, in its exact centered order.
        expected = np.zeros((nq, 3))
        expected[:, 0 if embedded else 2] = np.arange(nq)/nq
        expected -= np.floor(expected+0.5)
        np.testing.assert_allclose(ks, expected, atol=1e-7)
        source_indices = np.rint((ks[:, 0 if embedded else 2] % 1)*nk).astype(int)
        step = 1e-3
        eigs = []
        for sign in (-1, 1):
            displaced = f'ion H 0 0 {z+sign*step:.16g} 1\nion H 0 0 {z+1.4} 1\n'
            scf = root/f'{case}-fd-{sign}'
            run(scf, common+settings+displaced+'dump End BandEigs\n')
            eigs.append(np.fromfile(scf/'result.eigenvals', dtype=np.float64).reshape(nk, bands))
        measured = ((eigs[1]-eigs[0])/(2*step))[source_indices]
        gamma = np.stack([h[2, k, k] for k in range(nq)])*nq/scale[2]
        ncompare = 1 if bands == 1 else 2
        diagonal = np.diagonal(gamma, axis1=-2, axis2=-1)[:, :ncompare]
        error = np.linalg.norm(diagonal-measured[:, :ncompare])/np.linalg.norm(measured[:, :ncompare])
        assert error < 3e-3, f'{case}: screened derivative relative error {error:g}'
        print(f'{case}: native normalization, centered k order, reciprocity and band finite differences passed ({error:.3g})', flush=True)
        w = wannier(root/f'{case}-wannier', settings, ions, nq, directory)
        print(f'{case}: native Wannier produced {w.size} real-space coupling values', flush=True)
        if bands == 1 and not embedded:
            native = root/'native-phonon'
            run(native, common+settings+ions+f'''initial-state {directory}/result.$VAR
phonon supercell {qgrid} dr 0.001 saveHsub yes rSmooth 0.1
dump End None
''', executable=build/'phonon')
            native_ks, native_scale, reference = load(native, nq, bands, states=False)
            np.testing.assert_allclose(ks, native_ks)
            # Native phononBasis prints displacement vectors with six decimals.
            np.testing.assert_allclose(scale, native_scale, rtol=3e-5)
            error = np.linalg.norm(h-reference)/np.linalg.norm(reference)
            assert error < 3e-3, f'DFPT vs native frozen-phonon matrix error {error:g}'
            print(f'DFPT versus native phonon full complex matrices: relative error {error:.3g}', flush=True)
        if mpi_args and bands > 1:
            mpi = root/'mpi'
            run(mpi, common+settings+ions+response+f'phonon-dfpt {qgrid}\nphonon-dfpt-q-groups 2\n',
                [*mpi_args, '4'])
            _, _, other = load(mpi, nq, bands)
            # Compare whole separated manifolds; the last band cuts a multiplet.
            window = 7
            np.testing.assert_allclose(np.linalg.norm(other[..., :window, :window], axis=(-1, -2)),
                                       np.linalg.norm(h[..., :window, :window], axis=(-1, -2)), rtol=2e-4, atol=3e-7)
            # Reciprocity now compares complex entries across q groups, which
            # requires the shared gauge, rather than only scattering norms.
            mw = wannier(root/'mpi-wannier', settings, ions, nq, mpi)
            np.testing.assert_allclose(mw, w, rtol=2e-4, atol=3e-7)
            print('MPI q-group complex reciprocity and Wannier couplings passed', flush=True)
    bad = root/'incompatible-mesh'
    run(bad, common+settings+ions+response+'phonon-dfpt '+('4 1 1\n' if embedded else '1 1 2\n'), success=False)
    assert 'electronic k mesh containing the phonon q grid' in (bad/'output.out').read_text()
    print('Incompatible native export mesh rejected', flush=True)
