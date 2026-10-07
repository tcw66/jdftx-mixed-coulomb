#!/usr/bin/env python3
"""Compare finite-q GGA DFPT against independently relaxed supercell forces.
Usage: checkResponse.py BUILD_DIR H_NORM_CONSERVING_UPF [XC_FUNCTIONAL] [slab|embedded]
"""
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

build = Path(sys.argv[1]).resolve()
pseudo = Path(sys.argv[2]).resolve()
functional = sys.argv[3] if len(sys.argv) > 3 else 'gga-PBE'
embedded = len(sys.argv) > 4 and sys.argv[4] == 'embedded'
slab = embedded or (len(sys.argv) > 4 and sys.argv[4] == 'slab')
axis = 0 if slab else 2
geometry = 'embedded slab' if embedded else ('slab' if slab else 'periodic')
script = Path(__file__).resolve().parents[2]/'scripts'/'phononDFPT'
loader = importlib.machinery.SourceFileLoader('phononDFPT', str(script))
spec = importlib.util.spec_from_loader(loader.name, loader)
interpolation = importlib.util.module_from_spec(spec)
loader.exec_module(interpolation)
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
common = f'''coords-type cartesian
ion-species {pseudo}
symmetries none
core-overlap-check none
elec-cutoff 8
elec-ex-corr {functional}
electronic-minimize nIterations 250 energyDiffThreshold 1e-13
dump-name result.$VAR
'''
if slab:
    common += 'coulomb-interaction Slab 001\n'
if embedded:
    common += 'coulomb-truncation-embed 0.3 0.4 8.137\n'


def run(directory, text):
    directory.mkdir()
    (directory/'input.in').write_text(text)
    subprocess.run([str(build/'jdftx'), '-i', 'input.in', '-o', 'output.out'],
                   cwd=directory, env=env, check=True, stdout=subprocess.DEVNULL)


with tempfile.TemporaryDirectory(prefix='jdftx-dfpt-') as name:
    root = Path(name)
    primitive = root/'primitive'
    positions = np.array([[3.3, 0, 7.7], [4.7, 0, 8.3]] if slab else [[0, 0, 3.3], [0, 0, 4.7]])
    if embedded:
        positions[:, 2] += 0.137
    ions = ''.join('ion H '+ ' '.join(map(str, xyz))+' 1\n' for xyz in positions)
    lattice = 'lattice Orthorhombic 8 8 16\nkpoint-folding 3 1 1\n' if slab else 'lattice Cubic 8\nkpoint-folding 1 1 3\n'
    grid = '3 1 1' if slab else '1 1 3'
    run(primitive, common + lattice + ions +
        'perturb-minimize nIterations 250 residualTol 1e-8 residualDiffThreshold 0\n'
        f'phonon-dfpt {grid}\ndump End None\n')
    tokens = np.loadtxt(primitive/'result.phononForceMatrixQ', skiprows=1, max_rows=1, dtype=int)
    nq, nm = tokens
    raw = np.fromstring(' '.join((primitive/'result.phononForceMatrixQ').read_text().splitlines()[2:]), sep=' ')
    chunks = raw.reshape(nq, 3+2*nm*nm)
    qs = chunks[:, :3]
    elements = chunks[:, 3:].reshape(nq, nm, nm, 2)
    force_q = elements[..., 0]+1j*elements[..., 1]
    cells, omega = interpolation.load(str(primitive/'result.'))
    dyn = interpolation.interpolate(cells, omega, qs)
    mass = interpolation.read_text_array(primitive/'result.phononBasis', dtype=str)[:, -1].astype(float)*1822.88839
    np.testing.assert_allclose(dyn*np.sqrt(mass[:, None]*mass[None, :]), force_q, atol=1e-11)
    np.testing.assert_allclose(force_q[1], force_q[2].conj(), atol=1e-11)
    translate = np.tile(np.eye(3), (2, 1))
    np.testing.assert_allclose(force_q[0]@translate, 0., atol=1e-12)
    assert np.max(np.abs(force_q[1].imag)) > 1e-5, 'complex off-diagonal phase was lost'
    # q=1/3: a displacement in one Cartesian direction of atom 0 in each cell.
    # cos(qR) and -sin(qR) isolate real and imaginary parts of the response.
    h = 2e-4
    for column in ((0, 2) if slab else (2,)):
        measured = np.zeros(6, dtype=complex)
        for component, pattern in enumerate((np.cos, lambda x: -np.sin(x))):
            forces = []
            for sign in (-1, 1):
                atom_text = ''
                for cell in range(3):
                    displaced = positions.copy()
                    displaced[:, axis] += 8*cell
                    displaced[0, column] += sign*h*pattern(2*np.pi*cell/3)
                    atom_text += ''.join('ion H '+ ' '.join(f'{x:.16g}' for x in xyz)+' 1\n' for xyz in displaced)
                directory = root/f'displaced-{column}-{component}-{sign}'
                supercell = 'lattice Orthorhombic 24 8 16\n' if slab else 'lattice Orthorhombic 8 8 24\n'
                run(directory, common+supercell+atom_text+'dump End Forces\n')
                rows = [line.split() for line in (directory/'result.force').read_text().splitlines()
                        if line.startswith('force ')]
                forces.append(np.array([[float(v) for v in row[2:5]] for row in rows])[:2].ravel())
            derivative = -(forces[1]-forces[0])/(2*h)
            measured += (1 if component == 0 else 1j)*derivative
        error = np.linalg.norm(measured-force_q[1, :, column])/np.linalg.norm(measured)
        assert error < 3e-3, f'finite-q {geometry} {functional} force response relative error {error:g}'
        print(f'Finite-q {geometry} {functional} DFPT versus supercell forces (column {column}): relative error {error:.3g}')
    if slab:
        source = root/'scf'
        source_text = common.replace('symmetries none', 'symmetries automatic')+lattice+ions
        run(source, source_text+'electronic-scf nIterations 250 energyDiffThreshold 1e-13\n'
            'dump End State Fillings\n')
        restart = root/'restart'
        restart_text = (common+lattice+ions+
            f'phonon-dfpt-initial-state {source}/input.in {source}/result.$VAR\n'
            'perturb-minimize nIterations 250 residualTol 1e-8 residualDiffThreshold 0\n'
            f'phonon-dfpt {grid}\ndump End None\n')
        run(restart, restart_text)
        raw_restart = np.fromstring(' '.join((restart/'result.phononForceMatrixQ').read_text().splitlines()[2:]), sep=' ')
        np.testing.assert_allclose(raw_restart, raw, atol=2e-6)
        print(f'Symmetry-reduced {geometry} SCF restart matches direct full-mesh response')
        # Geometry changes invalidate a checkpoint even when the electronic mesh matches.
        mismatch = root/'mismatch-source.in'
        mismatch.write_text(source_text.replace('coulomb-interaction Slab 001', 'coulomb-interaction Periodic'))
        invalid = {
            'q-normal': ((primitive/'input.in').read_text().replace('phonon-dfpt 3 1 1','phonon-dfpt 3 1 2'),
                         'q-grid dimension 1 along the truncated direction'),
            'restart-coulomb': (restart_text.replace(str(source/'input.in'),str(mismatch)),
                                'restart must match the Coulomb geometry'),
        }
        if embedded:
            invalid['restart-center'] = (restart_text.replace('coulomb-truncation-embed 0.3 0.4 8.137',
                'coulomb-truncation-embed 0.3 0.4 8.237'), 'restart must match the embedding center')
        for label, (text, message) in invalid.items():
            directory = root/label
            directory.mkdir()
            (directory/'input.in').write_text(text)
            result = subprocess.run([str(build/'jdftx'), '-n', '-i', 'input.in', '-o', 'output.out'],
                                    cwd=directory, env=env, stdout=subprocess.DEVNULL)
            assert result.returncode != 0 and message in (directory/'output.out').read_text(), label
        print('Unsupported normal q grid and incompatible Coulomb restart rejected')
    # Exercise uneven MPI distribution: three k points over two ranks.
    launcher = os.environ.get('DFPT_MPI_LAUNCH')
    if launcher:
        import shlex
        mpi_dir = root/'mpi'
        mpi_dir.mkdir()
        (mpi_dir/'input.in').write_text((primitive/'input.in').read_text())
        subprocess.run(shlex.split(launcher)+[str(build/'jdftx'), '-i', 'input.in', '-o', 'output.out'],
                       cwd=mpi_dir, env=env, check=True, stdout=subprocess.DEVNULL)
        raw_mpi = np.fromstring(' '.join((mpi_dir/'result.phononForceMatrixQ').read_text().splitlines()[2:]), sep=' ')
        np.testing.assert_allclose(raw_mpi, raw, atol=2e-7)
        print('MPI finite-q response matches serial result')
