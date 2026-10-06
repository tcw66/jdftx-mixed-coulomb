#!/usr/bin/env python3
"""Compare smeared metallic DFPT against independently relaxed supercell forces.
Usage: checkMetallic.py BUILD_DIR H_NORM_CONSERVING_UPF [SMEARING]
"""
from contextlib import nullcontext
import importlib.machinery
import importlib.util
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

build = Path(sys.argv[1]).resolve()
binary = Path(os.environ.get('JDFTX_BINARY', str(build/'jdftx'))).resolve()
pseudo = Path(sys.argv[2]).resolve()
functional = 'gga-PBE'
smearing = sys.argv[3] if len(sys.argv) > 3 else 'Fermi'
width = 0.025 if smearing == 'Fermi' else 0.05
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
elec-smearing {smearing} {width}
electronic-minimize nIterations 500 energyDiffThreshold 1e-14
electronic-scf nIterations 150 nEigSteps 30 energyDiffThreshold 0 eigDiffThreshold 0 residualThreshold 1e-10 mixFraction 0.4
dump-name result.$VAR
'''


def run(directory, text):
    directory.mkdir()
    (directory/'input.in').write_text(text)
    subprocess.run([str(binary), '-i', 'input.in', '-o', 'output.out'],
                   cwd=directory, env=env, check=True, stdout=subprocess.DEVNULL)


with (nullcontext(os.environ['DFPT_TEST_DIR']) if 'DFPT_TEST_DIR' in os.environ
      else tempfile.TemporaryDirectory(prefix='jdftx-dfpt-metal-')) as name:
    root = Path(name)
    root.mkdir(parents=True, exist_ok=True)
    primitive = root/'primitive'
    ions = 'ion H 0 0 3.3 1\nion H 0 0 4.7 1\n'
    run(primitive, common + 'lattice Cubic 8\nkpoint-folding 1 1 3\nelec-n-bands 10\n' + ions +
        'perturb-minimize nIterations 500 residualTol 1e-8 residualDiffThreshold 0\n'
        'phonon-dfpt 1 1 3\ndump End None\n')
    log = (primitive/'output.out').read_text()
    assert 'Metallic DFPT: smearing density-matrix response' in log
    charges = [float(v) for v in re.findall(r'Metallic DFPT Gamma charge response: ([0-9.eE+-]+)', log)]
    assert charges and max(charges) < 1e-10, f'Gamma charge conservation failed: {charges}'
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
    measured = np.zeros(6, dtype=complex)
    for component, pattern in enumerate((np.cos, lambda x: -np.sin(x))):
        forces = []
        for sign in (-1, 1):
            atom_text = ''
            for cell in range(3):
                displacement = sign*h*pattern(2*np.pi*cell/3)
                atom_text += f'ion H 0 0 {3.3+8*cell+displacement:.16g} 1\n'
                atom_text += f'ion H 0 0 {4.7+8*cell:.16g} 1\n'
            directory = root/f'displaced-{component}-{sign}'
            run(directory, common+'lattice Orthorhombic 8 8 24\nelec-n-bands 30\n'+atom_text+'dump End Forces\n')
            rows = [line.split() for line in (directory/'result.force').read_text().splitlines()
                    if line.startswith('force ')]
            forces.append(np.array([[float(v) for v in row[2:5]] for row in rows])[:2].ravel())
        derivative = -(forces[1]-forces[0])/(2*h)
        measured += (1 if component == 0 else 1j)*derivative
    error = np.linalg.norm(measured-force_q[1, :, 2])/np.linalg.norm(measured)
    assert error < 3e-3, f'finite-q {smearing} metallic force response relative error {error:g}'
    print(f'Finite-q {smearing} metallic DFPT versus supercell forces: relative error {error:.3g}')
    gamma_forces = []
    for sign in (-1, 1):
        directory = root/f'gamma-{sign}'
        displaced = f'ion H 0 0 {3.3+sign*h:.16g} 1\nion H 0 0 4.7 1\n'
        run(directory, common+'lattice Cubic 8\nkpoint-folding 1 1 3\nelec-n-bands 10\n'+
            displaced+'dump End Forces\n')
        rows = [line.split() for line in (directory/'result.force').read_text().splitlines()
                if line.startswith('force ')]
        gamma_forces.append(np.array([[float(v) for v in row[2:5]] for row in rows]).ravel())
    measured_gamma = -(gamma_forces[1]-gamma_forces[0])/(2*h)
    gamma_error = np.linalg.norm(measured_gamma-force_q[0, :, 2].real)/np.linalg.norm(measured_gamma)
    assert gamma_error < 3e-3, f'Gamma {smearing} metallic force response relative error {gamma_error:g}'
    print(f'Gamma {smearing} DFPT versus fixed-charge primitive-cell forces: relative error {gamma_error:.3g}')
    # Exercise uneven MPI distribution: three k points over two ranks.
    launcher = os.environ.get('DFPT_MPI_LAUNCH')
    if launcher:
        import shlex
        mpi_dir = root/'mpi'
        mpi_dir.mkdir()
        (mpi_dir/'input.in').write_text((primitive/'input.in').read_text())
        subprocess.run(shlex.split(launcher)+[str(binary), '-i', 'input.in', '-o', 'output.out'],
                       cwd=mpi_dir, env=env, check=True, stdout=subprocess.DEVNULL)
        raw_mpi = np.fromstring(' '.join((mpi_dir/'result.phononForceMatrixQ').read_text().splitlines()[2:]), sep=' ')
        np.testing.assert_allclose(raw_mpi, raw, atol=2e-7)
        print('MPI finite-q response matches serial result')
