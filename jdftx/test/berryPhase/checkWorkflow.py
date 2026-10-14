#!/usr/bin/env python3
"""Physical finite-field regression: response, energy curvature, relaxation, guards."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

build, example = map(lambda x: Path(x).resolve(), sys.argv[1:3])
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

analysis = module('berry_analysis',example/'analyze.py')
runner = module('berry_runner',example/'run.py')
lattice = np.array([[0.,5.,5.],[5.,0.,5.],[5.,5.,0.]])
volume = np.linalg.det(lattice)
mesh = np.array([4,4,4])
directions = np.repeat(np.arange(3),16)
zero = dict(berry_phases=np.zeros(48),berry_phase_directions=directions,
            spin_weight=np.array([2]),ionic=np.array([.02,.03,.04]),total=np.array([.02,.03,.04]))
phases = np.zeros(48); phases[:16] = .002
delta_ionic = np.array([1e-6,2e-6,3e-6])
expected = 2*lattice[:,0]*.002/(2*np.pi*volume)+delta_ionic
shift = 2*lattice[:,0]/(16*volume)
field = np.array([0.,0.,.0001])
phases[0] += 2*np.pi # changing one string shifts averaged polarization by a fractional quantum
data = dict(zero,berry_phases=phases,ionic=zero['ionic']+delta_ionic,
            total=zero['total']+expected+shift,field=field,
            enthalpy=np.array([-1.4-volume*np.dot(field,shift)]))
actual, enthalpy = analysis.aligned_change(data,zero,lattice,mesh)
np.testing.assert_allclose(actual,expected,atol=1e-14)
np.testing.assert_allclose(enthalpy,-1.4,atol=1e-14)
# A quantum shift follows the strained cell. Use its lattice and preserve
# the geometric polarization change rather than reusing the reference volume.
strained = np.diag([1.01,.99,1.02]) @ lattice
strained_shift = 2*strained[:,0]/(16*np.linalg.det(strained))
data_cell = dict(data,lattice=strained.reshape(-1),total=zero['total']+expected+strained_shift,
                 enthalpy=np.array([-1.4-np.linalg.det(strained)*np.dot(field,strained_shift)]))
actual, enthalpy = analysis.aligned_change(data_cell,zero,lattice,mesh)
np.testing.assert_allclose(actual,expected,atol=1e-14)
np.testing.assert_allclose(enthalpy,-1.4,atol=1e-14)
ratios = runner.field_gap_ratios([.001],['z'],[3,4,5],lattice,.02)
np.testing.assert_allclose(ratios[0]['directional_ratios'],[.75,1.,0.])

with tempfile.TemporaryDirectory(prefix='jdftx-berry-') as tmp:
    root = Path(tmp)
    subprocess.run([sys.executable, str(example / 'run.py'), '--build', str(build), '--work', str(root),
                    '--mesh', '4', '--fields', '0.0002', '0.0001', '--cutoff', '12', '--axes', 'z',
                    '--threads', '2', '--relax-cell', '--run'], check=True)
    work = root / 'k4'
    # Cite the finite-field method only in runs that enable Berry enthalpy.
    for output in work.glob('*.out'):
        expected_citations = 0 if output.stem == 'gap' else 1
        assert output.read_text().count('doi:10.1103/PhysRevLett.89.117602') == expected_citations, output
    rows = json.loads((work / 'dielectric.json').read_text())
    by_mode = {mode: [r for r in rows if r['mode'] == mode] for mode in ['clamped', 'relaxed', 'cell-relaxed']}
    for mode in by_mode:
        values = [r['dielectric'][2][2] for r in by_mode[mode]]
        assert all(8. < value < 11. for value in values), (mode, values)
        assert abs(values[0]-values[1]) < 0.01, ('field linearity', values)
        for row in by_mode[mode]:
            assert abs(row['dielectric'][2][2]-row['curvature']['z']) < 0.02, ('enthalpy curvature', row)
    for clamped, relaxed in zip(by_mode['clamped'], by_mode['relaxed']):
        assert abs(clamped['dielectric'][2][2]-relaxed['dielectric'][2][2]) < 0.002, 'Si lattice contribution'
    for cell, relaxed in zip(by_mode['cell-relaxed'], by_mode['relaxed']):
        assert abs(cell['dielectric'][2][2]-relaxed['dielectric'][2][2]) < 0.1, 'Si cell contribution'
    reference = analysis.read(work / 'zero.polarization')
    assert np.linalg.norm(reference['total']) < 1e-6, 'Si nonpolar branch'
    for name in ['equilibrate', 'cell-zero'] + [j['name'] for j in json.loads((work/'workflow.json').read_text())['jobs']
                                             if j['mode'] == 'cell-relaxed']:
        cell = analysis.read(work / f'{name}.polarization')
        stress = np.loadtxt(work / f'{name}.stress')
        assert np.max(np.abs(stress))*np.linalg.det(cell['lattice'].reshape(3,3)) < 1.1e-6, name
    # Parameter errors should be detected in dry runs, without evaluating orbitals.
    input_zero = (work / 'zero.in').read_text()
    for label, contents, diagnostic in [
        ('empty', input_zero.replace('elec-n-bands 4', 'elec-n-bands 5'), 'fully occupied'),
        ('scf', input_zero.replace('berry-phase-field 0 0 0', 'berry-phase-field 0 0 0.0001') +
         'electronic-scf nIterations 10\n', 'requires self-consistent electronic-minimize'),
        ('smearing', input_zero.replace('elec-n-bands 4', 'elec-n-bands 8') +
         'elec-smearing Fermi 0.001\n', 'constant, fully occupied'),
        ('empty_cleanup', input_zero + 'converge-empty-states yes\n', 'forbids converge-empty-states'),
        ('dfpt', input_zero + 'perturb-minimize nIterations 1\n', 'finite-field DFPT Hessians'),
    ]:
        (work / f'{label}.in').write_text(contents)
        result = subprocess.run([str(build / 'jdftx'), '-n', '-i', f'{label}.in', '-o', f'{label}.out'],
                                cwd=work, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode != 0 and diagnostic in (work / f'{label}.out').read_text(), label
print('Berry finite-field silicon workflow regressions passed.')
