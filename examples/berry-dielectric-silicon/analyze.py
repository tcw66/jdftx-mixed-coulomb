#!/usr/bin/env python3
"""Central polarization differences and independent electric-enthalpy curvature."""
import argparse
import json
from pathlib import Path
import numpy as np


def minimized(output, prefix):
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    return bool(lines) and 'Converged' in lines[-1]


def read(path):
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith('#'):
            words = line.split()
            result[words[0]] = np.array([float(v) for v in words[1:]])
    return result


def aligned_change(data, zero, lattice, mesh):
    """Align individual string phases before averaging a weak-field response."""
    volume = np.linalg.det(lattice)
    if 'berry_phases' in data and 'berry_phases' in zero:
        np.testing.assert_array_equal(data['berry_phase_directions'], zero['berry_phase_directions'])
        np.testing.assert_array_equal(data['spin_weight'], zero['spin_weight'])
        phase = (data['berry_phases']-zero['berry_phases']+np.pi) % (2*np.pi)-np.pi
        if np.max(np.abs(phase)) >= np.pi/2:
            raise ValueError('field changes string phases too far for a zero-field derivative; reduce field')
        directions = data['berry_phase_directions'].astype(int)
        reduced = np.array([phase[directions == d].sum()/(np.prod(mesh)/mesh[d]) for d in range(3)])
        delta = data['spin_weight'][0] * (lattice @ reduced)/(2*np.pi*volume)
        delta += data['ionic']-zero['ionic']
    else:
        # Compatibility with previously generated polarization files.
        delta = data['total']-zero['total']
    reduced = volume*np.linalg.solve(lattice, delta)
    delta = lattice @ (reduced-np.rint(reduced))/volume
    # A branch shift at one string is a quantum/Ntransverse, rather than
    # necessarily a whole polarization quantum. Use the actual dipole shift.
    shift = data['total']-zero['total']-delta
    enthalpy = float(data['enthalpy'][0]) + volume*np.dot(data['field'], shift)
    return delta, enthalpy


def analyze(directory):
    metadata = json.loads((directory / 'workflow.json').read_text())
    lattice = np.array(metadata['lattice'])
    volume = np.linalg.det(lattice)
    zero = read(directory / 'zero.polarization')
    results = []
    for h in metadata['fields']:
        tensors = {}
        for mode in ['clamped', 'relaxed'] if metadata['relaxed'] else ['clamped']:
            tensor = np.full((3, 3), np.nan)
            curvature = {}
            for axis in metadata['axes']:
                beta = 'xyz'.index(axis)
                pair = []
                for sign in [-1, 1]:
                    job = next(j for j in metadata['jobs'] if j['mode'] == mode and j['axis'] == axis
                               and j['amplitude'] == h and j['sign'] == sign)
                    data = read(directory / (job['name'] + '.polarization'))
                    output = (directory / (job['name'] + '.out')).read_text().split('Start date and time:')[-1]
                    if not minimized(output, 'ElecMinimize:'):
                        raise ValueError(f'{job["name"]}: unconverged electrons')
                    if mode == 'relaxed' and not minimized(output, 'IonicMinimize:'):
                        raise ValueError(f'{job["name"]}: unconverged ions')
                    expected = np.eye(3)[beta] * sign * h
                    np.testing.assert_allclose(data['field'], expected, atol=1e-15)
                    delta, enthalpy = aligned_change(data, zero, lattice, np.array(metadata['mesh']))
                    pair.append((delta, enthalpy))
                tensor[:, beta] = np.eye(3)[:, beta] + 4*np.pi*(pair[1][0] - pair[0][0])/(2*h)
                curvature[axis] = 1 - 4*np.pi*(pair[1][1]+pair[0][1]-2*zero['enthalpy'][0])/(volume*h*h)
            tensors[mode] = tensor
            print(f'{directory.name}, h={h:g}, {mode}:\n{tensor}')
            print(f'Diagonal from enthalpy curvature: {curvature}')
            if len(metadata['axes']) == 3:
                print(f'Max tensor asymmetry: {np.max(np.abs(tensor-tensor.T)):.3e}')
            # Use None for uncomputed tensor columns in strict JSON.
            values = [[float(v) if np.isfinite(v) else None for v in row] for row in tensor]
            results.append(dict(amplitude=h, mode=mode, dielectric=values, curvature=curvature))
        if metadata['relaxed']:
            print(f'Lattice contribution:\n{tensors["relaxed"]-tensors["clamped"]}')
    (directory / 'dielectric.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    for directory in parser.parse_args().directories:
        analyze(directory)
