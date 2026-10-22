#!/usr/bin/env python3
"""Check completed coarse-screening alpha transfer and summarize orbital results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for block in iter(lambda: fp.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def table(path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 6 or not np.isfinite(data).all():
        raise ValueError(f'Invalid screening table: {path}')
    if not np.array_equal(data[:, 0], np.arange(len(data))):
        raise ValueError(f'Invalid orbital order: {path}')
    np.testing.assert_allclose(data[:, 3], data[:, 2]*data[:, 4], atol=1e-10, rtol=1e-10)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-directory', type=Path, help='Write summaries elsewhere when auditing an existing run')
    args = parser.parse_args()
    root = args.directory.resolve()
    config = json.loads((root/'manifest.json').read_text())['configuration']
    stages = json.loads((root/'stages.json').read_text())
    required = ['coarse-nscf'] + [f'{name}-{stage}' for name in ['valence', 'conduction']
        for stage in ['fine-orbitals', 'coarse-localize', 'coarse-screen', 'fine-hamiltonian']]
    for name in required:
        if name not in stages:
            raise ValueError(f'Missing completed stage: {name}')
        for path, sha in stages[name]['products'].items():
            if digest(path) != sha:
                raise ValueError(f'Output changed after stage completion: {path}')
    rows = []
    for name in ['valence', 'conduction']:
        coarse = table(root/f'coarse/{name}.koopmansScreening')
        imported = table(root/f'fine/{name}.alpha')
        fine = table(root/f'fine/{name}.koopmansScreening')
        mapping = json.loads((root/f'fine/{name}.alpha.mapping.json').read_text())
        permutation = np.array(mapping['fine_to_coarse'], dtype=int)
        if sorted(permutation.tolist()) != list(range(len(coarse))):
            raise ValueError(f'Invalid orbital map: {name}')
        np.testing.assert_allclose(imported[:, 1:], coarse[permutation, 1:], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fine[:, 4], imported[:, 4], atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(fine[:, 1], imported[:, 1])
        if mapping['coarse_screening_sha256'] != digest(root/f'coarse/{name}.koopmansScreening'):
            raise ValueError(f'Mapping references a different screening table: {name}')
        log = (root/f'logs/{name}-fine-hamiltonian.out').read_text()
        if 'no screening response solves' not in log:
            raise ValueError(f'Fine stage did not confirm alpha import: {name}')
        for i, row in enumerate(fine):
            rows.append({'manifold': name, 'orbital': i, 'coarse_orbital': int(permutation[i]),
                'occupation': int(row[1]), 'density_overlap': mapping['overlap'][i],
                'alpha': row[4], 'coarse_bare_Ha': imported[i, 2],
                'coarse_relaxed_Ha': imported[i, 3], 'fine_bare_Ha': row[2],
                'fine_alpha_scaled_curvature_Ha': row[3]})
    report = {'coarse_grid': config['coarse_grid'], 'fine_grid': config['fine_grid'],
        'backend': config['backend'], 'checks_passed': True, 'orbitals': rows,
        'note': 'Checks validate workflow integrity and alpha transfer; they do not certify physical convergence.'}
    output = args.output_directory.resolve() if args.output_directory else root
    output.mkdir(parents=True, exist_ok=True)
    (output/'coarse-screening-summary.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'coarse-screening-summary.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Coarse screening, orbital mapping, and fine-grid alpha import verified.')
    print(f'Wrote {output}/coarse-screening-summary.{{json,csv}}')


if __name__ == '__main__':
    main()
