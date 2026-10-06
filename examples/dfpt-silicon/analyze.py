#!/usr/bin/env python3
"""Verify Fourier interpolation and plot silicon DFPT phonons."""
import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import re
import numpy as np


def load_tool(path):
    loader = importlib.machinery.SourceFileLoader('phononDFPT', str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def verify(directory, tool):
    prefix = str(directory/'phonon.')
    lines = (directory/'phonon.phononForceMatrixQ').read_text().splitlines()
    numbers = np.fromstring(' '.join(line for line in lines if not line.startswith('#')), sep=' ')
    nq, nm = map(int, numbers[:2])
    chunks = numbers[2:].reshape(nq, 3+2*nm*nm)
    q = chunks[:, :3]
    entries = chunks[:, 3:].reshape(nq, nm, nm, 2)
    force_q = entries[..., 0]+1j*entries[..., 1]
    cells, omega_r = tool.load(prefix)
    mass = tool.read_text_array(prefix+'phononBasis', str)[:, -1].astype(float)*1822.88839
    dyn = tool.interpolate(cells, omega_r, q)
    reconstructed = dyn*np.sqrt(mass[:, None]*mass[None, :])
    error = np.max(np.abs(reconstructed-force_q))
    scale = max(1e-30, np.max(np.abs(force_q)))
    if error > 1e-9 + 1e-7*scale:
        raise ValueError(f'coarse-q Fourier round trip failed: {error/scale:g}')
    gamma = np.argmin(np.linalg.norm(q, axis=1))
    if np.linalg.norm(q[gamma]) > 1e-12:
        raise ValueError('coarse mesh does not contain Gamma')
    translations = np.tile(np.eye(3), (nm//3, 1))
    asr = np.max(np.abs(force_q[gamma]@translations))
    if asr > 1e-9:
        raise ValueError(f'Gamma acoustic sum rule failed: {asr:g}')
    log = (directory/'dfpt.out').read_text()
    residuals = [float(x) for x in re.findall(r'DFPT final relative residual:\s*(\S+)', log)]
    reduction = re.search(r'DFPT q symmetry reduction:\s*(\d+)/(\d+) representatives(?: \((\d+) displacement solves\))?', log)
    representatives = int(reduction[1]) if reduction else nq
    expected_solves = int(reduction[3]) if reduction and reduction[3] else representatives*nm
    if not 1 <= representatives <= nq or (reduction and int(reduction[2]) != nq):
        raise ValueError('invalid DFPT q symmetry counts')
    groups = re.search(r'DFPT q groups completed:\s*(\d+) response solves; maximum relative residual:\s*(\S+)', log)
    if groups:
        if int(groups[1]) != expected_solves:
            raise ValueError('missing DFPT responses from q groups')
        residuals = [float(groups[2])]
    if (not groups and len(residuals) != expected_solves) or not np.isfinite(residuals).all():
        raise ValueError('missing or nonfinite DFPT response residuals')
    report = {'q_points': nq, 'modes': nm, 'real_space_cells': len(cells),
              'q_representatives': representatives,
              'response_solves': expected_solves,
              'fourier_relative_error': float(error/scale),
              'acoustic_sum_rule_error_Hartree_per_bohr2': float(asr),
              'max_response_relative_residual': max(residuals),
              'gamma_frequencies_cm-1': tool.frequencies(dyn[gamma:gamma+1])[0].tolist()}
    (directory/'checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Fourier round trip: {error/scale:.3e}; acoustic sum rule: {asr:.3e}')
    return report


def plot(directory, tool):
    # Matplotlib is only needed for plotting; use a headless backend.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metadata = json.loads((directory/'qpath.json').read_text())
    path = tool.read_text_array(directory/'qpath-distance.dat')
    bands = tool.read_text_array(directory/'frequencies.dat')
    if path.shape[0] != bands.shape[0] or not np.allclose(path[:, 1:], bands[:, :3], atol=1e-12):
        raise ValueError('q path does not match the interpolated frequencies')
    np.savetxt(directory/'dispersion.dat', np.column_stack((path[:, :1], bands)),
               header='distance[bohr^-1] q0 q1 q2 frequencies[cm^-1]; negative means imaginary')
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    breaks = metadata['breaks'] + [len(path)]
    for start, stop in zip(breaks[:-1], breaks[1:]):
        ax.plot(path[start:stop, 0], bands[start:stop, 3:], color='#195a9b', linewidth=1.)
    positions, labels = [], []
    for tick in metadata['ticks']:
        label = r'$\Gamma$' if tick['label'] == 'G' else tick['label']
        if positions and abs(tick['distance']-positions[-1]) < 1e-12:
            labels[-1] += ' | '+label
        else:
            positions.append(tick['distance'])
            labels.append(label)
    ax.set_xticks(positions, labels)
    for position in positions:
        ax.axvline(position, color='0.85', linewidth=.7, zorder=0)
    ax.axhline(0, color='0.35', linewidth=.7)
    ax.set_xlim(path[0, 0], path[-1, 0])
    ax.set_ylabel(r'Frequency (cm$^{-1}$)')
    ax.set_xlabel('Wavevector path')
    ax.set_title('Silicon phonons — DFPT and Fourier interpolation')
    fig.savefig(directory/'dispersion.png', dpi=180)
    fig.savefig(directory/'dispersion.pdf')
    plt.close(fig)
    print(f'Wrote {directory / "dispersion.png"} and dispersion.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--interpolator', type=Path,
                        default=Path(__file__).resolve().parents[2]/'jdftx/scripts/phononDFPT')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    try:
        tool = load_tool(args.interpolator)
        verify(args.directory, tool)
        if not args.no_plot:
            plot(args.directory, tool)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f'analyze.py: {error}\n')


if __name__ == '__main__':
    main()
