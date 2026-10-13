#!/usr/bin/env python3
"""Separate coarse NSCF/Wannier/DFPT and fine KI Hamiltonian, following koopmans."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

REFERENCE = '023fe26857466c5fae2520f5794dd5a86c416a84'
REPO = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def commands(path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError(f'Include cycle at {path}')
    # Native inputs in this workflow use unquoted paths and ordinary # comments.
    text = '\n'.join(line.split('#', 1)[0] for line in path.read_text().splitlines())
    text = re.sub(r'\\\s*\n', ' ', text)
    result = []
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == 'include':
            result.extend(commands(path.parent/words[1], (*stack, path)))
        else:
            result.append(' '.join(words))
    return result


def find_command(lines, key):
    matches = [line.split()[1:] for line in lines if line.split()[0] == key]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {key}, found {len(matches)}')
    return matches[0]


def density_data(prefix):
    words = Path(str(prefix)+'.koopmansDensityFormat').read_text().split()
    kind, count, size, *rest = words
    if kind != 'float64':
        raise ValueError('Unsupported orbital-density format')
    count, size = int(count), int(size)
    grid = np.array(list(map(int, rest[:3])))
    lattice = np.array(list(map(float, rest[3:]))).reshape(3, 3)
    data = np.fromfile(str(prefix)+'.koopmansOrbitalDensities', dtype='<f8').reshape(count, size)
    if size != np.prod(grid) or not np.isfinite(data).all():
        raise ValueError('Invalid orbital-density data')
    charge = data.sum(axis=1)*abs(np.linalg.det(lattice))/size
    np.testing.assert_allclose(charge, 1., atol=2e-6, rtol=2e-6)
    return data, grid, lattice


def transfer_alpha(fine, coarse, destination, minimum_overlap=.9, alpha_tolerance=1e-4, explicit_map=None):
    a, grid_a, lattice_a = density_data(fine)
    b, grid_b, lattice_b = density_data(coarse)
    if a.shape != b.shape or not np.array_equal(grid_a, grid_b) or not np.allclose(lattice_a, lattice_b, rtol=1e-12, atol=1e-12):
        raise ValueError('Fine/coarse orbital-density counts, grids or lattices differ')
    table_path = Path(str(coarse)+'.koopmansScreening')
    table = np.loadtxt(table_path, ndmin=2)
    if len(table) != len(a) or not np.array_equal(table[:, 0], np.arange(len(a))):
        raise ValueError('Coarse screening table does not match its orbital-density order')
    scores = (a @ b.T)/np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    if not np.isfinite(scores).all():
        raise ValueError('Invalid matching scores')
    rows, mapping = linear_sum_assignment(-scores)
    if explicit_map is not None:
        mapping = np.array(explicit_map, dtype=int)
        if len(mapping) != len(a) or sorted(mapping.tolist()) != list(range(len(a))):
            raise ValueError('Explicit orbital map must be a permutation of all zero-based coarse indices')
    matched = scores[np.arange(len(a)), mapping]
    report = {'method': 'Gamma-periodic density cosine overlap; alpha-equivalent permutations allowed',
              'fine_to_coarse': mapping.tolist(), 'overlap': matched.tolist(), 'scores': scores.tolist(),
              'minimum_overlap': minimum_overlap, 'alpha_equivalence_tolerance': alpha_tolerance,
              'fine_density_sha256': digest(str(fine)+'.koopmansOrbitalDensities'),
              'coarse_density_sha256': digest(str(coarse)+'.koopmansOrbitalDensities'),
              'coarse_screening_sha256': digest(table_path)}
    Path(str(destination)+'.mapping.json').write_text(json.dumps(report, indent=2)+'\n')
    if np.min(matched) < minimum_overlap:
        raise ValueError(f'Unreliable orbital correspondence: minimum density overlap {np.min(matched):.6g}; inspect {destination}.mapping.json or use a denser coarse grid')
    if explicit_map is None and len(a) > 1:
        best = scores[rows, mapping].sum()
        for i, j in enumerate(mapping):
            alternative = scores.copy(); alternative[i, j] = -1e6
            r, perm = linear_sum_assignment(-alternative)
            if best-scores[r, perm].sum() <= 1e-4*len(a):
                if np.max(abs(table[mapping, 4]-table[perm, 4])) > alpha_tolerance:
                    raise ValueError(f'Ambiguous matching of orbitals with different alphas; inspect {destination}.mapping.json and provide --orbital-map, or refine the coarse grid')
    header = [line for line in table_path.read_text().splitlines() if line.startswith('#')]
    output = '\n'.join(header)+f'\n# coarse_screening_sha256 {digest(table_path)}\n'
    for target, source in enumerate(mapping):
        row = table[source].copy(); row[0] = target
        output += '# source_orbital '+str(target)+' '+str(source)+'\n'
        output += ' '.join(f'{v:.17g}' for v in row)+'\n'
    Path(destination).write_text(output)
    return report


def localization_quality(log):
    text = Path(log).read_text()
    gradients = re.findall(r'WannierMinimize: Iter:.*?\|grad\|_K:\s*(\S+)', text)
    gradient = float(gradients[-1]) if gradients else float('inf')
    if re.search(r'WannierMinimize:\s+Converged', text):
        return {'termination': 'converged', 'last_gradient': gradient}
    # A failed line search is acceptable only at a measured numerical floor.
    # A large-gradient stop (observed on Si 2^3) remains a hard failure.
    if 'roundoff error limit. (Stopping)' in text and gradient <= 1e-6:
        return {'termination': 'roundoff_floor', 'last_gradient': gradient, 'acceptance_threshold': 1e-6}
    raise ValueError(f'Wannier localization did not converge (last gradient {gradient:.6g}); inspect {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--coarse-grid', '--dfpt-coarse-grid', type=int, nargs=3, required=True)
    parser.add_argument('--directory', type=Path, required=True, help='Isolated output directory')
    parser.add_argument('--scf-log', type=Path, help='Fine-grid log containing the chosen FFT grid')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--ranks', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--epsilon-infinity', type=float, help='Same electrostatic convention at coarse and fine stages')
    parser.add_argument('--minimum-density-overlap', type=float, default=.9)
    parser.add_argument('--orbital-map', type=Path, help='JSON with valence/conduction fine-to-coarse permutations')
    parser.add_argument('--plot', action='store_true', help='Plot the completed fine-grid Hamiltonians and search the gap')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', choices=('nscf', 'wannier', 'screening', 'hamiltonian'), default='hamiltonian')
    args = parser.parse_args()
    if min(*args.coarse_grid, args.ranks, args.threads) < 1 or not (0 < args.minimum_density_overlap <= 1):
        parser.error('Grid, ranks, threads and minimum overlap must be positive')
    if args.epsilon_infinity is not None and not (args.epsilon_infinity == 0 or math.isfinite(args.epsilon_infinity) and args.epsilon_infinity >= 1):
        parser.error('epsilon-infinity must be 0 or finite and >=1')
    root, folder = args.example_dir.resolve(), args.directory.resolve()
    if folder == root:
        parser.error('Select an isolated --directory')
    existing = folder.exists() and any(folder.iterdir())
    if existing and not (args.resume or args.prepare_only):
        parser.error('Output directory exists; use --resume or a new directory')
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'.run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('inputs', 'logs', 'coarse', 'fine', 'reference'):
        (folder/name).mkdir(exist_ok=True)
    common = commands(root/'common.in')
    fine_grid = list(map(int, find_command(common, 'kpoint-folding')))
    # These are separate electronic models; no subset/divisibility relation
    # between their k grids is required, unlike the q-only approximation.
    originals = {}
    for name, choices in [('valence', ('02-valence.in', '02-valence-localize.in')),
                          ('conduction', ('03-conduction.in', '03-conduction-localize.in'))]:
        path = next((root/p for p in choices if (root/p).exists()), None)
        if path is None:
            raise ValueError(f'Missing {name} localization input in {root}')
        originals[name] = commands(path)
    state_patterns = {str((root/find_command(lines, 'wannier-initial-state')[0]).resolve()) for lines in originals.values()}
    if len(state_patterns) != 1:
        raise ValueError('Manifolds use different SCF states')
    state = state_patterns.pop()
    source_files = [Path(state.replace('$VAR', suffix)) for suffix in ('wfns', 'eigenvals', 'n')]
    density_source = source_files[-1]
    for lines in originals.values():
        prefix = (root/find_command(lines, 'wannier-dump-name')[0].replace('.$VAR', '')).resolve()
        source_files.extend(Path(str(prefix)+'.'+s) for s in ('mlwfU', 'mlwfM0', 'mlwfState'))
        for line in lines:
            if line.startswith('wannier ') and 'numericalOrbitals' in line.split():
                words = line.split(); trial = (root/words[words.index('numericalOrbitals')+1]).resolve()
                source_files.extend((trial, Path(str(trial)+'.header')))
    state_hashes = {str(p): digest(p) for p in source_files}
    candidates = [args.scf_log] if args.scf_log else [root/p for p in ('scf.out', 'logs/scf.out', 'valence.out', 'logs/valence-localize.out')]
    grid = None
    for path in candidates:
        if path is None or not path.exists():
            continue
        matches = re.findall(r'Chosen fftbox size, S = \[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]', path.read_text())
        if matches:
            grid = list(map(int, matches[-1])); break
    if grid is None or np.prod(grid)*8 != density_source.stat().st_size:
        raise ValueError('Cannot establish the fine SCF density FFT grid; provide --scf-log')
    density = folder/'reference/scf.n'
    explicit = json.loads(args.orbital_map.read_text()) if args.orbital_map else {}
    config = {'reference_commit': REFERENCE, 'source_directory': str(root), 'coarse_grid': args.coarse_grid,
              'fine_grid': fine_grid, 'fft_grid': grid, 'source_states': state_hashes, 'source_inputs': originals,
              'epsilon_override': args.epsilon_infinity, 'minimum_density_overlap': args.minimum_density_overlap,
              'orbital_map': explicit, 'backend': 'cpu' if args.cpu else 'gpu', 'ranks': args.ranks, 'threads': args.threads}
    manifest_path = folder/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['configuration'] != config:
        raise ValueError('Model, state or workflow settings changed; select a new output directory')
    shutil.copyfile(density_source, density)
    manifest_path.write_text(json.dumps({'configuration': config, 'launcher_sha256': digest(__file__),
                                        'prepared_local': datetime.now().isoformat()}, indent=2)+'\n')
    excluded = {'kpoint-folding', 'electronic-scf', 'fftbox', 'dump', 'dump-name', 'initial-state',
                'fix-electron-density', 'fix-electron-potential'}
    model = [line for line in common if line.split()[0] not in excluded]
    model += ['fftbox '+' '.join(map(str, grid))]
    coarse_common = model+['kpoint-folding '+' '.join(map(str, args.coarse_grid))]
    nscf = coarse_common+[f'fix-electron-density {folder}/reference/scf.$VAR',
                          f'dump-name {folder}/coarse/scf.$VAR', 'dump End State BandEigs']
    (folder/'inputs/coarse-nscf.in').write_text('\n'.join(nscf)+'\n')
    input_paths = {}
    for name, lines in originals.items():
        # Every localization option/projection comes from the source workflow.
        body = [line for line in lines if line.startswith('wannier') and not line.startswith('wannier-koopmans')
                and line.split()[0] not in ('wannier-initial-state', 'wannier-dump-name')]
        frozen = [re.sub(r'nIterations\s+\d+', 'nIterations 0', line) if line.startswith('wannier-minimize ') else line for line in body]
        frozen = [re.sub(r'loadRotations\s+(yes|no)', 'loadRotations yes', line) if line.startswith('wannier ') else line for line in frozen]
        if not any('loadRotations' in line for line in frozen if line.startswith('wannier ')):
            frozen = [line+' loadRotations yes' if line.startswith('wannier ') else line for line in frozen]
        fine_prefix, coarse_prefix = folder/'fine'/name, folder/'coarse'/name
        source_prefix = (root/find_command(lines, 'wannier-dump-name')[0].replace('.$VAR', '')).resolve()
        for suffix in ('mlwfU', 'mlwfM0', 'mlwfState'):
            shutil.copyfile(Path(str(source_prefix)+'.'+suffix), Path(str(fine_prefix)+'.'+suffix))
        fine_text = common+[f'wannier-initial-state {state}', f'wannier-dump-name {fine_prefix}.$VAR']+frozen
        response = [line for line in lines if line.startswith('wannier-koopmans') and line.split()[0] not in
                    ('wannier-koopmans-screening-mesh', 'wannier-koopmans-alpha-file', 'wannier-koopmans-reference-density',
                     'wannier-koopmans-screening-only', 'wannier-koopmans-save-densities')]
        if not any(line.startswith('wannier-koopmans ') for line in response):
            response = commands(root/'ki-response.in')
            response = [line for line in response if line.split()[0] != 'wannier-koopmans-screening-mesh']
        for i, line in enumerate(response):
            if line.startswith('wannier-koopmans '):
                words = line.split()
                if args.epsilon_infinity is not None:
                    if len(words) == 1: words.append(str(args.epsilon_infinity))
                    else: words[1] = str(args.epsilon_infinity)
                if len(words) > 7: words[7] = str(min(int(words[7]), args.ranks))
                response[i] = ' '.join(words)
        coarse_body = [re.sub(r'loadRotations\s+(yes|no)', 'loadRotations no', line) if line.startswith('wannier ') else line for line in body]
        # Numerical projections are the same physical trial functions on
        # both grids. Native BlipResampler converts their supercell grid;
        # coarse Wannierization remains an independent optimization.
        coarse_text = coarse_common+[f'wannier-initial-state {folder}/coarse/scf.$VAR', f'wannier-dump-name {coarse_prefix}.$VAR']+coarse_body
        fine_export = fine_text+['wannier-koopmans-save-densities yes']
        coarse_localize = coarse_text+['wannier-koopmans-save-densities yes']
        coarse_screen = [re.sub(r'nIterations\s+\d+', 'nIterations 0', line) if line.startswith('wannier-minimize ') else line for line in coarse_text]
        coarse_screen = [re.sub(r'loadRotations\s+(yes|no)', 'loadRotations yes', line) if line.startswith('wannier ') else line for line in coarse_screen]
        if not any('loadRotations' in line for line in coarse_screen if line.startswith('wannier ')):
            coarse_screen = [line+' loadRotations yes' if line.startswith('wannier ') else line for line in coarse_screen]
        coarse_screen += response+[f'wannier-koopmans-reference-density {density}', 'wannier-koopmans-screening-only yes']
        fine_ham = fine_text+response+[f'wannier-koopmans-reference-density {density}', f'wannier-koopmans-alpha-file {folder}/fine/{name}.alpha']
        for stage, text in [('fine-orbitals', fine_export), ('coarse-localize', coarse_localize), ('coarse-screen', coarse_screen), ('fine-hamiltonian', fine_ham)]:
            path = folder/'inputs'/f'{name}-{stage}.in'; path.write_text('\n'.join(text)+'\n')
            input_paths[(name, stage)] = path
    print(f'Prepared coarse electronic/Wannier grid {args.coarse_grid}, fine Hamiltonian grid {fine_grid}: {folder}', flush=True)
    if args.prepare_only: return
    suffix = '' if args.cpu else '_gpu'
    binaries = {name: Path(os.environ.get(name.upper()+'_BINARY', REPO/'build'/(name+suffix))).resolve() for name in ('jdftx', 'wannier')}
    launch = [os.environ.get('MPIEXEC', '/usr/local/bin/mpiexec'), '-n', str(args.ranks)] if args.ranks > 1 else []
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS='1')
    status_path = folder/'stages.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    def stage(executable, inp, label, products):
        for path, sha in state_hashes.items():
            if digest(path) != sha: raise ValueError('Source SCF changed during the workflow')
        binary = binaries[executable]
        library = binary.parent/('libjdftx'+suffix+'.so')
        signature = {'input': digest(inp), 'executable': digest(binary), 'library': digest(library),
                     'reference_density': digest(density)}
        for line in inp.read_text().splitlines():
            if line.startswith('wannier-koopmans-alpha-file '): signature['alpha'] = digest(line.split()[1])
            if line.startswith('wannier-dump-name ') and 'nIterations 0' in inp.read_text():
                prefix = line.split()[1].replace('.$VAR', '')
                signature['rotations'] = digest(prefix+'.mlwfU')
            if line.startswith('wannier-initial-state '):
                pattern = line.split()[1]
                signature['wavefunctions'] = digest(pattern.replace('$VAR', 'wfns'))
                signature['eigenvalues'] = digest(pattern.replace('$VAR', 'eigenvals'))
            if line.startswith('wannier ') and 'numericalOrbitals' in line.split():
                words = line.split(); trial = root/words[words.index('numericalOrbitals')+1]
                signature['trial'] = digest(trial)
                signature['trial_header'] = digest(str(trial)+'.header')
        previous = status.get(label, {})
        if args.resume and previous.get('signature') == signature and previous.get('products') == {str(p): digest(p) for p in products if p.exists()} and len(previous.get('products', {})) == len(products):
            if label.endswith('coarse-localize'): localization_quality(folder/'logs'/(label+'.out'))
            print('Reusing '+label, flush=True); return
        log = folder/'logs'/(label+'.out')
        print('Running '+label, flush=True)
        with (folder/'logs'/(label+'.launcher.out')).open('w') as out:
            result = subprocess.run(launch+[str(binary), '-c', str(args.threads), '-i', str(inp), '-o', str(log)], cwd=root, env=env, stdout=out, stderr=out)
        if result.returncode or not log.read_text().rstrip().endswith('Done!') or not all(p.exists() for p in products):
            raise ValueError(f'{label} failed; inspect {log}')
        quality = localization_quality(log) if label.endswith('coarse-localize') else None
        status[label] = {'signature': signature, 'products': {str(p): digest(p) for p in products}, 'localization': quality}
        status_path.write_text(json.dumps(status, indent=2)+'\n')

    stage('jdftx', folder/'inputs/coarse-nscf.in', 'coarse-nscf', [folder/'coarse'/('scf.'+s) for s in ('wfns', 'eigenvals')])
    if args.stop_after == 'nscf': return
    if (folder/'inputs/coarse-seed.in').exists():
        stage('wannier', folder/'inputs/coarse-seed.in', 'coarse-seed', [folder/'coarse/conduction-seed.mlwfC'])
    for name in originals:
        for step, side in [('fine-orbitals', 'fine'), ('coarse-localize', 'coarse')]:
            stage('wannier', input_paths[(name, step)], name+'-'+step,
                  [folder/side/(name+'.'+s) for s in ('koopmansDensityFormat', 'koopmansOrbitalDensities', 'mlwfU', 'mlwfM0', 'mlwfState')])
    if args.stop_after == 'wannier': return
    for name in originals:
        stage('wannier', input_paths[(name, 'coarse-screen')], name+'-coarse-screen', [folder/'coarse'/(name+'.koopmansScreening')])
        transfer_alpha(folder/'fine'/name, folder/'coarse'/name, folder/'fine'/(name+'.alpha'),
                       args.minimum_density_overlap, explicit_map=explicit.get(name))
    if args.stop_after == 'screening': return
    for name in originals:
        stage('wannier', input_paths[(name, 'fine-hamiltonian')], name+'-fine-hamiltonian',
              [folder/'fine'/(name+'.'+s) for s in ('koopmansScreening', 'mlwfKI', 'mlwfHki', 'koopmansBands')])
    if args.plot:
        xc = find_command(common, 'elec-ex-corr')[0]
        label = {'gga-PBE': 'PBE', 'lda-PW': 'LDA-PW'}.get(xc, xc)
        subprocess.run([sys.executable, str(REPO/'examples/koopmans/plot_separate_bands.py'),
                        '--valence', str(folder/'fine/valence'), '--conduction', str(folder/'fine/conduction'),
                        '--fold', *map(str, fine_grid), '--output-dir', str(folder/'fine'),
                        '--dft-label', label], env=env, check=True)
    print('Completed separate coarse screening and fine Hamiltonians: '+str(folder/'fine'), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, AssertionError) as exc:
        sys.exit(str(exc))
