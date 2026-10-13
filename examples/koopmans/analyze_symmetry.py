#!/usr/bin/env python3
"""Audit finite-q KI symmetry and partition Gamma screening sources safely.

Copies Wannier checkpoints into an isolated directory; never solves response
equations or changes production outputs. --refine first improves localization.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def expand(path, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError(f'Recursive include: {path}')
    lines = []
    for line in path.read_text().splitlines():
        words = line.split()
        if words and words[0] == 'include':
            lines.append(expand(path.parent/words[1], (*stack, path)))
        else:
            lines.append(line)
    return '\n'.join(lines)+'\n'


def summarize(log):
    """Parse the native decisions, keeping every finite-q mapping inspectable."""
    rows, current = [], None
    for line in log.splitlines():
        match = re.search(r'KI orbital (\d+)/(\d+), q (\d+)/(\d+) \[([^]]+)\]', line)
        if match:
            orbital, count, iq, nk = map(int, match.groups()[:4])
            current = {'orbital': orbital, 'mesh_q': iq,
                       'q_fractional': list(map(float, match[5].split())), 'reused': False}
            rows.append(current)
        match = re.search(r'KI response reuse from orbital (\d+), q (\d+) \(relative density error (\S+)\)', line)
        if match and current is not None:
            current.update(reused=True, source_orbital=int(match[1]),
                           source_mesh_q=int(match[2]), density_error=float(match[3]))
    if not rows:
        raise ValueError('No screening-source audit records found in native output')
    gamma = [row for row in rows if max(map(abs, row['q_fractional'])) < 1e-12]
    parent = list(range(count+1))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for row in gamma:
        if row['reused'] and row['source_mesh_q'] == row['mesh_q']:
            parent[find(row['orbital'])] = find(row['source_orbital'])
    classes = {}
    for orbital in range(1, count+1):
        classes.setdefault(find(orbital), []).append(orbital)
    counts = re.search(r'KI (?:response solves|symmetry audit pairs): (\d+)/(\d+)', log)
    if not counts:
        raise ValueError('Missing final symmetry audit summary')
    tolerance = re.search(r'density tolerance (\S+)', log)
    return {'Gamma_screening_source_classes': sorted(classes.values()),
            'symmetry_tolerance': float(tolerance[1].rstrip('.')) if tolerance else None,
            'class_note': 'Classes certify Gamma source equivalence at the requested tolerance; '
                          'finite-q reuse is independently validated in each mapping below.',
            'independent_orbital_q_pairs': int(counts[1]),
            'total_orbital_q_pairs': int(counts[2]),
            'computed_q_points': len({row['mesh_q'] for row in rows if not row['reused']}),
            'maximum_accepted_density_error': max((row['density_error'] for row in rows if row['reused']), default=0.),
            'finite_q_mappings': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'03-conduction.in')
    parser.add_argument('--prefix', type=Path, default=ROOT/'results/conduction')
    parser.add_argument('--refine', action='store_true', help='Improve localization on the copied checkpoints')
    parser.add_argument('--diagnostics', action='store_true', help='Print nearest rejected source mappings (slower)')
    parser.add_argument('--symmetry-tolerance', type=float, help='Override the source/curvature reuse tolerance for this audit')
    parser.add_argument('--screening-mesh', type=int, nargs=3, metavar=('NX', 'NY', 'NZ'))
    parser.add_argument('--gpu', action='store_true', help='Use wannier_gpu (default: CPU)')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output', type=Path, help='New audit directory; default: timestamped archive directory')
    args = parser.parse_args()
    if args.threads < 1 or (args.screening_mesh and min(args.screening_mesh) < 1):
        parser.error('Threads and mesh dimensions must be positive')
    if args.symmetry_tolerance is not None and (not math.isfinite(args.symmetry_tolerance) or not 0 < args.symmetry_tolerance <= .01):
        parser.error('Symmetry tolerance must be finite and in (0, 0.01]')
    repo = ROOT.parents[1]
    binary = Path(os.environ.get('WANNIER_BINARY', repo/'build'/('wannier_gpu' if args.gpu else 'wannier'))).resolve()
    prefix, inp = args.prefix.resolve(), args.input.resolve()
    sources = [Path(str(prefix)+'.'+suffix) for suffix in ('mlwfU', 'mlwfM0')]
    state = Path(str(prefix)+'.mlwfState')
    if state.is_file():
        sources.append(state)
    for path in [binary, inp, *sources]:
        if not path.is_file():
            parser.error(f'Missing {path}')
    text = expand(inp)
    base = inp.parent
    text = re.sub(r'(?m)^wannier-initial-state\s+(\S+)',
                  lambda m: 'wannier-initial-state '+str((base/m[1]).resolve()), text)
    text = re.sub(r'\bnumericalOrbitals\s+(\S+)',
                  lambda m: 'numericalOrbitals '+str((base/m[1]).resolve()), text)
    output = args.output.resolve() if args.output else ROOT/'archive'/('symmetry-audit-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True, exist_ok=False)
    target = output/prefix.name
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for source in sources:
        shutil.copy2(source, output/source.name)
    text = re.sub(r'(?m)^wannier-dump-name\s+\S+', 'wannier-dump-name '+str(target)+'.$VAR', text)
    text = re.sub(r'\b(?:saveWfns|loadRotations)\s+(?:yes|no)', '', text)
    text = re.sub(r'(?m)^wannier\s+', 'wannier saveWfns no loadRotations yes ', text)
    minimize = ('nIterations 600 knormThreshold 1e-8 energyDiffThreshold 1e-14 convergeAll yes'
                if args.refine else 'nIterations 0 knormThreshold 1e-8')
    text, minimizers = re.subn(r'(?m)^wannier-minimize(?:[^\n]*\\\n)*[^\n]*', 'wannier-minimize '+minimize, text)
    if minimizers != 1:
        raise ValueError('Input must contain exactly one wannier-minimize command')
    def response(match):
        settings = match[1].split()
        defaults = ['0', '400', '1e-7', 'yes', '1e-6', 'yes', '1', 'yes']
        settings += defaults[len(settings):]
        settings[3], settings[6] = 'yes', '1'
        if args.symmetry_tolerance is not None:
            settings[4] = str(args.symmetry_tolerance)
        return 'wannier-koopmans '+' '.join(settings)
    text, commands = re.subn(r'(?m)^wannier-koopmans\s+([^\n#]+)', response, text)
    if commands != 1:
        raise ValueError('Input must contain exactly one wannier-koopmans command')
    text = re.sub(r'(?m)^wannier-koopmans-verbose[^\n]*', '', text)
    text += '\nwannier-koopmans-verbose yes\n'
    if args.screening_mesh:
        text = re.sub(r'(?m)^wannier-koopmans-screening-mesh[^\n]*', '', text)
        text += 'wannier-koopmans-screening-mesh '+' '.join(map(str, args.screening_mesh))+' 0\n'
    audit_input, log = output/'audit.in', output/'audit.out'
    audit_input.write_text(text)
    env = dict(os.environ, JDFTX_KI_SYMMETRY_AUDIT='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS=str(args.threads))
    if args.diagnostics:
        env['JDFTX_KI_SYMMETRY_DIAGNOSTICS'] = '1'
    env.setdefault('JDFTX_PSEUDO_DIR', str(repo/'build/pseudopotentials'))
    print(f'Auditing copied checkpoints in {output}', flush=True)
    with (output/'launcher.log').open('w') as stream:
        result = subprocess.run([str(binary), '-c', str(args.threads), '-i', str(audit_input), '-o', str(log)],
                                cwd=base, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode or not log.read_text().rstrip().endswith('Done!'):
        raise ValueError(f'Native audit failed; see {log} and {output}/launcher.log')
    contents = log.read_text()
    report = summarize(contents)
    gradients = re.findall(r'WannierMinimize: Iter:.*?\|grad\|_K:\s*(\S+)', contents)
    report.update(input=str(inp), prefix=str(prefix), refined=args.refine, source_sha256=hashes,
                  input_sha256=hashlib.sha256(text.encode()).hexdigest(), response_solves_performed=0,
                  final_localization_gradient=float(gradients[-1]) if gradients else None,
                  localization_roundoff_stop='Probably at roundoff error limit.' in contents)
    for path in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]:
            raise ValueError(f'Production checkpoint changed during the audit: {path}')
    (output/'symmetry-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('finite_q_mappings', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f'Symmetry audit failed: {error}')
