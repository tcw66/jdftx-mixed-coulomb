#!/usr/bin/env python3
"""Check native Wannier citation provenance with inexpensive dry runs."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=Path(__file__).resolve().parents[2]/'build')
    parser.add_argument('--gpu-build', type=Path)
    parser.add_argument('--mpi', action='store_true')
    args = parser.parse_args()
    build = args.build.resolve()
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    header = '# Berry-phase polarization: physical positive-ion convention, atomic units\n'
    cases = [('ordinary', None, [str(build/'wannier')], ''),
             ('unrelated', '# Other polarization output\n', [str(build/'wannier')], ''),
             ('berry', header, [str(build/'wannier')], ''),
             ('berry-duplicate', header, [str(build/'wannier')], 'berry-phase-field 0 0 0\n')]
    if args.mpi:
        cases.append(('berry-mpi', header, ['mpirun', '-n', '2', str(build/'wannier')], ''))
    if args.gpu_build:
        cases.append(('berry-gpu', header, [str(args.gpu_build.resolve()/'wannier_gpu')], ''))
    with tempfile.TemporaryDirectory(prefix='wannier-citation-') as temp:
        directory = Path(temp)
        common = (f'ion-species {build}/pseudopotentials/SG15/Si_ONCV_PBE.upf\n'
                  'coords-type lattice\nlattice 0 5.13 5.13 5.13 0 5.13 5.13 5.13 0\n'
                  'ion Si 0 0 0 1\nion Si .25 .25 .25 1\nsymmetries none\n'
                  'kpoint-reduce-inversion no\nkpoint-folding 2 2 2\n'
                  'elec-cutoff 12\nelec-n-bands 4\n'
                  f'wannier-initial-state {directory}/state.$VAR\n'
                  f'wannier-dump-name {directory}/wannier.$VAR\n')
        for center in [(0.125,0.125,0.125),(-.375,.125,.125),(.125,-.375,.125),(.125,.125,-.375)]:
            common += 'wannier-center Gaussian '+' '.join(map(str, center))+'\n'
        # Dry runs skip the wavefunctions but initialize the occupied eigenvalue array.
        (directory/'state.eigenvals').write_bytes(struct.pack('32d', *([0.]*32)))
        for name, metadata, command, extra in cases:
            sidecar = directory/'state.polarization'
            if metadata:
                sidecar.write_text(metadata)
            elif sidecar.exists():
                sidecar.unlink()
            inp = directory/'test.in'
            inp.write_text(common+extra)
            out = directory/f'{name}.out'
            subprocess.run(command+['-n', '-i', str(inp), '-o', str(out), '-c', '2'],
                           env=env, check=True)
            log = out.read_text()
            count = log.count('doi:10.1103/PhysRevLett.89.117602')
            expected = int(name.startswith('berry'))
            if count != expected or 'Dry run successful' not in log:
                raise AssertionError(f'{name}: expected {expected} PRL citation, got {count}\n{log}')
            print(f'{name}: passed (PRL citation count {count})')


if __name__ == '__main__':
    main()
