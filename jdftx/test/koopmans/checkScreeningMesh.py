#!/usr/bin/env python3
"""Check coarse-q quadrature, full-mesh Hamiltonians and SCF restart identity.

Usage: checkScreeningMesh.py BUILD H_NC_UPF [--launcher MPIEXEC]
Set JDFTX_TEST_EXECUTABLE_SUFFIX=_gpu for CUDA. Results use temporary storage.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('pseudo', type=Path)
    parser.add_argument('--launcher')
    parser.add_argument('--work-dir', type=Path)
    args = parser.parse_args()
    build, pseudo = args.build.resolve(), args.pseudo.resolve()
    suffix = os.environ.get('JDFTX_TEST_EXECUTABLE_SUFFIX', '')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    temporary = tempfile.TemporaryDirectory(prefix='jdftx-ki-screening-') if not args.work_dir else None
    folder = args.work_dir.resolve() if args.work_dir else Path(temporary.name)
    folder.mkdir(parents=True, exist_ok=True)

    def run(executable, text, label, ranks=1, error=None):
        inp, out = folder/(label+'.in'), folder/(label+'.out')
        inp.write_text(text)
        out.unlink(missing_ok=True)
        command = [str(build/(executable+suffix)), '-c', '1', '-i', inp.name, '-o', out.name]
        if ranks > 1:
            command = [args.launcher, '-n', str(ranks)]+command
        result = subprocess.run(command, cwd=folder, env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, timeout=180)
        log = out.read_text() if out.exists() else ''
        if error:
            assert result.returncode and error in log+result.stderr, log[-4000:]+result.stderr
        else:
            assert not result.returncode and log.rstrip().endswith('Done!'), log[-4000:]+result.stderr
            progress = re.findall(r'KI group (\d+)/(\d+): computed q (\d+)/(\d+), mesh q (\d+)/(\d+)', log)
            computed = {}
            for group, groups, counter, total, iq, mesh_size in progress:
                seen = computed.setdefault(int(group), {})
                if int(iq) not in seen:
                    seen[int(iq)] = len(seen)+1
                assert int(counter) == seen[int(iq)], (counter, iq, seen)
                assert 1 <= int(counter) <= int(total)
                assert re.search(rf'KI group {group}/{groups}: {total} independent q points assigned\.', log)
            for group, seen in computed.items():
                assert re.search(rf'KI group {group}/\d+: {len(seen)} independent q points assigned\.', log)
                assert re.search(rf'KI group {group}/\d+: computed q points total: {len(seen)}\.', log)
        return log

    def read(prefix):
        kind, _, _ = Path(str(prefix)+'.koopmansFormat').read_text().split()
        dtype = np.float64 if kind == 'real64' else np.complex128
        return [np.loadtxt(str(prefix)+'.koopmansScreening', ndmin=2),
                np.loadtxt(str(prefix)+'.koopmansBands', ndmin=2),
                np.fromfile(str(prefix)+'.mlwfKI', dtype=dtype)]

    def check_quadrature(prefix, nq, epsilon=2):
        table = np.loadtxt(str(prefix)+'.koopmansResponse', ndmin=2)
        screening = read(prefix)[0]
        for row in screening:
            orbital = int(row[0])
            q = table[table[:, 1] == orbital]
            selected = q[q[:, 8] == 1]
            assert len(q) == 9 and len(selected) == nq
            full_bare = (q[:, 5]+q[:, 7]).mean()
            screened_bare = (selected[:, 5]+selected[:, 7]).mean()
            screened_relaxed = (selected[:, 5]+selected[:, 6]+selected[:, 7]/epsilon).mean()
            np.testing.assert_allclose(row[2], full_bare, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(row[4], screened_relaxed/screened_bare, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(row[3], row[4]*full_bare, atol=2e-12, rtol=2e-12)
        return table

    common = f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
ion H 4 4 3.3 1
ion H 4 4 4.7 1
spintype no-spin
symmetries automatic
kpoint-folding 3 3 1
fftbox 24 24 24
core-overlap-check none
elec-cutoff 6
elec-ex-corr gga-PBE
elec-n-bands 4
converge-empty-states yes
electronic-scf nIterations 200 nEigSteps 40 energyDiffThreshold 1e-13 residualThreshold 1e-10
electronic-minimize nIterations 300 energyDiffThreshold 1e-13
'''
    run('jdftx', common+'dump-name scf.$VAR\ndump End State BandEigs\n', 'scf')
    for empty in (False, True):
        prefix = folder/('empty' if empty else 'valence')
        base = common+f'''wannier-initial-state {folder}/scf.$VAR
wannier-dump-name {prefix}.$VAR
wannier bStart {1 if empty else 0} saveWfns no
wannier-center Gaussian 4 4 3.3
wannier-minimize nIterations 100 knormThreshold 1e-8
'''
        run('wannier', base, prefix.name+'-localize')
        frozen = base.replace('saveWfns no', 'saveWfns no loadRotations yes').replace('nIterations 100 knormThreshold', 'nIterations 0 knormThreshold')
        response = 'wannier-koopmans 2 400 1e-9 no 1e-6 yes 1 yes\nwannier-koopmans-spin yes\n'
        run('wannier', frozen+response, prefix.name+'-full')
        reference = read(prefix)
        full_q = check_quadrature(prefix, 9)
        run('wannier', frozen+response+'wannier-koopmans-screening-mesh 3 3 1\n', prefix.name+'-explicit-full')
        for value, expected in zip(read(prefix), reference):
            np.testing.assert_allclose(value, expected, atol=3e-10, rtol=3e-9)
        for mesh, nq in [('1 3 1', 3), ('1 1 1', 1)]:
            coarse = frozen+response+f'wannier-koopmans-screening-mesh {mesh}\n'
            run('wannier', coarse, prefix.name+'-coarse-'+str(nq))
            coarse_reference = read(prefix)
            table = check_quadrature(prefix, nq)
            # Bare contributions use the full Wannier mesh in every case;
            # selected responses agree with the exhaustive fine-q calculation.
            np.testing.assert_allclose(table[:, 5], full_q[:, 5], atol=3e-10, rtol=3e-9)
            selected = table[:, 8] == 1
            np.testing.assert_allclose(table[selected, 6], full_q[selected, 6], atol=3e-8, rtol=3e-7)
            np.testing.assert_allclose(table[:, 7], full_q[:, 7], atol=3e-10, rtol=3e-9)
            reduced = coarse.replace('1e-9 no 1e-6', '1e-9 yes 1e-6')
            log = run('wannier', reduced, prefix.name+'-reduced-'+str(nq))
            assert int(re.search(r'KI response solves: \d+/(\d+)', log)[1]) == nq
            for value, expected in zip(read(prefix), coarse_reference):
                np.testing.assert_allclose(value, expected, atol=3e-8, rtol=3e-7)
            if args.launcher:
                mpi_input = reduced.replace('yes 1 yes\n', 'yes 2 yes\n')
                log = run('wannier', mpi_input, prefix.name+'-mpi-'+str(nq), ranks=2)
                assert ('2 active' if nq > 1 else '1 active') in log
                for value, expected in zip(read(prefix), coarse_reference):
                    np.testing.assert_allclose(value, expected, atol=3e-8, rtol=3e-7)
                check_quadrature(prefix, nq)
        run('wannier', frozen+response+'wannier-koopmans-screening-mesh 2 3 1\n', prefix.name+'-invalid-mesh',
            error='must divide kpoint-folding')
        print(prefix.name+': coarse weights, full bare Hamiltonian, symmetry and MPI agreement passed', flush=True)

    # A byte change is caught even when path, size and modification time stay
    # fixed. Global phase leaves the physical state unchanged for comparison.
    wfns = folder/'scf.wfns'
    original = wfns.read_bytes()
    times = wfns.stat()
    wfns.write_bytes((-np.frombuffer(original, dtype=np.complex128)).tobytes())
    os.utime(wfns, ns=(times.st_atime_ns, times.st_mtime_ns))
    prefix = folder/'valence'
    base = common+f'''wannier-initial-state {folder}/scf.$VAR
wannier-dump-name {prefix}.$VAR
wannier bStart 0 loadRotations yes saveWfns no
wannier-center Gaussian 4 4 3.3
wannier-minimize nIterations 0 knormThreshold 1e-8
'''
    run('wannier', base, 'changed-scf-zero-iterations', error='requires nonzero wannier-minimize')
    log = run('wannier', base.replace('nIterations 0', 'nIterations 100'), 'changed-scf-relocalize')
    assert 'restart invalidated' in log and 'Computing initial overlaps' in log
    wfns.write_bytes(original)
    # Legacy U may be newer than the SCF while its cached M0 is stale, as in
    # the production regression. Both checkpoints must be checked together.
    Path(str(prefix)+'.mlwfState').unlink()
    old_ns = wfns.stat().st_mtime_ns-1_000_000_000
    os.utime(str(prefix)+'.mlwfM0', ns=(old_ns, old_ns))
    new_ns = wfns.stat().st_mtime_ns+1_000_000_000
    os.utime(str(prefix)+'.mlwfU', ns=(new_ns, new_ns))
    run('wannier', base, 'legacy-stale-overlap', error='requires nonzero wannier-minimize')
    print('SCF content identity and legacy stale-overlap invalidation passed.', flush=True)
    if temporary:
        temporary.cleanup()


if __name__ == '__main__':
    main()
