#!/usr/bin/env python3
"""Small Si integration/gradient regression. Usage: checkTruncatedDC.py BUILD_DIR.
Runs in a temporary directory; requires the SG15 silicon pseudopotential.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

build = Path(sys.argv[1]).resolve()
pseudo = build / 'pseudopotentials/SG15/Si_ONCV_PBE.upf'
env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
with tempfile.TemporaryDirectory(prefix='wannier-tdc-') as tmp:
    root = Path(tmp)
    common = f'''ion-species {pseudo}
lattice Face-Centered Cubic 10.26
ion Si 0 0 0 1
ion Si 0.25 0.25 0.25 1
elec-cutoff 5
elec-n-bands 8
kpoint-folding 2 2 2
'''
    (root / 'scf.in').write_text(common + 'dump-name scf.$VAR\ndump End State BandEigs\nelectronic-minimize nIterations 50\n')
    def run(exe, name):
        prefix = shlex.split(os.environ.get('WANNIER_MPI', '')) if exe == 'wannier' else []
        subprocess.run(prefix + [str(build / exe), '-i', name+'.in', '-o', name+'.out'],
                       cwd=root, env=env, check=True, stdout=subprocess.DEVNULL)
        return (root / (name+'.out')).read_text()
    run('jdftx', 'scf')
    for name, window in [('isolated', 'bStart 0'),
                         ('entangled', 'outerWindow -100 100 innerWindow -100 -0.15'),
                         ('unconstrained', 'outerWindow -100 100')]:
        text = common + f'''wannier-initial-state scf.$VAR
wannier-dump-name {name}.$VAR
wannier localizationMeasure TruncatedDC {window} saveWfns no
wannier-center Gaussian 0.10 0.16 0.13
wannier-center Gaussian 0.62 0.11 0.16
wannier-center Gaussian 0.15 0.59 0.10
wannier-center Gaussian 0.11 0.14 0.64
wannier-minimize nIterations 30 fdTest yes knormThreshold 1e-7
'''
        (root / (name+'.in')).write_text(text)
        out = run('wannier', name)
        ratios = [float(x) for x in re.findall(r'fdTest:.*Ratio:\s*(\S+)', out)]
        assert len(ratios) >= 6, out[-4000:]
        assert min(abs(x-1) for x in ratios[2:7]) < 1e-4, ratios
        energies = [float(x) for x in re.findall(r'WannierMinimize:.*Omega:\s*(\S+)', out)]
        assert energies and energies[-1] <= energies[0] + 1e-9, out[-4000:]
        assert 'TruncatedDC spreads' in out
        print(name, 'gradient ratios:', ratios[2:7], 'spread:', energies[0], '->', energies[-1])
    (root / 'pinned.in').write_text(text.replace('wannier-center Gaussian', 'wannier-center-pinned Gaussian', 1))
    result = subprocess.run([str(build / 'wannier'), '-i', 'pinned.in', '-o', 'pinned.out'],
                            cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert result.returncode != 0
    assert 'does not support pinned' in (root / 'pinned.out').read_text()
