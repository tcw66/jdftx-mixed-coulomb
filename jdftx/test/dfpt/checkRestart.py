#!/usr/bin/env python3
"""Check occupied-band restart counting on a dense k mesh."""
from array import array
import os
from pathlib import Path
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
pseudo = Path(sys.argv[2]).resolve()
with tempfile.TemporaryDirectory(prefix='jdftx-fillings-restart-') as name:
    directory = Path(name)
    (directory/'input.in').write_text(f'''lattice Cubic 8
coords-type cartesian
ion-species {pseudo}
ion H 0 0 3.3 1
ion H 0 0 4.7 1
symmetries none
core-overlap-check none
elec-cutoff 8
kpoint-folding 6 6 6
elec-n-bands 1
elec-initial-fillings read fillings
''')
    # The 216 weights produce a total just above an integer in floating point.
    # A filled band must pass; a materially overfilled band must still fail.
    for filling, should_pass in ((2., True), (2.00002, False)):
        with (directory/'fillings').open('wb') as stream:
            array('d', [filling]*216).tofile(stream)
        result = subprocess.run([str(binary), '-c', '1', '-d', '-n', '-i', 'input.in', '-o', 'output.out'],
                                cwd=directory, env=dict(os.environ, OMP_NUM_THREADS='1'),
                                capture_output=True, text=True)
        log = (directory/'output.out').read_text()
        if should_pass:
            assert result.returncode == 0 and 'Dry run successful' in log, log
        else:
            assert result.returncode != 0 and 'bands insufficient' in log, log
print('Dense-mesh occupied-band restart and overfilled-band rejection passed.')
