#!/usr/bin/env python3
"""Compare serial AE exports with uneven and empty MPI k-point ownership."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

exe, dataset, launcher, count_flag = sys.argv[1:5]
exe, dataset = Path(exe).resolve(), Path(dataset).resolve()
env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
with tempfile.TemporaryDirectory(prefix="jdftx-paw-export-mpi-") as directory:
    work = Path(directory)
    common = f"""ion-species {dataset}
lattice Face-Centered Cubic 10.26
ion Si 0 0 0 0
ion Si .25 .25 .25 0
elec-cutoff 8
elec-n-bands 4
kpoint-folding 2 2 2
"""

    def run(label, commands, ranks=1):
        inp, out = work / f"{label}.in", work / f"{label}.out"
        inp.write_text(common + commands)
        command = [str(exe), "-i", str(inp), "-o", str(out)]
        if ranks > 1:
            command = [launcher, count_flag, str(ranks)] + command
        proc = subprocess.run(command, cwd=work, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=90)
        log = out.read_text() if out.exists() else ""
        assert proc.returncode == 0, f"{label}: {log[-5000:]}\n{proc.stdout}"

    run("serial", f"""electronic-scf nIterations 40 energyDiffThreshold 1e-10 eigDiffThreshold 1e-8
dump-name {work}/serial.$VAR
dump End State AEWfns AERealSpaceWfns AEOverlap AEMomenta AEBandEigs
paw-reconstruction-cutoff 30
""")
    metadata = json.loads((work / "serial.paw.json").read_text())
    assert metadata["states"] == 3  # Two ranks have unequal work; four includes an idle rank.
    for ranks in (2, 4):
        label = f"mpi{ranks}"
        run(label, f"""initial-state {work}/serial.$VAR
dump-only
dump-name {work}/{label}.$VAR
dump End AEWfns AERealSpaceWfns AEOverlap AEMomenta AEBandEigs
paw-reconstruction-cutoff 30
""", ranks)
        assert json.loads((work / f"{label}.paw.json").read_text()) == metadata
        for suffix, dtype in [("eigenvals", np.float64), ("overlap", np.complex128),
                              ("momenta", np.complex128), ("wfns", np.complex128)]:
            reference = np.fromfile(work / f"serial.paw.{suffix}", dtype=dtype)
            actual = np.fromfile(work / f"{label}.paw.{suffix}", dtype=dtype)
            np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-11 if suffix == "wfns" else 2e-10)
        assert (work / f"{label}.paw.Gvectors").read_bytes() == (work / "serial.paw.Gvectors").read_bytes()
        for q in range(metadata["states"]):
            for band in range(metadata["bands"]):
                suffix = f"paw.q{q}.b{band}.rs"
                reference = np.fromfile(work / f"serial.{suffix}", dtype=np.complex128)
                actual = np.fromfile(work / f"{label}.{suffix}", dtype=np.complex128)
                np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-11)
    print("PAW serial/2-rank/4-rank AE exports agree")
