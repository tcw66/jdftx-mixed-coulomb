#!/usr/bin/env python3
"""Exercise SCF, exports, both fixed-H restart paths, and native Wannierization."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

exe, wannier, dataset = [Path(arg).resolve() for arg in sys.argv[1:4]]
env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
with tempfile.TemporaryDirectory(prefix="jdftx-paw-workflow-") as work:
    work = Path(work)
    common = f"""ion-species {dataset}
lattice Face-Centered Cubic 10.26
ion Si 0 0 0 0
ion Si .25 .25 .25 0
elec-cutoff 8
elec-n-bands 4
kpoint-folding 2 2 2
"""
    def run(binary, label, commands, success=True):
        inp, out = work / f"{label}.in", work / f"{label}.out"
        inp.write_text(commands)
        proc = subprocess.run([str(binary), "-i", str(inp), "-o", str(out)], cwd=work, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = out.read_text() if out.exists() else proc.stdout
        assert (proc.returncode == 0) == success, f"{label}: {log[-5000:]}\n{proc.stdout}"
        return log
    run(exe, "scf", common + f"""electronic-scf nIterations 40 energyDiffThreshold 1e-10 eigDiffThreshold 1e-8
 dump-name {work}/scf.$VAR
 dump End State ElecDensity Vscloc BandEigs AEWfns AEMatrices
 paw-reconstruction-cutoff 30
 """)
    meta = json.loads((work / "scf.paw.json").read_text())
    nq, nb = meta["states"], meta["bands"]
    eigs = np.fromfile(work / "scf.paw.eigenvals").reshape(nq, nb)
    assert np.array_equal(eigs.ravel(), np.fromfile(work / "scf.eigenvals"))
    mats = np.fromfile(work / "scf.paw.matrices", dtype=np.complex128).reshape(nq, 5, nb, nb).swapaxes(-1, -2)
    assert np.max(abs(mats[:, 0] - np.eye(nb))) < 1e-9
    assert np.max(abs(mats[:, 1] - np.array([np.diag(e) for e in eigs]))) < 1e-12
    assert np.max(abs(mats - mats.conj().swapaxes(-1, -2))) < 1e-10
    for q in range(nq):
        gs = np.loadtxt(work / f"scf.paw.q{q}.Gvectors", dtype=int)
        c = np.fromfile(work / f"scf.paw.q{q}.wfns", dtype=np.complex128).reshape(nb, len(gs))
        assert np.isfinite(c).all()
        norm = meta["volume"] * np.sum(abs(c)**2, axis=1)
        assert np.max(abs(norm - 1)) < .03, norm
    reader_path = Path(__file__).resolve().parents[2] / "scripts" / "read_paw.py"
    spec = importlib.util.spec_from_file_location("read_paw", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    exported = reader.PAWExport(work / "scf.paw")
    assert np.array_equal(exported.matrices, mats)
    _, coefficients = exported.wavefunction(0)
    u = exported.real_space(0)
    direct_norm = np.mean(abs(u)**2, axis=(1, 2, 3))
    assert np.max(abs(direct_norm - np.sum(abs(coefficients)**2, axis=1))) < 1e-12
    run(exe, "high-reconstruction", common + f"""initial-state {work}/scf.$VAR
dump-only
dump-name {work}/high.$VAR
dump End AEWfns AEMatrices
paw-reconstruction-cutoff 80
""")
    high = reader.PAWExport(work / "high.paw")
    def reconstruction_error(export):
        lattice = np.array(export.metadata["lattice_columns_bohr"]).T
        reciprocal = 2 * np.pi * np.linalg.inv(lattice)
        norm_error, momentum_error = 0., 0.
        for q in range(export.states):
            gs, coeff = export.wavefunction(q)
            vol = export.metadata["volume"]
            norm_error = max(norm_error, np.max(abs(vol * np.sum(abs(coeff)**2, axis=1) - 1)))
            k = (gs + np.array(export.metadata["kpoints_reciprocal"][q])) @ reciprocal
            for axis in range(3):
                actual = vol * (coeff.conj() * k[:, axis]) @ coeff.T
                momentum_error = max(momentum_error, np.max(abs(actual - export.matrices[q, 2 + axis])))
        return norm_error, momentum_error
    low_error, high_error = reconstruction_error(exported), reconstruction_error(high)
    assert high_error[0] < low_error[0], (low_error, high_error)
    assert high_error[1] < low_error[1], (low_error, high_error)
    print("Reconstruction norm/momentum errors:", low_error, "->", high_error)
    for kind in ["density", "potential"]:
        run(exe, kind, common + f"""fix-electron-{kind} {work}/scf.$VAR
wavefunction read {work}/scf.wfns
electronic-minimize nIterations 60 energyDiffThreshold 1e-11
dump-name {work}/{kind}.$VAR
dump End BandEigs
""")
        error = np.max(abs(np.fromfile(work / f"{kind}.eigenvals") - eigs.ravel()))
        assert error < 3e-5, (kind, error)
    run(wannier, "wannier", common + f"""wannier-initial-state {work}/scf.$VAR
wannier-dump-name {work}/w.$VAR
wannier-center Gaussian .125 .125 .125 1
wannier-center Gaussian -.375 .125 .125 1
wannier-center Gaussian .125 -.375 .125 1
wannier-center Gaussian .125 .125 -.375 1
wannier saveMomenta yes
wannier-minimize nIterations 20
""")
    rejected = run(wannier, "realspace-rejected", common + f"""wannier-initial-state {work}/scf.$VAR
wannier localizationMeasure RealSpace
""", False)
    assert "PAW Wannierization requires" in rejected
    assert (work / "w.mlwfH").stat().st_size > 0
    assert (work / "w.mlwfP").stat().st_size > 0
    assert np.isfinite(np.fromfile(work / "w.mlwfP")).all()
    run(exe, "spin-rejected", common + "spintype z-spin\n", False)
    run(exe, "forces-rejected", common + "dump End Forces\n", False)
    (work / "scf.pawState").unlink()
    run(exe, "missing-state", common + f"fix-electron-density {work}/scf.$VAR\n", False)
    print("PAW SCF/export/restart/Wannier workflow passed")
