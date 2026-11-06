#!/usr/bin/env python3
"""Independent AE Bloch -> Wannier/FFT/operator checks, symmetry and MPI exports."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

exe, wannier, dataset = [Path(x).resolve() for x in sys.argv[1:4]]
launcher = sys.argv[4:6]
env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
spec = importlib.util.spec_from_file_location("paw", Path(__file__).resolve().parents[2] / "scripts/read_paw.py")
paw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paw)
reader_spec = importlib.util.spec_from_file_location("paw_wannier", Path(__file__).resolve().parents[2] / "scripts/read_paw_wannier.py")
reader = importlib.util.module_from_spec(reader_spec)
reader_spec.loader.exec_module(reader)


def run(work, binary, label, commands, ranks=1, success=True):
    inp, out = work / f"{label}.in", work / f"{label}.out"
    inp.write_text(commands)
    command = [str(binary), "-i", str(inp), "-o", str(out)]
    if ranks > 1:
        command = [launcher[0], launcher[1], str(ranks)] + command
    proc = subprocess.run(command, cwd=work, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=180)
    log = out.read_text() if out.exists() else ""
    assert (proc.returncode == 0) == success, f"{label}: {log[-6000:]}\n{proc.stdout[-3000:]}"
    return log


def read_wannier(prefix):
    meta = json.loads(Path(str(prefix) + ".mlwfCAE.json").read_text())
    gs = np.loadtxt(str(prefix) + ".mlwfCAE.header", skiprows=2, dtype=int, ndmin=2)
    coeff = np.fromfile(str(prefix) + ".mlwfCAE", dtype="<c16").reshape(meta["orbitals"], len(gs))
    return meta, gs, coeff


def reference(prefix, bloch):
    meta, gs, actual = read_wannier(prefix)
    nk, nw, nb = len(meta["kmesh"]), meta["orbitals"], meta["source_bands"]
    u = np.fromfile(str(prefix) + ".mlwfU", dtype="<c16", count=nk*nw*nb).reshape(nk, nw, nb).swapaxes(1, 2)
    supercell = np.array(meta["supercell_matrix"])
    qsuper = np.array(meta["supercell_kpoint"])
    lookup = {tuple(g): i for i, g in enumerate(gs)}
    expected = np.zeros_like(actual)
    full = []
    for ik, entry in enumerate(meta["kmesh"]):
        q, invert = entry["reduced_state"], entry["invert"]
        g, c = bloch.wavefunction(q)
        kc = np.array(bloch.metadata["kpoints_reciprocal"][q])
        rot, a = np.array(entry["rotation"]), np.array(entry["translation"])
        phase = np.exp(invert * 2j*np.pi * ((g+kc) @ a))
        c = u[ik].T @ (c if invert > 0 else c.conj()) * phase
        physical = (g+kc) @ rot * invert
        gfull = np.rint(physical - entry["k"]).astype(int)
        full.append((gfull, c))
        gsuper = np.rint(physical @ supercell - qsuper).astype(int)
        indices = [lookup[tuple(v)] for v in gsuper]
        expected[:, indices] += entry["weight"] * c
    expected *= np.exp(-2j*np.pi * ((gs+qsuper) @ np.array(meta["translation_supercell"])))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-10)
    shape = tuple(meta["fft_grid"])
    fourier = np.zeros((nw, *shape), dtype=complex)
    index = gs % shape
    fourier[:, index[:, 0], index[:, 1], index[:, 2]] = actual
    real = np.fft.ifftn(fourier, axes=(-3, -2, -1)) * np.prod(shape)
    for axis in range(3):
        dims = [1, 1, 1, 1]
        dims[axis+1] = shape[axis]
        real *= np.exp(2j*np.pi*qsuper[axis]*np.arange(shape[axis])/shape[axis]).reshape(dims)
    saved = np.stack([np.fromfile(str(prefix) + f".{n}.mlwfAE", dtype="<c16").reshape(shape) for n in range(nw)])
    np.testing.assert_allclose(saved, real, rtol=0, atol=3e-11)
    exported = reader.PAWWannierExport(prefix)
    np.testing.assert_array_equal(exported.coefficients(0), actual[0])
    np.testing.assert_allclose(exported.real_space(0), saved[0], rtol=0, atol=3e-11)
    norm = meta["supercell_volume"] * np.sum(abs(actual)**2, axis=1)
    assert np.max(abs(norm-1)) < .03, norm
    print("AE Wannier norms:", norm, flush=True)
    return meta, u, full, actual


def check_operators(prefix, bloch, meta, rotations, full):
    nk, nw = len(full), meta["orbitals"]
    kmesh = np.array([k["k"] for k in meta["kmesh"]])
    lattice = np.array(meta["primitive_lattice_columns_bohr"]).T
    reciprocal = 2*np.pi*np.linalg.inv(lattice)
    volume = bloch.metadata["volume"]
    operators = json.loads(Path(str(prefix) + ".mlwfPAW.json").read_text())
    h = np.zeros((nk, 1, nw, nw), complex)
    p = np.zeros((nk, 3, nw, nw), complex)
    r = np.zeros_like(p)
    rp = np.zeros((nk, 3, 3, nw, nw), complex)
    w = np.zeros_like(h)
    for ik, entry in enumerate(meta["kmesh"]):
        u, q = rotations[ik], entry["reduced_state"]
        h[ik, 0] = u.conj().T @ (bloch.eigenvalues[q, :, None] * u)
        ip = 1j * bloch.momenta[q]
        if entry["invert"] < 0:
            ip = ip.conj()
        ip = np.array([u.conj().T @ operator @ u for operator in ip])
        cartrot = lattice @ entry["rotation"] @ np.linalg.inv(lattice)
        p[ik] = np.einsum("aij,ab->bij", ip, cartrot)
        gi, ci = full[ik]
        w[ik, 0] = volume * ci.conj() @ ci.T  # slabWeight of width one is constant one.
        physical = (gi+kmesh[ik]) @ reciprocal
        for edge in operators["derivative_stencil"]:
            kj = kmesh[ik] + np.array(edge["dk"])
            delta = kmesh-kj
            errors = np.max(abs(delta-np.rint(delta)), axis=1)
            jk = int(np.argmin(errors))
            assert errors[jk] < 1e-8
            shift = np.rint(kj-kmesh[jk]).astype(int)
            gj, cj = full[jk]
            lookup = {tuple(g-shift): index for index, g in enumerate(gj)}
            pairs = [(i, lookup[tuple(g)]) for i, g in enumerate(gi) if tuple(g) in lookup]
            ii, jj = np.array(pairs).T
            overlap = volume * cj[:, jj].conj() @ ci[:, ii].T
            for mu in range(3):
                weight = -.5j * edge["weight"] * edge["b_cartesian"][mu]
                r[ik, mu] += weight * overlap
                for nu in range(3):
                    derivative = volume * cj[:, jj].conj() @ (ci[:, ii] * (1j*physical[ii, nu])).T
                    rp[ik, mu, nu] += weight * derivative
    # Binary matrices use cell, operator, ket, bra; the cell order is the native k-folding box.
    cells = np.array(list(np.ndindex(2, 2, 2)))
    phase = np.exp(-2j*np.pi * (cells @ kmesh.T)) / nk
    errors = {}
    for name, values in [("H", h), ("P", p), ("R", r), ("RP", rp), ("W", w)]:
        expected = np.einsum("ck,k...->c...", phase, values).real
        actual = np.fromfile(str(prefix) + f".mlwf{name}", dtype="<f8").reshape(expected.shape).swapaxes(-1, -2)
        errors[name] = float(np.max(abs(actual-expected)))
        # R has analytic PAW augmentation; this Fourier reference has finite-cutoff norm loss.
        tolerance = .02 if name == "R" else 2e-8
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance, err_msg=name)
    print("Independent AE Wannier operator errors:", errors, flush=True)
    return errors


with tempfile.TemporaryDirectory(prefix="jdftx-paw-wannier-") as directory:
    work = Path(directory)
    centers = """wannier-center Gaussian .125 .125 .125 1
wannier-center Gaussian -.375 .125 .125 1
wannier-center Gaussian .125 -.375 .125 1
wannier-center Gaussian .125 .125 -.375 1
"""
    common = f"""ion-species {dataset}
lattice Face-Centered Cubic 10.26
ion Si 0 0 0 0
ion Si .25 .25 .25 0
elec-cutoff 8
elec-n-bands 4
kpoint-folding 2 2 2
"""
    def scf(label, base):
        run(work, exe, label, base + f"""electronic-scf nIterations 40 energyDiffThreshold 1e-10 eigDiffThreshold 1e-8
 dump-name {work}/{label}.$VAR
 dump End State BandEigs AEWfns AEOverlap AEMomenta AEBandEigs
 paw-reconstruction-cutoff 30
 """)
        return paw.PAWExport(work / f"{label}.paw")

    bloch = scf("scf", common)
    prefix = work / "w"
    base = common + f"wannier-initial-state {work}/scf.$VAR\nwannier-dump-name {prefix}.$VAR\n" + centers
    options = """wannier saveAEWfns yes saveAEWfnsRealSpace yes saveMomenta yes saveR yes saveRP yes slabWeight .5 .5 .1
paw-reconstruction-cutoff 30
"""
    run(work, wannier, "wannier", base + options + "wannier-minimize nIterations 10\n")
    meta, rotations, full, coeff = reference(prefix, bloch)
    assert len(meta["kmesh"]) > bloch.states
    assert any(np.any(entry["translation"]) for entry in meta["kmesh"]), "Nonsymmorphic coverage"
    low_errors = check_operators(prefix, bloch, meta, rotations, full)
    saved = {suffix: np.fromfile(str(prefix)+suffix) for suffix in (".mlwfH", ".mlwfP", ".mlwfR", ".mlwfRP", ".mlwfW")}
    if launcher:
        for ranks in (2, 4):
            run(work, wannier, f"mpi{ranks}", base + options.replace("wannier saveAEWfns", "wannier loadRotations yes saveAEWfns") + "wannier-minimize nIterations 0\n", ranks)
            _, _, actual = read_wannier(prefix)
            np.testing.assert_allclose(actual, coeff, rtol=0, atol=3e-10)
            for suffix, expected in saved.items():
                np.testing.assert_allclose(np.fromfile(str(prefix)+suffix), expected, rtol=0, atol=2e-9)
    # Reuse the fixed Wannier gauge to test each wavefunction selector independently.
    for option, reciprocal_output in [("saveAEWfns", True), ("saveAEWfnsRealSpace", False)]:
        output = work / option
        for suffix in (".mlwfU", ".mlwfState", ".mlwfM0"):
            shutil.copyfile(str(prefix)+suffix, str(output)+suffix)
        independent = base.replace(str(prefix)+".$VAR", str(output)+".$VAR")
        run(work, wannier, option, independent + f"wannier loadRotations yes {option} yes\npaw-reconstruction-cutoff 30\nwannier-minimize nIterations 0\n")
        assert Path(str(output)+".mlwfCAE").exists() == reciprocal_output
        assert Path(str(output)+".0.mlwfAE").exists() != reciprocal_output
        assert reader.PAWWannierExport(output).real_space(0).shape == tuple(meta["fft_grid"])
    tdc = work / "tdc"
    for suffix in (".mlwfU", ".mlwfState", ".mlwfM0"):
        shutil.copyfile(str(prefix)+suffix, str(tdc)+suffix)
    tdc_base = base.replace(str(prefix)+".$VAR", str(tdc)+".$VAR")
    run(work, wannier, "tdc", tdc_base + "wannier loadRotations yes localizationMeasure TruncatedDC saveAEWfns yes\npaw-reconstruction-cutoff 30\nwannier-minimize nIterations 0\n")
    np.testing.assert_allclose(np.fromfile(str(tdc)+".mlwfCAE", dtype="<c16").reshape(coeff.shape), coeff, rtol=0, atol=3e-10)
    # Analytic H/P/R stay in the same gauge; finite-cutoff AE norms approach them.
    high_prefix = work / "high-w"
    for suffix in (".mlwfU", ".mlwfState", ".mlwfM0"):
        shutil.copyfile(str(prefix)+suffix, str(high_prefix)+suffix)
    run(work, exe, "high-bloch", common + f"initial-state {work}/scf.$VAR\ndump-only\ndump-name {work}/high.$VAR\ndump End AEWfns AEOverlap AEMomenta AEBandEigs\npaw-reconstruction-cutoff 80\n")
    high_base = base.replace(str(prefix)+".$VAR", str(high_prefix)+".$VAR")
    run(work, wannier, "high-wannier", high_base + options.replace("wannier saveAEWfns", "wannier loadRotations yes saveAEWfns").replace("cutoff 30", "cutoff 80") + "wannier-minimize nIterations 0\n")
    high_bloch = paw.PAWExport(work / "high.paw")
    high_meta, high_u, high_full, _ = reference(high_prefix, high_bloch)
    high_errors = check_operators(high_prefix, high_bloch, high_meta, high_u, high_full)
    assert high_errors["R"] < low_errors["R"], (low_errors, high_errors)
    for suffix in (".mlwfH", ".mlwfP", ".mlwfR"):
        np.testing.assert_allclose(np.fromfile(str(high_prefix)+suffix), saved[suffix], rtol=0, atol=2e-9)
    run(work, wannier, "bad-cutoff", base + "wannier saveAEWfns yes\npaw-reconstruction-cutoff 4\n", success=False)
    # Half-shifted mesh with only time reversal exercises conjugation and supercell Bloch phase.
    shifted_common = common + "symmetries manual\nsymmetry-matrix 1 0 0 0 1 0 0 0 1 0 0 0\nkpoint-reduce-inversion yes\nkpoint .5 .5 .5 1\n"
    shifted_bloch = scf("shifted", shifted_common)
    shifted = work / "shifted-w"
    run(work, wannier, "shifted-wannier", shifted_common + f"wannier-initial-state {work}/shifted.$VAR\nwannier-dump-name {shifted}.$VAR\n" + centers +
        "wannier saveAEWfns yes saveAEWfnsRealSpace yes\npaw-reconstruction-cutoff 30\nwannier-minimize nIterations 10\n")
    shifted_meta, _, _, _ = reference(shifted, shifted_bloch)
    assert any(k["invert"] < 0 for k in shifted_meta["kmesh"])
    assert np.linalg.norm(shifted_meta["supercell_kpoint"]) > .1
    print("PAW AE Wannier wavefunction/operator, symmetry, shifted-mesh and MPI checks passed", flush=True)
