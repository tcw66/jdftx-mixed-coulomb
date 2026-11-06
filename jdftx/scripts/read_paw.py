#!/usr/bin/env python3
"""Read JDFTx PAW exports and optionally reconstruct periodic/Bloch orbitals.

Usage: read_paw.py results/Si.paw [--state 0 --real-space state.npz] [--bloch]
Reads single-file JDFTx-PAW-2 and older q-indexed JDFTx-PAW-1 exports.
Coefficients are not renormalized. Requires NumPy.
"""
import argparse
import json
from pathlib import Path
import numpy as np


class PAWExport:
    def __init__(self, prefix):
        self.prefix = str(prefix)
        self.metadata = json.loads(Path(self.prefix + ".json").read_text())
        version = self.metadata["format"]
        if version not in ("JDFTx-PAW-1", "JDFTx-PAW-2"):
            raise ValueError("Unsupported PAW export version")
        self.legacy = version == "JDFTx-PAW-1"
        order = "<" if self.metadata.get("byte_order", "little") == "little" else ">"
        self.complex_dtype = np.dtype(order + "c16")
        self.real_dtype = np.dtype(order + "f8")
        self.states, self.bands = self.metadata["states"], self.metadata["bands"]
        self._gvectors = None
        self.outputs = self.metadata.get("outputs", {
            "wavefunctions": Path(self.prefix + ".q0.wfns").exists(),
            "real_space_wavefunctions": False,
            "eigenvalues": True,
            "overlap": Path(self.prefix + ".matrices").exists(),
            "momenta": Path(self.prefix + ".matrices").exists(),
        })
        self.eigenvalues = (np.fromfile(self.prefix + ".eigenvals", dtype=self.real_dtype)
                            .reshape(self.states, self.bands) if self.outputs["eigenvalues"] else None)
        if self.legacy:
            path = Path(self.prefix + ".matrices")
            matrices = (np.fromfile(path, dtype=self.complex_dtype)
                        .reshape(self.states, 5, self.bands, self.bands).swapaxes(-1, -2)
                        if path.exists() else None)
            self.overlap = matrices[:, 0] if matrices is not None else None
            self.momenta = matrices[:, 2:] if matrices is not None else None
        else:
            self.overlap = (np.fromfile(self.prefix + ".overlap", dtype=self.complex_dtype)
                            .reshape(self.states, self.bands, self.bands).swapaxes(-1, -2)
                            if self.outputs["overlap"] else None)
            self.momenta = (np.fromfile(self.prefix + ".momenta", dtype=self.complex_dtype)
                            .reshape(self.states, 3, self.bands, self.bands).swapaxes(-1, -2)
                            if self.outputs["momenta"] else None)

    @property
    def matrices(self):
        """Compatibility view [S, diagonal H, px, py, pz]; H is not stored on disk."""
        if any(x is None for x in (self.overlap, self.momenta, self.eigenvalues)):
            return None
        hamiltonian = self.eigenvalues[:, :, None] * np.eye(self.bands)
        return np.concatenate((self.overlap[:, None], hamiltonian[:, None], self.momenta), axis=1)

    def _check_state(self, state):
        if not 0 <= state < self.states:
            raise IndexError("State index outside exported k mesh")

    def wavefunction(self, state):
        self._check_state(state)
        if not self.outputs["wavefunctions"]:
            raise ValueError("Reciprocal-space wavefunctions were not requested (dump End AEWfns)")
        if self.legacy:
            gs = np.loadtxt(self.prefix + f".q{state}.Gvectors", dtype=int, ndmin=2)
            coefficients = np.fromfile(self.prefix + f".q{state}.wfns", dtype=self.complex_dtype)
            return gs, coefficients.reshape(self.bands, len(gs))
        sizes = np.asarray(self.metadata["basis_sizes"], dtype=np.int64)
        expected_offsets = np.r_[0, np.cumsum(sizes[:-1])] * self.bands * self.complex_dtype.itemsize
        if not np.array_equal(expected_offsets, self.metadata["wavefunction_offsets_bytes"]):
            raise ValueError("Invalid ColumnBundle offsets in PAW metadata")
        path = Path(self.prefix + ".wfns")
        if path.stat().st_size != int(np.sum(sizes)) * self.bands * self.complex_dtype.itemsize:
            raise ValueError("PAW wavefunction file has the wrong length")
        if self._gvectors is None:
            self._gvectors = np.loadtxt(self.prefix + ".Gvectors", dtype=int, ndmin=2)
            if self._gvectors.shape != (int(np.sum(sizes)), 3):
                raise ValueError("PAW G-vector file has the wrong length")
        start = int(np.sum(sizes[:state]))
        gs = self._gvectors[start:start + sizes[state]]
        coefficients = np.memmap(path, dtype=self.complex_dtype, mode="r",
                                 offset=int(expected_offsets[state]), shape=(self.bands, sizes[state]))
        return gs, coefficients

    def _bloch_phase(self, orbitals, state):
        shape = orbitals.shape[1:]
        k = self.metadata["kpoints_reciprocal"][state]
        for axis in range(3):
            dimensions = [1, 1, 1, 1]
            dimensions[axis + 1] = shape[axis]
            phase = np.exp(2j * np.pi * k[axis] * np.arange(shape[axis]) / shape[axis])
            orbitals *= phase.reshape(dimensions)
        return orbitals

    def real_space_dump(self, state, bloch=False):
        """Read AERealSpaceWfns files; axes are band, x, y, z (z fastest)."""
        self._check_state(state)
        if not self.outputs["real_space_wavefunctions"]:
            raise ValueError("Real-space wavefunctions were not requested (dump End AERealSpaceWfns)")
        shape = tuple(self.metadata["fft_grid"])
        orbitals = np.stack([np.fromfile(self.prefix + f".q{state}.b{band}.rs", dtype=self.complex_dtype)
                             .reshape(shape) for band in range(self.bands)])
        return self._bloch_phase(orbitals, state) if bloch else orbitals

    def real_space(self, state, shape=None, bloch=False):
        if not self.outputs["wavefunctions"]:
            if shape is not None and tuple(shape) != tuple(self.metadata["fft_grid"]):
                raise ValueError("Real-space-only exports require the stored FFT grid")
            return self.real_space_dump(state, bloch)
        gs, coefficients = self.wavefunction(state)
        minimum = 2 * np.max(abs(gs), axis=0) + 1
        default = self.metadata.get("fft_grid") or minimum
        shape = np.asarray(default if shape is None else shape, dtype=int)
        if shape.shape != (3,) or np.any(shape < minimum):
            raise ValueError(f"Grid must have at least {tuple(minimum)} points to avoid aliasing")
        reciprocal = np.zeros((self.bands, *shape), dtype=complex)
        indices = gs % shape
        reciprocal[:, indices[:, 0], indices[:, 1], indices[:, 2]] = coefficients
        u = np.fft.ifftn(reciprocal, axes=(-3, -2, -1)) * np.prod(shape)
        return self._bloch_phase(u, state) if bloch else u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix")
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--real-space", type=Path, help="Write this state's real-space orbitals to NPZ")
    parser.add_argument("--bloch", action="store_true", help="Include exp(i k.r) in real-space orbitals")
    args = parser.parse_args()
    data = PAWExport(args.prefix)
    print(f"{data.states} states, {data.bands} bands; eigenvalues in Hartree when requested")
    if data.overlap is not None:
        print("Maximum PAW overlap error:", np.max(abs(data.overlap - np.eye(data.bands))))
    if data.outputs["wavefunctions"]:
        for state in range(data.states):
            _, c = data.wavefunction(state)
            print(f"state {state} reconstructed norms:", data.metadata["volume"] * np.sum(abs(c)**2, axis=1))
    if args.real_space:
        values = dict(u=data.real_space(args.state, bloch=args.bloch),
                      k=data.metadata["kpoints_reciprocal"][args.state],
                      volume=data.metadata["volume"], bloch=args.bloch)
        if data.outputs["wavefunctions"]:
            values["G"], values["coefficients"] = data.wavefunction(args.state)
        np.savez_compressed(args.real_space, **values)


if __name__ == "__main__":
    main()
