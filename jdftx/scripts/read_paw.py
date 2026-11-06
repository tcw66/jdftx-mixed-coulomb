#!/usr/bin/env python3
"""Read JDFTx PAW exports and optionally reconstruct periodic/Bloch orbitals.

Usage: read_paw.py results/Si.paw [--state 0 --real-space state.npz] [--bloch]
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
        if self.metadata["format"] != "JDFTx-PAW-1":
            raise ValueError("Unsupported PAW export version")
        order = "<" if self.metadata.get("byte_order", "little") == "little" else ">"
        self.complex_dtype = np.dtype(order + "c16")
        self.real_dtype = np.dtype(order + "f8")
        self.states, self.bands = self.metadata["states"], self.metadata["bands"]
        self.eigenvalues = np.fromfile(self.prefix + ".eigenvals", dtype=self.real_dtype).reshape(self.states, self.bands)
        path = Path(self.prefix + ".matrices")
        self.matrices = (np.fromfile(path, dtype=self.complex_dtype)
                         .reshape(self.states, 5, self.bands, self.bands).swapaxes(-1, -2)
                         if path.exists() else None)

    def wavefunction(self, state):
        if not 0 <= state < self.states:
            raise IndexError("State index outside exported k mesh")
        gs = np.loadtxt(self.prefix + f".q{state}.Gvectors", dtype=int, ndmin=2)
        coefficients = np.fromfile(self.prefix + f".q{state}.wfns", dtype=self.complex_dtype)
        return gs, coefficients.reshape(self.bands, len(gs))

    def real_space(self, state, shape=None, bloch=False):
        gs, coefficients = self.wavefunction(state)
        minimum = 2 * np.max(abs(gs), axis=0) + 1
        shape = np.asarray(minimum if shape is None else shape, dtype=int)
        if shape.shape != (3,) or np.any(shape < minimum):
            raise ValueError(f"Grid must have at least {tuple(minimum)} points to avoid aliasing")
        reciprocal = np.zeros((self.bands, *shape), dtype=complex)
        indices = gs % shape
        reciprocal[:, indices[:, 0], indices[:, 1], indices[:, 2]] = coefficients
        u = np.fft.ifftn(reciprocal, axes=(-3, -2, -1)) * np.prod(shape)
        if bloch:
            k = self.metadata["kpoints_reciprocal"][state]
            for axis in range(3):
                dimensions = [1, 1, 1, 1]
                dimensions[axis + 1] = shape[axis]
                phase = np.exp(2j * np.pi * k[axis] * np.arange(shape[axis]) / shape[axis])
                u *= phase.reshape(dimensions)
        return u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix")
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--real-space", type=Path, help="Write this state's real-space orbitals to NPZ")
    parser.add_argument("--bloch", action="store_true", help="Include exp(i k.r) in real-space orbitals")
    args = parser.parse_args()
    data = PAWExport(args.prefix)
    print(f"{data.states} states, {data.bands} bands; eigenvalues in Hartree")
    if data.matrices is not None:
        print("Maximum PAW overlap error:", np.max(abs(data.matrices[:, 0] - np.eye(data.bands))))
    for state in range(data.states):
        path = Path(data.prefix + f".q{state}.wfns")
        if path.exists():
            _, c = data.wavefunction(state)
            print(f"state {state} reconstructed norms:", data.metadata["volume"] * np.sum(abs(c)**2, axis=1))
    if args.real_space:
        gs, c = data.wavefunction(args.state)
        np.savez_compressed(args.real_space, u=data.real_space(args.state, bloch=args.bloch),
                            G=gs, coefficients=c, k=data.metadata["kpoints_reciprocal"][args.state],
                            volume=data.metadata["volume"], bloch=args.bloch)


if __name__ == "__main__":
    main()
