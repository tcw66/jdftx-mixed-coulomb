#!/usr/bin/env python3
"""Read AE Wannier exports with wannier-dump-name <prefix>.$VAR.

Usage: read_paw_wannier.py results/Si --orbital 0 --real-space w0.npz
Binary data are little-endian complex128. No normalization or phase fixing.
"""
import argparse
import json
from pathlib import Path
import numpy as np


class PAWWannierExport:
    def __init__(self, prefix):
        self.prefix = str(prefix)
        self.metadata = json.loads(Path(self.prefix + ".mlwfCAE.json").read_text())
        if self.metadata["format"] != "JDFTx-PAW-Wannier-1":
            raise ValueError("Unsupported PAW Wannier format")
        self.orbitals = self.metadata["orbitals"]
        self.basis_size = self.metadata["basis_size"]
        self.gvectors = np.loadtxt(self.prefix + ".mlwfCAE.header", skiprows=2, dtype=int, ndmin=2)
        if self.gvectors.shape != (self.basis_size, 3):
            raise ValueError("AE Wannier basis header has the wrong length")

    def coefficients(self, orbital):
        if not 0 <= orbital < self.orbitals:
            raise IndexError("Wannier orbital outside exported range")
        if not self.metadata["reciprocal_wavefunctions"]:
            raise ValueError("Reciprocal orbitals require wannier saveAEWfns yes")
        path = Path(self.prefix + ".mlwfCAE")
        if path.stat().st_size != self.orbitals * self.basis_size * 16:
            raise ValueError("AE Wannier coefficient file has the wrong length")
        return np.memmap(path, dtype="<c16", mode="r", offset=orbital*self.basis_size*16,
                         shape=(self.basis_size,))

    def real_space(self, orbital):
        if not 0 <= orbital < self.orbitals:
            raise IndexError("Wannier orbital outside exported range")
        shape = tuple(self.metadata["fft_grid"])
        if not self.metadata["reciprocal_wavefunctions"]:
            if not self.metadata["real_space_wavefunctions"]:
                raise ValueError("No wavefunctions selected")
            return np.fromfile(self.prefix + f".{orbital}.mlwfAE", dtype="<c16").reshape(shape)
        fourier = np.zeros(shape, dtype=complex)
        indices = self.gvectors % shape
        fourier[indices[:, 0], indices[:, 1], indices[:, 2]] = self.coefficients(orbital)
        psi = np.fft.ifftn(fourier) * np.prod(shape)
        for axis, k in enumerate(self.metadata["supercell_kpoint"]):
            dimensions = [1, 1, 1]
            dimensions[axis] = shape[axis]
            psi *= np.exp(2j*np.pi*k*np.arange(shape[axis])/shape[axis]).reshape(dimensions)
        return psi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="Prefix before .$VAR in wannier-dump-name")
    parser.add_argument("--orbital", type=int, default=0)
    parser.add_argument("--real-space", type=Path, help="Write the selected supercell orbital to NPZ")
    args = parser.parse_args()
    data = PAWWannierExport(args.prefix)
    print(f"{data.orbitals} AE Wannier orbitals, cutoff {data.metadata['reconstruction_cutoff']} Ha")
    if data.metadata["reciprocal_wavefunctions"]:
        norm = data.metadata["supercell_volume"] * np.sum(abs(data.coefficients(args.orbital))**2)
        print(f"orbital {args.orbital}: finite-cutoff norm {norm:.12g}")
    if args.real_space:
        np.savez_compressed(args.real_space, psi=data.real_space(args.orbital),
                            lattice_columns_bohr=data.metadata["lattice_columns_bohr"],
                            supercell_kpoint=data.metadata["supercell_kpoint"],
                            orbital=args.orbital, reconstruction_cutoff=data.metadata["reconstruction_cutoff"])


if __name__ == "__main__":
    main()
