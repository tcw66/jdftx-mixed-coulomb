#!/usr/bin/env python3
"""Independent supercell finite-difference validation of native KI(2).
Usage: checkKI.py BUILD_DIR H_NORM_CONSERVING_UPF [MPIEXEC] [--spin] [--gga]
Checks LDA/PBE kernel/curvature, screened curvature against independently relaxed
SCF densities, valence/conduction shifts, q wrapping, Ewald scaling, and MPI.
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import numpy as np

build = Path(sys.argv[1]).resolve()
pseudo = Path(sys.argv[2]).resolve()
spin = '--spin' in sys.argv[3:]
gga = '--gga' in sys.argv[3:]
launcher = next((arg for arg in sys.argv[3:] if arg not in ('--spin', '--gga')), None)
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def vxc(n):
    """Independent analytic unpolarized Slater exchange + PZ correlation."""
    n = np.maximum(n, 1e-20)
    r = (3/(4*np.pi*n))**(1/3)
    den = 1+1.0529*np.sqrt(r)+0.3334*r
    ec = np.where(r < 1, (0.0311+0.002*r)*np.log(r)-0.048-0.0116*r, -0.1423/den)
    er = np.where(r < 1, 0.0311/r+0.002*(1+np.log(r))-0.0116,
                  0.1423*(1.0529/(2*np.sqrt(r))+0.3334)/den**2)
    return -(3*n/np.pi)**(1/3)+ec-r*er/3


def kernel(n, rho, lengths):
    g = [2*np.pi*np.fft.fftfreq(s, d=l/s) for s, l in zip(n.shape, lengths)]
    g2 = sum(x*x for x in np.meshgrid(*g, indexing='ij'))
    coulomb = np.zeros_like(g2)
    np.divide(4*np.pi, g2, out=coulomb, where=g2 > 0)
    hartree = np.fft.ifftn(coulomb*np.fft.fftn(rho)).real
    h = 1e-4
    fxc = (vxc(n*(1+h))-vxc(n*(1-h)))/(2*h*np.maximum(n, 1e-20))
    return hartree+fxc*rho


def magnetic_kernel(n, rho):
    """Independent Slater/PZ longitudinal spin Hessian at zero magnetization."""
    n = np.maximum(n, 1e-20)
    r = (3/(4*np.pi*n))**(1/3)
    para = np.where(r < 1, (0.0311+0.002*r)*np.log(r)-0.048-0.0116*r,
                    -0.1423/(1+1.0529*np.sqrt(r)+0.3334*r))
    ferro = np.where(r < 1, (0.01555+0.0007*r)*np.log(r)-0.0269-0.0048*r,
                     -0.0843/(1+1.3981*np.sqrt(r)+0.2611*r))
    fmm = -(3/np.pi)**(1/3)/(3*n**(2/3)) + 4*(ferro-para)/(9*(2**(1/3)-1)*n)
    return fmm*rho


def read_ki(prefix, variable):
    dtype, cells, centers = Path(str(prefix)+'.koopmansFormat').read_text().split()
    return np.fromfile(str(prefix)+'.'+variable, dtype=np.float64 if dtype == 'real64' else np.complex128)


with tempfile.TemporaryDirectory(prefix='jdftx-ki-') as name:
    root = Path(name)
    common = f'''coords-type cartesian
ion-species {pseudo}
symmetries none
kpoint-reduce-inversion no
core-overlap-check none
elec-cutoff 6
elec-ex-corr {'gga-PBE' if gga else 'lda-PZ'}
converge-empty-states yes
electronic-scf nIterations 200 nEigSteps 40 energyDiffThreshold 1e-13 residualThreshold 1e-10
electronic-minimize nIterations 300 energyDiffThreshold 1e-13
'''

    def run(folder, executable, text, ranks=1):
        folder.mkdir(exist_ok=True)
        (folder/'input.in').write_text(text)
        (folder/'output.out').unlink(missing_ok=True)
        command = [str(build/executable), '-c', '1', '-i', 'input.in', '-o', 'output.out']
        if ranks > 1:
            command = [launcher, '-n', str(ranks)]+command
        subprocess.run(command, cwd=folder, env=env, check=True, stdout=subprocess.DEVNULL)
        output = (folder/'output.out').read_text()
        assert 'Done!' in output and 'Stack trace:' not in output, output[-2500:]
        return output

    for fold in (1, 3):
        primitive = root/f'primitive-{fold}'
        geometry = ('lattice Cubic 8\nfftbox 24 24 24\nelec-n-bands 2\n'
                    f'kpoint-folding 1 1 {fold}\nion H 4 4 3.3 1\nion H 4 4 4.7 1\n')
        run(primitive, 'jdftx', common+geometry+'dump-name scf.$VAR\ndump End State BandEigs ElecDensity\n')
        for empty in (False, True):
            label = 'empty' if empty else 'valence'
            prefix = primitive/label
            ki_input = (common+geometry+f'wannier-initial-state {primitive}/scf.$VAR\n'
                        f'wannier-dump-name {prefix}.$VAR\nwannier-center Gaussian 4 4 3.3\n'
                        f'wannier bStart {int(empty)} saveWfnsRealSpace yes\n'
                        'wannier-minimize nIterations 100 knormThreshold 1e-8\nwannier-koopmans 0 400 1e-9\n'
                        + ('' if spin else 'wannier-koopmans-spin no\n'))
            log = run(primitive, 'wannier', ki_input)
            spectrum = np.loadtxt(str(prefix)+'.koopmansBands', ndmin=2)
            inverse_phases = np.exp(2j*np.pi*spectrum[:, 2, None]*np.arange(fold))
            h_from_cells = inverse_phases@read_ki(prefix, 'mlwfHki')
            np.testing.assert_allclose(h_from_cells.real, spectrum[:, 3], atol=1e-10)
            np.testing.assert_allclose(h_from_cells.imag, 0., atol=1e-10)
            screen = np.loadtxt(str(prefix)+'.koopmansScreening', ndmin=2)[0]
            assert screen[1] == int(not empty)
            assert np.isfinite(screen[4])
            # Wannier real-space output is centered by a half-supercell translation.
            shape = (24, 24, 24*fold)
            grid_log = log.split('Initializing supercell grid for Wannier functions')[1]
            wfshape = tuple(map(int, re.search(r'Chosen fftbox size, S = \[\s*(\d+)\s+(\d+)\s+(\d+)', grid_log).groups()))
            psi = np.fromfile(str(prefix)+'.0.mlwf', dtype=np.float64).reshape(wfshape)
            psi = np.roll(psi, tuple(s//2 for s in wfshape), axis=(0, 1, 2))
            # Embed plane-wave coefficients in the SCF density FFT grid.
            coeff = np.fft.fftn(psi)/psi.size
            enlarged = np.zeros(shape, dtype=complex)
            indices = [np.rint(np.fft.fftfreq(s)*s).astype(int)%t for s, t in zip(wfshape, shape)]
            enlarged[np.ix_(*indices)] = coeff
            psi = (np.fft.ifftn(enlarged)*np.prod(shape)).real
            dv = (8/24)**3
            rho = psi**2
            np.testing.assert_allclose(rho.sum()*dv, 1., atol=1e-8)
            # Build the independent Gamma supercell with the same finite sampling.
            supercell = root/f'super-{fold}-{label}'
            ions = ''.join(f'ion H 4 4 {z+8*r} 1\n' for r in range(fold) for z in (3.3, 4.7))
            super_geometry = (f'lattice Orthorhombic 8 8 {8*fold}\nfftbox 24 24 {24*fold}\n'
                              f'elec-n-bands {fold+1}\n'+ions)
            spin_input = 'spintype z-spin\n' if spin else ''
            run(supercell, 'jdftx', common+super_geometry+spin_input+'dump-name scf.$VAR\ndump End State ElecDensity\n')
            if spin:
                density = sum(np.fromfile(supercell/f'scf.n_{s}', dtype=np.float64).reshape(shape) for s in ('up', 'dn'))
            else:
                density = np.fromfile(supercell/'scf.n', dtype=np.float64).reshape(shape)
            if gga:
                # Independent finite differences of the full native Hxc
                # potential, without using the DFPT Hessian implementation.
                fixed = re.sub(r'^electronic-scf .*\n', '', common, flags=re.M)
                fixed = re.sub(r'^electronic-minimize .*\n',
                               'electronic-minimize nIterations 0\n', fixed, flags=re.M)
                potentials = []
                step = 1e-6
                for sign in (-1, 1):
                    folder = root/f'kernel-{fold}-{label}-{sign}'
                    folder.mkdir()
                    if spin:
                        (.5*density+sign*step*rho).astype(np.float64).tofile(folder/'fixed.n_up')
                        (.5*density).astype(np.float64).tofile(folder/'fixed.n_dn')
                    else:
                        (density+sign*step*rho).astype(np.float64).tofile(folder/'fixed.n')
                    run(folder, 'jdftx', fixed+super_geometry+spin_input
                        +'fix-electron-density fixed.$VAR\ndump-name kernel.$VAR\ndump End Vscloc\n')
                    names = ('Vscloc_up', 'Vscloc_dn') if spin else ('Vscloc',)
                    # Vscloc stores the grid-weighted potential JdagOJ V.
                    potentials.append([np.fromfile(folder/('kernel.'+s), dtype=np.float64).reshape(shape)/dv
                                       for s in names])
                up = (potentials[1][0]-potentials[0][0])/(2*step)
                down = (potentials[1][1]-potentials[0][1])/(2*step) if spin else up
                charge, magnetic = .5*(up+down), .5*(up-down)
            else:
                charge = kernel(density, rho, (8, 8, 8*fold))
                magnetic = magnetic_kernel(density, rho) if spin else np.zeros_like(charge)
            potential = charge+magnetic
            bare = dv*np.sum(rho*potential)
            np.testing.assert_allclose(screen[2], bare, atol=2e-6, rtol=2e-5)
            responses = []
            h = 1e-3
            for sign in (-1, 1):
                folder = root/f'fd-{fold}-{label}-{sign}'
                folder.mkdir()
                (sign*h*potential).astype(np.float64).tofile(folder/'external')
                if spin:
                    (sign*h*(charge-magnetic)).astype(np.float64).tofile(folder/'external-dn')
                run(folder, 'jdftx', common+super_geometry+spin_input+f'initial-state {supercell}/scf.$VAR\n'
                    + ('Vexternal external external-dn\n' if spin else 'Vexternal external\n')
                    + 'dump-name fd.$VAR\ndump End ElecDensity\n')
                if spin:
                    up = np.fromfile(folder/'fd.n_up', dtype=np.float64).reshape(shape)
                    down = np.fromfile(folder/'fd.n_dn', dtype=np.float64).reshape(shape)
                    responses.append(dv*np.sum((charge+magnetic)*up+(charge-magnetic)*down))
                else:
                    responses.append(dv*np.sum(potential*np.fromfile(folder/'fd.n', dtype=np.float64).reshape(shape)))
            relaxation = (responses[1]-responses[0])/(2*h)
            np.testing.assert_allclose(screen[3], bare+relaxation, atol=2e-6, rtol=2e-4)
            # Occupied correction is exactly the screened scalar shift.
            if not empty:
                delta = read_ki(prefix, 'mlwfKI')
                np.testing.assert_allclose(delta[0], -0.5*screen[3], atol=1e-10)
                np.testing.assert_allclose(delta[1:], 0., atol=1e-10)
            else:
                # Independently contract the conduction potential with translated
                # supercell Wannier functions, then Fourier transform Eq. 22.
                real_elements = np.array([dv*np.sum(np.roll(psi, 24*r, axis=2)*potential*psi)
                                          for r in range(fold)])
                real_elements[0] -= 0.5*bare
                qs = np.loadtxt(str(prefix)+'.koopmansBands', ndmin=2)[:, 2]
                phases = np.exp(2j*np.pi*qs[:, None]*np.arange(fold))
                measured = phases@read_ki(prefix, 'mlwfKI')
                expected = screen[4]*(phases@real_elements)
                np.testing.assert_allclose(measured, expected, atol=3e-6, rtol=2e-4)
            print(f'{label}, mesh 1x1x{fold}: bare {bare:.9g}, relaxed FD {bare+relaxation:.9g}, alpha {screen[4]:.9g}')
            if launcher and fold == 3:
                serial_bands = np.loadtxt(str(prefix)+'.koopmansBands')
                run(primitive, 'wannier', ki_input, ranks=2)
                np.testing.assert_allclose(np.loadtxt(str(prefix)+'.koopmansScreening')[2:], screen[2:], atol=1e-10)
                np.testing.assert_allclose(np.loadtxt(str(prefix)+'.koopmansBands'), serial_bands, atol=1e-10)
            # For the same orbitals the point-charge Ewald terms add D and D/eps.
            run(primitive, 'wannier', ki_input.replace('wannier-koopmans 0', 'wannier-koopmans 2'))
            corrected = np.loadtxt(str(prefix)+'.koopmansScreening')
            np.testing.assert_allclose(corrected[2]-screen[2], corrected[5], atol=1e-10)
            np.testing.assert_allclose(corrected[3]-screen[3], corrected[5]/2, atol=1e-10)
    def reject(text, message, dry=False):
        folder = root/'invalid'
        folder.mkdir(exist_ok=True)
        (folder/'input.in').write_text(text)
        (folder/'output.out').unlink(missing_ok=True)
        command = [str(build/'wannier'), '-c', '1', '-i', 'input.in', '-o', 'output.out']
        if dry:
            command.append('-n')
        result = subprocess.run(command, cwd=folder, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert result.returncode != 0
        assert message in (folder/'output.out').read_text()

    reject(ki_input.replace('wannier-koopmans 0', 'wannier-koopmans 0.5'), 'epsilonInfinity must be')
    mixed = (ki_input.replace('wannier bStart 1', 'wannier bStart 0')
             .replace('wannier-center Gaussian 4 4 3.3',
                      'wannier-center Gaussian 4 4 3.3\nwannier-center Gaussian 4 4 4.7'))
    reject(mixed, 'KI Wannier rotations mix occupied and empty')
    print('KI independent kernel, supercell SCF response, orbital shifts, wrapping, Ewald and input tests passed.')
