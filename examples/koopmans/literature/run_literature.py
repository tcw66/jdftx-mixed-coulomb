#!/usr/bin/env python3
"""Prepare/run isolated silicon KI(2) calculations with literature XC choices."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT.parent
REPO = EXAMPLE.parents[1]
EV = 27.211386245988
PSEUDOS = {
    'lda-pw': ('pw', '686dd9f7d58fe63bdb1e595f0aeecf7d70d2857f06ffb00b8273950d6431e805'),
    'pbe': ('pbe', '39822757f53f36e3bf3bfb779356152a8d3f21199c7db9dd5a931e5d18c45282'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pseudo(functional, override):
    flavor, expected = PSEUDOS[functional]
    url = f'https://www.pseudo-dojo.org/pseudos/nc-sr-04_{flavor}_standard/Si.upf.gz'
    target = override.resolve() if override else ROOT/'pseudopotentials'/f'Si-{flavor}.upf'
    if not target.exists():
        if override:
            raise ValueError(f'Missing pseudopotential: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'JDFTx-literature-inputs/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = gzip.decompress(response.read())
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Downloaded pseudopotential changed; review its version before updating the pinned checksum')
        target.write_bytes(data)
    text = target.read_text()
    match = re.search(r'functional\s*=\s*"([^"]+)"', text)
    xc = match[1].strip() if match else ''
    element = re.search(r'element\s*=\s*"\s*([^"\s]+)\s*"', text)
    valence = re.search(r'z_valence\s*=\s*"\s*([^"\s]+)\s*"', text)
    if not element or element[1] != 'Si' or not valence or float(valence[1]) != 4:
        raise ValueError('These four-occupied-band presets require a silicon UPF with four valence electrons')
    correct = ('PBE' in xc.upper()) if functional == 'pbe' else ('PW' in xc and 'PBE' not in xc and 'NOGX' in xc)
    if not correct or not re.search(r'pseudo_type\s*=\s*"NC"', text):
        raise ValueError(f'{target}: expected a norm-conserving {functional} pseudopotential, found XC {xc!r}')
    if not override and digest(target) != expected:
        raise ValueError(f'Cached pseudopotential checksum changed: {target}')
    return target, {'source_url': url if not override else None, 'sha256': digest(target), 'functional': xc}


def write_models(folder, config):
    functional = 'lda-PW' if config['functional'] == 'lda-pw' else 'gga-PBE'
    model = f'''# Generated literature XC preset; all native energies/cutoffs are Hartree.
lattice face-centered Cubic {config['lattice_bohr']}
coords-type lattice
ion-species pseudo/Si.upf
ion Si 0 0 0 1
ion Si 0.25 0.25 0.25 1
spintype no-spin
elec-ex-corr {functional}
elec-cutoff {config['cutoff_Ha']}
elec-n-bands {config['stored_bands']}
converge-empty-states yes
electronic-minimize nIterations 400 energyDiffThreshold 1e-12
'''
    common = f'''include model.in
kpoint-folding {' '.join(map(str, config['mesh']))}
electronic-scf nIterations 200 nEigSteps 40 residualThreshold 1e-10
'''
    (folder/'model.in').write_text(model)
    (folder/'common.in').write_text(common)
    (folder/'01-scf.in').write_text('include common.in\ndump-name state/scf.$VAR\n'
                                 'dump End State BandEigs EigStats ElecDensity Vscloc\n')
    for name in ('paths', 'state', 'results', 'logs', 'pseudo', 'qe/out'):
        (folder/name).mkdir(parents=True, exist_ok=True)
    for path in (EXAMPLE/'paths').iterdir():
        if path.is_file():
            shutil.copy2(path, folder/'paths'/path.name)
    # QE automatic zero-offset grids include Gamma, including even dimensions
    # (PW/src/kpoint_grid.f90). Symmetry reduction avoids a full explicit list.
    dielectric_grid = ' '.join(map(str, config['dielectric_mesh']))
    qe_xc = 'SLA PW NOGX NOGC' if config['functional'] == 'lda-pw' else 'PBE'
    qe = f'''&CONTROL
 calculation='scf', prefix='si', outdir='./out', pseudo_dir='../pseudo', verbosity='high'
/
&SYSTEM
 ibrav=2, celldm(1)={config['lattice_bohr']}, nat=2, ntyp=1,
 ecutwfc={2*config['cutoff_Ha']}, ecutrho={8*config['cutoff_Ha']},
 input_dft='{qe_xc}', occupations='fixed', nspin=1, nbnd=4
/
&ELECTRONS
 conv_thr=1.0d-12, electron_maxstep=200, mixing_beta=0.4
/
ATOMIC_SPECIES
Si 28.0855 Si.upf
ATOMIC_POSITIONS crystal
Si 0 0 0
Si 0.25 0.25 0.25
K_POINTS automatic
{dielectric_grid} 0 0 0
'''
    (folder/'qe/scf.in').write_text(qe)
    (folder/'qe/dielectric.in').write_text('''Silicon electronic dielectric tensor at Gamma
&INPUTPH
 prefix='si', outdir='./out', tr2_ph=1.0d-16, epsil=.true., trans=.false.
/
0.0 0.0 0.0
''')


def write_wannier(folder, config, epsilon):
    import numpy as np
    eigs = np.fromfile(folder/'state/scf.eigenvals', dtype=np.float64)
    bands = config['stored_bands']
    if eigs.size % bands or not eigs.size:
        raise ValueError('SCF eigenvalues are missing or inconsistent with the stored band count')
    eigs = eigs.reshape(-1, bands)
    vbm, cbm = float(eigs[:, 3].max()), float(eigs[:, 4].min())
    if cbm <= vbm:
        raise ValueError('The saved SCF is not an integer-occupation insulator')
    lower = (vbm+cbm)/2
    # No more than four frozen empty states at any k. Include at least the
    # four lowest empty states everywhere in the outer disentanglement window.
    inner = min(cbm+3.0/EV, float(eigs[:, 8].min())-1e-6)
    outer = max(float(eigs[:, 7].max())+.5/EV, cbm+12.0/EV)
    if inner <= lower:
        raise ValueError('Cannot construct a four-orbital empty frozen window from this SCF')
    prefix = 'include common.in\nwannier-initial-state state/scf.$VAR\n'
    minimize = 'wannier-minimize nIterations 600 energyDiffThreshold 1e-14 knormThreshold 1e-8 convergeAll yes\n'
    occupied = prefix+'wannier-dump-name results/valence.$VAR\n'
    occupied += 'wannier bStart 0 loadRotations yes saveWfns no\n'+minimize
    for center in ((.125,.125,.125),(.625,.125,.125),(.125,.625,.125),(.125,.125,.625)):
        occupied += 'wannier-center Gaussian '+' '.join(map(str, center))+'\n'
    empty = prefix+'wannier-dump-name results/conduction.$VAR\n'
    empty += f'wannier loadRotations yes saveWfns no innerWindow {lower:.15g} {inner:.15g} outerWindow {lower:.15g} {outer:.15g}\n'+minimize
    # Four atom-centered tetrahedral sp3 trial combinations. Occupied bands
    # are excluded by the outer lower bound, yielding antibonding trials.
    for signs in ((-1,-1,-1),(1,1,-1),(1,-1,1),(-1,1,1)):
        terms = ['Si 2 s 0.5']+[f'Si 2 p{axis} {sign*.5}' for axis, sign in zip('xyz', signs)]
        empty += 'wannier-center '+' '.join(terms)+'\n'
    (folder/'02-valence-localize.in').write_text(occupied)
    (folder/'03-conduction-localize.in').write_text(empty)
    for number, name, text in ((4,'valence',occupied),(5,'conduction',empty)):
        frozen = text.replace(minimize, 'wannier-minimize nIterations 0\n')+'include ki-response.in\n'
        (folder/f'{number:02d}-{name}-ki.in').write_text(frozen)
    response = f'''# One-shot KI(2); dielectric constant computed for this XC/geometry/mesh.
wannier-koopmans {epsilon:.15g} 400 1e-7 yes 1e-6 yes {config['q_groups']} yes
wannier-koopmans-spin yes
wannier-koopmans-verbose no
wannier-koopmans-response no yes 0.4 8 80 2048 512 4096
'''
    (folder/'ki-response.in').write_text(response)
    windows = {'SCF_VBM_Ha': vbm, 'SCF_CBM_Ha': cbm, 'SCF_mesh_gap_eV': (cbm-vbm)*EV,
               'outer_lower_Ha': lower, 'inner_upper_Ha': inner, 'outer_upper_Ha': outer,
               'conduction_wanniers': 4, 'epsilon_infinity': epsilon,
               'window_note': 'SCF-relative energy windows; choices for this native rerun, not archived Wannier90 window values'}
    (folder/'wannier-windows.json').write_text(json.dumps(windows, indent=2)+'\n')


def dielectric(path):
    import numpy as np
    text = path.read_text()
    start = text.lower().rfind('dielectric constant in cartesian axis')
    if start < 0 or 'JOB DONE.' not in text:
        raise ValueError(f'Missing completed dielectric tensor in {path}')
    rows = []
    for line in text[start:].splitlines()[1:]:
        values = re.findall(r'[-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?', line)
        if len(values) == 3:
            rows.append([float(x.replace('D','E').replace('d','e')) for x in values])
            if len(rows) == 3:
                break
    if len(rows) != 3:
        raise ValueError('Cannot parse the electronic dielectric tensor')
    tensor = np.array(rows)
    epsilon = float(np.trace(tensor)/3)
    if not np.isfinite(tensor).all() or epsilon < 1 or np.linalg.norm(tensor-epsilon*np.eye(3)) > 1e-3*epsilon:
        raise ValueError('The computed dielectric tensor is not a finite cubic electronic dielectric constant')
    return epsilon, tensor.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--functional', choices=PSEUDOS, default='lda-pw')
    parser.add_argument('--mesh', type=int, nargs=3, default=(4,4,4))
    parser.add_argument('--dielectric-mesh', type=int, nargs=3, default=(12,12,12),
                        help='Separate QE mesh for electronic dielectric response (default 12 12 12)')
    parser.add_argument('--cutoff', type=float, default=40., help='Wavefunction cutoff in Ha; 40 Ha = 80 Ry')
    parser.add_argument('--bands', type=int, default=20)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--pseudo', type=Path, help='Explicit matching NC UPF instead of the pinned PseudoDojo file')
    parser.add_argument('--epsilon-infinity', type=float, help='Override QE DFPT; 0 explicitly selects background electrostatics')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Prepare and check native SCF input; check Wannier inputs if SCF exists')
    parser.add_argument('--resume', action='store_true', help='Reuse completed matching stages in the selected directory')
    parser.add_argument('--stop-after', choices=('scf','dielectric','wannier','ki'), default='ki')
    args = parser.parse_args()
    ranks, threads = int(os.environ.get('MPI_RANKS','2')), int(os.environ.get('WANNIER_THREADS','4'))
    if min((*args.mesh,*args.dielectric_mesh)) < 1 or not math.isfinite(args.cutoff) or args.cutoff <= 0 or args.bands < 9 or ranks < 1 or threads < 1:
        parser.error('Mesh, cutoff, ranks and threads must be positive; at least nine stored bands are required')
    if args.epsilon_infinity is not None and not (args.epsilon_infinity == 0 or 1 <= args.epsilon_infinity < float('inf')):
        parser.error('epsilon-infinity must be 0 or finite and >=1')
    pp, metadata = pseudo(args.functional, args.pseudo)
    config = {'functional': args.functional, 'spectral_functional': 'one-shot KI(2)', 'mesh': list(args.mesh),
              'cutoff_Ha': args.cutoff, 'stored_bands': args.bands, 'lattice_bohr': 10.2622,
              'occupied_wanniers': 4, 'empty_wanniers': 4, 'q_groups': ranks,
              'epsilon_override': args.epsilon_infinity, 'pseudo': metadata,
              'dielectric_mesh': list(args.dielectric_mesh),
              'backend': 'cpu' if args.cpu else 'gpu', 'ranks': ranks, 'threads': threads}
    folder = (args.directory or ROOT/'runs'/(args.functional+'-ki2-'+'x'.join(map(str,args.mesh)))).resolve()
    # Serialize invocations, including dry runs, before preparing any files.
    folder.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (folder/'.run.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError(f'Another launcher is using {folder}')
    manifest = folder/'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['configuration'] != config:
            raise ValueError('Run settings changed; choose a new --directory to preserve the existing calculation')
        if not (args.resume or args.prepare_only or args.dry_run):
            raise ValueError(f'{folder} already exists; use --resume or select a new --directory')
    else:
        saved = {'configuration': config, 'created_utc': datetime.now(timezone.utc).isoformat(),
                 'reference': 'https://doi.org/10.1021/acs.jctc.2c00161', 'launcher_sha256': digest(__file__)}
    saved['last_launcher_sha256'] = digest(__file__)
    write_models(folder, config)
    shutil.copy2(pp, folder/'pseudo/Si.upf')
    manifest.write_text(json.dumps(saved, indent=2)+'\n')
    print(f'Prepared {args.functional} + KI(2), four occupied/four empty Wanniers: {folder}', flush=True)
    print('Dielectric: '+('QE DFPT for the selected XC' if args.epsilon_infinity is None else str(args.epsilon_infinity)), flush=True)
    if args.prepare_only:
        return
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1')
    launch = [os.environ.get('MPIEXEC','mpirun'), '-n', str(ranks)] if ranks > 1 else []
    suffix = '' if args.cpu else '_gpu'
    jdftx = Path(os.environ.get('JDFTX_BINARY', REPO/'build'/('jdftx'+suffix)))
    wannier = Path(os.environ.get('WANNIER_BINARY', REPO/'build'/('wannier'+suffix)))
    status_path = folder/'stages.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    def stage(binary, inp, label, cwd=folder, native=True, dry=False):
        binary = Path(binary).resolve()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f'Missing executable {binary}')
        hasher = hashlib.sha256()
        def include(path):
            hasher.update(path.read_bytes())
            if native:
                for line in path.read_text().splitlines():
                    words = line.split()
                    if words and words[0] == 'include':
                        include(folder/words[1])
        include(cwd/inp)
        if native and binary == wannier.resolve():
            hasher.update((folder/'state/scf.wfns').read_bytes())
            if label in ('valence','conduction'):
                hasher.update((folder/f'results/{label}.mlwfU').read_bytes())
        if label == 'qe-dielectric':
            hasher.update((folder/'logs/qe-scf.out').read_bytes())
        sha = hasher.hexdigest()
        runtime = {'binary': str(binary), 'binary_sha256': digest(binary)}
        binary_suffix = '_gpu' if binary.name.endswith('_gpu') else ''
        library = binary.parent/('libjdftx'+binary_suffix+'.so')
        if native and library.is_file():
            runtime['library_sha256'] = digest(library)
        products = {
            'scf': ['state/scf.'+x for x in ('wfns','eigenvals','n','Vscloc')],
            'qe-scf': ['qe/out/si.save/data-file-schema.xml'],
            'valence-localize': ['results/valence.mlwfU','results/valence.mlwfM0'],
            'conduction-localize': ['results/conduction.mlwfU','results/conduction.mlwfM0'],
            'valence': ['results/valence.'+x for x in ('mlwfHki','mlwfKI','koopmansScreening','koopmansBands')],
            'conduction': ['results/conduction.'+x for x in ('mlwfHki','mlwfKI','koopmansScreening','koopmansBands')],
        }.get(label, [])
        log = folder/'logs'/(label+('.dry-run.out' if dry else '.out'))
        entry = status.get(label, {})
        products_ok = all((folder/p).is_file() for p in products)
        if args.resume and not dry and entry.get('input_sha256') == sha and entry.get('runtime') == runtime and entry.get('completed') and products_ok and log.exists():
            marker = 'Done!' if native else 'JOB DONE.'
            if marker in log.read_text()[-5000:]:
                print(f'Reusing completed {label}', flush=True)
                return
        if log.exists():
            archive = folder/'logs/archive'
            archive.mkdir(exist_ok=True)
            log.rename(archive/(log.name+'.'+datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')))
        command = launch+[str(binary)]
        if native:
            command += ['-c',str(threads),'-i',inp,'-o',str(log)]+(['-n'] if dry else [])
        else:
            command += ['-in',inp]
        print(f'Running {label}', flush=True)
        with (folder/'logs'/(label+'.launcher.log')).open('w') as out:
            if native:
                result = subprocess.run(command,cwd=cwd,env=env,stdout=out,stderr=out)
            else:
                with log.open('w') as qe_out:
                    result = subprocess.run(command,cwd=cwd,env=env,stdout=qe_out,stderr=out)
        text = log.read_text() if log.exists() else ''
        completed = result.returncode == 0 and ('Done!' in text[-5000:] if native else 'JOB DONE.' in text[-5000:])
        if not dry:
            status[label] = {'input_sha256': sha, 'completed': completed, 'runtime': runtime}
            status_path.write_text(json.dumps(status,indent=2)+'\n')
        if result.returncode or (not dry and not completed):
            raise ValueError(f'{label} failed; inspect {log} and {folder}/logs/{label}.launcher.log')

    if args.dry_run:
        # Dump Init is absent: SCF validation cannot overwrite the checkpoint.
        stage(jdftx, '01-scf.in', 'scf', dry=True)
        if (folder/'state/scf.eigenvals').exists():
            epsilon = args.epsilon_infinity
            if epsilon is None and (folder/'dielectric.json').exists():
                epsilon = json.loads((folder/'dielectric.json').read_text())['epsilon_infinity']
            if epsilon is not None:
                write_wannier(folder,config,epsilon)
                # Native Wannier dry runs write sidecars; isolate dump prefixes.
                for name in ('02-valence-localize.in','03-conduction-localize.in'):
                    content = (folder/name).read_text().replace('wannier-dump-name results/', 'wannier-dump-name logs/dry-')
                    inp = 'logs/check-'+name
                    (folder/inp).write_text(content)
                    stage(wannier,inp,Path(name).stem,dry=True)
        print('Dry run completed. Wannier windows are generated after a fresh SCF.',flush=True)
        return
    stage(jdftx,'01-scf.in','scf')
    if args.stop_after == 'scf':
        return
    if args.epsilon_infinity is None:
        qe_bin = Path(os.environ.get('QE_BIN', REPO.parent/'qe-7.5/bin'))
        stage(Path(os.environ.get('QE_PW_BINARY',qe_bin/'pw.x')),'scf.in','qe-scf',cwd=folder/'qe',native=False)
        stage(Path(os.environ.get('QE_PH_BINARY',qe_bin/'ph.x')),'dielectric.in','qe-dielectric',cwd=folder/'qe',native=False)
        epsilon,tensor = dielectric(folder/'logs/qe-dielectric.out')
        (folder/'dielectric.json').write_text(json.dumps({'epsilon_infinity':epsilon,'tensor':tensor,
            'functional':args.functional,'mesh':list(args.dielectric_mesh),'source':'QE ph.x electronic DFPT',
            'log_sha256':digest(folder/'logs/qe-dielectric.out')},indent=2)+'\n')
    else:
        epsilon = args.epsilon_infinity
    print(f'Using epsilonInfinity={epsilon:.12g}',flush=True)
    write_wannier(folder,config,epsilon)
    if args.stop_after == 'dielectric':
        return
    stage(wannier,'02-valence-localize.in','valence-localize')
    stage(wannier,'03-conduction-localize.in','conduction-localize')
    if args.stop_after == 'wannier':
        return
    stage(wannier,'04-valence-ki.in','valence')
    stage(wannier,'05-conduction-ki.in','conduction')
    subprocess.run([sys.executable,str(EXAMPLE/'plot_separate_bands.py'),
                    '--valence',str(folder/'results/valence'),'--conduction',str(folder/'results/conduction'),
                    '--fold',*map(str,args.mesh),'--output-dir',str(folder/'results'),
                    '--dft-label','LDA-PW' if args.functional == 'lda-pw' else 'PBE'],env=env,check=True)
    report = json.loads((folder/'results/koopmans-bands-comparison.json').read_text())
    label = 'LDA-PW' if args.functional == 'lda-pw' else 'PBE'
    issues = []
    if report['gaps'][label]['BZ_search']['gap_eV'] <= 0:
        issues.append('The interpolated base DFT bands overlap; converge the Wannier model and electronic mesh.')
    if report['gaps']['KI']['BZ_search']['gap_eV'] <= 0:
        issues.append('The interpolated KI bands overlap.')
    for manifold, checks in report['checks'].items():
        if checks['negative_relaxed_curvature_orbitals']:
            issues.append(f'{manifold} has negative relaxed orbital curvatures; converge its orbitals and screening.')
    quality = {'passes_basic_gap_checks': not issues, 'issues': issues,
               'note': 'Passing these checks does not replace cutoff, mesh, dielectric and Wannier convergence tests.'}
    (folder/'results/quality-checks.json').write_text(json.dumps(quality,indent=2)+'\n')
    for issue in issues:
        print('Gap comparison needs convergence: '+issue,flush=True)
    print(f'Completed. Results: {folder}/results',flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
