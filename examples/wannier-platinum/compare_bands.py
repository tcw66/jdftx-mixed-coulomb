#!/usr/bin/env python3
"""Prepare the fcc band path, or compare native DFT and Wannier bands."""
import argparse
import json
from pathlib import Path
import re
import numpy as np

EV = 27.211386245988
ROOT = Path(__file__).resolve().parent

def command(text, name):
    matches = re.findall(r'^'+re.escape(name)+r'\s+([^#\n]+)', text, re.M)
    if not matches:
        raise ValueError('Missing input command: '+name)
    return matches[-1].split()

def prepare(steps):
    # Primitive fcc reciprocal coordinates, for R columns (0,a/2,a/2), etc.
    labels = ['Γ', 'X', 'W', 'K', 'Γ', 'L', 'U', 'W', 'L', 'K']
    nodes = np.array([[0,0,0], [.5,0,.5], [.5,.25,.75], [.375,.375,.75],
                      [0,0,0], [.5,.5,.5], [.625,.25,.625],
                      [.5,.25,.75], [.5,.5,.5], [.375,.375,.75]])
    points = [nodes[0]]
    ticks = [0]
    for a,b in zip(nodes[:-1],nodes[1:]):
        points.extend(a+(b-a)*t/steps for t in range(1,steps+1))
        ticks.append(len(points)-1)
    points = np.asarray(points)
    model = (ROOT/'model.in').read_text()
    lattice = command(model, 'lattice')
    if lattice[:2] != ['face-centered', 'Cubic']:
        raise ValueError('This path is defined for the primitive fcc cubic lattice.')
    a = float(lattice[2])
    r = .5*a*(np.ones((3,3))-np.eye(3))
    reciprocal = 2*np.pi*np.linalg.inv(r).T
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points,axis=0)@reciprocal.T,axis=1))]
    result = ROOT/'results'
    result.mkdir(exist_ok=True)
    np.savetxt(result/'bandpath.dat', np.column_stack([distance,points]), header='distance(bohr^-1) k1 k2 k3')
    (result/'bandpath.json').write_text(json.dumps(dict(labels=labels,ticks=ticks)))
    (result/'bandpath.kpoints').write_text('kpoint-folding 1 1 1\nsymmetries none\n'+''.join(
        f'kpoint {x:.12f} {y:.12f} {z:.12f} {1/len(points):.17g}\n' for x,y,z in points))
    print(f'Prepared {len(points)} k points on '+ '–'.join(labels))

def compare():
    result = ROOT/'results'
    path = np.loadtxt(result/'bandpath.dat')
    meta = json.loads((result/'bandpath.json').read_text())
    common = (ROOT/'common.in').read_text()
    fold = np.array(command(common,'kpoint-folding'), dtype=int)
    centers = np.loadtxt(result/'pt.mlwfCenters', ndmin=2)
    nw = len(centers)
    nc = int(np.prod(fold))
    # Native no-spin Hamiltonian is real64, column-major within each cell.
    reduced = np.fromfile(result/'pt.mlwfH', dtype=np.float64)
    if reduced.size != nc*nw*nw:
        raise ValueError('Wannier files do not match the current folding and center count. Rerun Wannierization.')
    reduced = reduced.reshape(nc,nw,nw).swapaxes(1,2)
    cells = np.loadtxt(result/'pt.mlwfCellMap', usecols=(0,1,2), dtype=int, ndmin=2)
    weights = np.fromfile(result/'pt.mlwfCellWeights', dtype=np.float64).reshape(len(cells),nw,nw).swapaxes(1,2)
    indices = (cells % fold)@np.array([fold[1]*fold[2],fold[2],1])
    hopping = weights*reduced[indices]
    # Chunk the path to limit memory if the center count is increased.
    bands = []
    for points in np.array_split(path[:,1:4], max(1,len(path)//32)):
        h = np.einsum('kr,rij->kij', np.exp(2j*np.pi*points@cells.T), hopping)
        error = np.max(np.abs(h-h.swapaxes(1,2).conj()))
        if error > 1e-7:
            raise ValueError(f'Non-Hermitian interpolated Hamiltonian: {error:g} Ha')
        bands.append(np.linalg.eigvalsh(.5*(h+h.swapaxes(1,2).conj())))
    wannier = np.concatenate(bands)
    dft = np.fromfile(result/'bandstruct.eigenvals', dtype=np.float64)
    if dft.size % len(path):
        raise ValueError('DFT eigenvalue count does not match path.')
    dft = dft.reshape(len(path),-1)
    log = (result/'scf.out').read_text()
    mus = re.findall(r'FillingsUpdate:\s+mu:\s+(\S+)',log)
    if not mus:
        mus = re.findall(r'^\s*mu\s*:\s*(\S+)',log,re.M)
    mu = float(mus[-1])
    dft, wannier = (dft-mu)*EV, (wannier-mu)*EV
    np.savetxt(result/'dft-bands.dat',np.column_stack([path[:,0],dft]),header='distance(bohr^-1); band energies relative to SCF EF (eV)')
    np.savetxt(result/'wannier-bands.dat',np.column_stack([path[:,0],wannier]),header='distance(bohr^-1); band energies relative to SCF EF (eV)')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(10,6))
    ax.plot(path[:,0],dft,color='0.45',lw=1, label=None)
    ax.plot([],[],color='0.45',lw=1,label='DFT')
    ax.plot(path[:,0],wannier,'--',color='#cf392f',lw=1.2)
    measure = re.search(r'localizationMeasure\s+(\S+)', (result/'windows.in').read_text()).group(1)
    ax.plot([],[],'--',color='#cf392f',lw=1.2,label=f'{measure} ({nw} Wannier bands)')
    windows = (result/'windows.in').read_text().replace('\\\n',' ')
    inner = re.search(r'innerWindow\s+(\S+)\s+(\S+)',windows)
    if inner:
        low,high = (np.array([float(x) for x in inner.groups()])-mu)*EV
        ax.axhspan(low,high,color='#61bda5',alpha=.12,label='Inner window')
        mask = (dft>=low)&(dft<=high)
        errors = np.min(np.abs(dft[:,:,None]-wannier[:,None,:]),axis=2)[mask]
        if len(errors):
            report = dict(inner_window_states=int(len(errors)),nearest_band_rms_eV=float(np.sqrt(np.mean(errors**2))),nearest_band_max_eV=float(np.max(errors)))
            (result/'band-errors.json').write_text(json.dumps(report,indent=2)+'\n')
            print('Inner-window nearest-band errors (diagnostic, not band-index matching):',report)
    ax.axhline(0,color='k',lw=.7)
    for t in meta['ticks']:
        ax.axvline(path[t,0],color='0.8',lw=.5)
    ax.set_xticks(path[meta['ticks'],0],meta['labels'])
    ax.set(xlim=(path[0,0],path[-1,0]),ylim=(-12,12),ylabel=r'$E-E_F$ (eV)',title='Bulk fcc platinum: DFT and Wannier interpolation')
    ax.legend(loc='upper right')
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(result/f'band-comparison.{suffix}',dpi=180)
    print('Wrote results/band-comparison.png and .pdf, and band tables.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--steps',type=int,default=20,help='intervals per path segment')
    args = parser.parse_args()
    if args.steps<1:
        parser.error('--steps must be positive')
    prepare(args.steps) if args.prepare else compare()
