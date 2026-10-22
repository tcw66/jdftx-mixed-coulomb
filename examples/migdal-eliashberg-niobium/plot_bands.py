#!/usr/bin/env python3
"""Compute BCC Nb BdG bands using saved finite-temperature coherent parameters.

Uses the original Wannier Hamiltonian, rather than interpolating the finite
BdG export mesh. Adds Fermi crossings and dense samples near them to resolve
meV gaps inside eV-scale bands. No ME or DFPT recalculation is performed.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'jdftx/scripts'))
from me_postprocess.backend import host
from me_postprocess.bdg import coherent_reduce, nambu_normal, symmetry_errors
from me_postprocess.cli import model_from_config, check_saved_spectrum, write_json

BOHR_ANGSTROM = 0.529177210903


def path_definition(lattice):
    filename = HERE.parent/'dfpt-niobium/plot_dispersion.py'
    spec = importlib.util.spec_from_file_location('nb_bcc_path', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, metadata = module.bcc_path(lattice, 1)
    return metadata


def sample_path(model, definition, intervals, z0):
    reciprocal = 2*np.pi*np.linalg.inv(np.array(definition['primitive_lattice_columns_bohr']))
    points = definition['points']
    kpoints, distances, segment_ids, ticks, breaks, crossings = [], [], [], [], [], []
    distance, segment = 0., 0
    for route in definition['paths']:
        breaks.append(len(kpoints))
        ticks.append(dict(index=len(kpoints), label=route[0], distance=distance))
        for isegment, (start, stop) in enumerate(zip(route[:-1], route[1:])):
            k0, k1 = np.array(points[start]), np.array(points[stop])
            length = np.linalg.norm((k1-k0) @ reciprocal)/BOHR_ANGSTROM
            parameters = np.linspace(0, 1, intervals+1)
            energy = host(model.electrons(k0+parameters[:,None]*(k1-k0))[0])-model.mu
            additions = []
            for band in range(model.nw):
                def dispersion(t):
                    return float(host(model.electrons(k0+t*(k1-k0))[0])[band]-model.mu)
                for i in range(intervals):
                    if energy[i,band]*energy[i+1,band] < 0:
                        root = brentq(dispersion, parameters[i], parameters[i+1], xtol=1e-13)
                        step = 1e-5
                        slope = (dispersion(root+step)-dispersion(root-step))/(2*step*z0)
                        width = min(0.04, 0.012/max(abs(slope),1e-8))
                        additions.extend(np.linspace(max(0.,root-width),min(1.,root+width),121))
                        additions.append(root)
                        crossings.append(dict(segment=segment, start=start, stop=stop,
                            band_index=band, fraction=float(root),
                            distance_inv_angstrom=float(distance+root*length),
                            kpoint=(k0+root*(k1-k0)).tolist(),
                            normal_residual_ev=dispersion(root),
                            zoom_halfwidth_inv_angstrom=float(width*length)))
            parameters = np.unique(np.r_[parameters, additions])
            # Avoid duplicating segment junctions within each connected route.
            if isegment:
                parameters = parameters[1:]
            kpoints.extend(k0+parameters[:,None]*(k1-k0))
            distances.extend(distance+parameters*length)
            segment_ids.extend([segment]*len(parameters))
            distance += length
            ticks.append(dict(index=len(kpoints)-1, label=stop, distance=distance))
            segment += 1
    return (np.array(kpoints), np.array(distances), np.array(segment_ids),
            ticks, breaks, crossings)


def compute_bands(model, kpoints, z0, phi0, chunk):
    # Explicit partner indices support repeated path junctions and arbitrarily
    # close refined points without quantizing the reciprocal coordinates.
    nk = len(kpoints)
    closed = np.vstack((kpoints,-kpoints))
    partner = np.r_[np.arange(nk,2*nk),np.arange(nk)]
    h = np.concatenate([host(model.h(closed[i:i+chunk])) for i in range(0,len(closed),chunk)])
    if np.max(np.abs(h[partner]-h.conj()))>1e-8:
        raise ValueError('local singlet construction requires a real Wannier TR gauge')
    nspin = 2*model.nw
    static = nambu_normal(np.kron(h,np.eye(2)),model.mu,partner)
    phi = phi0*np.kron(np.eye(model.nw),np.array([[0,1],[-1,0]]))
    static[:,:nspin,nspin:] = phi
    static[:,nspin:,:nspin] = phi.conj().T
    metric = np.broadcast_to(z0*np.eye(2*nspin),static.shape)
    bdg, _, _ = coherent_reduce(static,metric,partner,backend=model.backend)
    checks = symmetry_errors(bdg,partner)
    xp = model.xp
    values, vectors = xp.linalg.eigh(xp.asarray(bdg))
    electron_weight = host(xp.sum(xp.abs(vectors[:,:2*model.nw,:])**2,axis=1))
    values = host(values)
    normal = (np.linalg.eigvalsh(h)-model.mu)/z0
    expected_positive = np.repeat(np.sqrt(normal**2+(phi0/z0)**2),2,axis=-1)
    expected = np.sort(np.concatenate((-expected_positive,expected_positive),axis=-1),axis=-1)
    checks['analytic_singlet_poles_ev'] = float(np.max(np.abs(values-expected)))
    if max(checks.values()) > 1e-8:
        raise ValueError(f'BdG band validation failed: {checks}')
    return values[:nk], electron_weight[:nk], normal[:nk], checks


def plot(out, distances, energies, weights, normal, segment_ids, ticks, breaks,
         crossings, gap, temperature):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    fig, axes = plt.subplots(3,1,figsize=(11.5,10),layout='constrained',
                             gridspec_kw={'height_ratios':[1.1,1,1.1]})
    norm = Normalize(0,1)
    def draw(ax, x, e, w, ranges, scale):
        collection = None
        # The two spin copies coincide in this nonmagnetic singlet model.
        for start,stop in ranges:
            if stop-start < 2:
                continue
            for band in range(0,e.shape[1],2):
                xy = np.column_stack((x[start:stop],scale*e[start:stop,band]))
                lines = np.stack((xy[:-1],xy[1:]),axis=1)
                collection = LineCollection(lines,cmap='coolwarm',norm=norm,linewidth=1.1,rasterized=True)
                collection.set_array((w[start:stop-1,band]+w[start+1:stop,band])/2)
                ax.add_collection(collection)
        return collection
    ranges = list(zip(breaks,breaks[1:]+[len(distances)]))
    positions,labels = [],[]
    for tick in ticks:
        label = r'$\Gamma$' if tick['label']=='G' else tick['label']
        if positions and abs(positions[-1]-tick['distance'])<1e-10:
            labels[-1] += ' | '+label
        else:
            positions.append(tick['distance']); labels.append(label)
    for ax, scale, limit in [(axes[0],1.,1.),(axes[1],1000.,8.)]:
        collection = draw(ax,distances,energies,weights,ranges,scale)
        ax.set_xlim(distances[0],distances[-1]); ax.set_ylim(-limit,limit)
        ax.set_xticks(positions,labels)
        for position in positions:
            ax.axvline(position,color='0.85',linewidth=.6)
        ax.axhline(0,color='0.4',linewidth=.6)
        ax.set_ylabel(r'$E-\mu$ ('+('eV)' if scale==1 else 'meV)'))
    axes[0].set_title(f'BCC Nb — coherent Wannier BdG bands at {temperature:g} K',loc='left')
    axes[1].axhspan(-1000*gap,1000*gap,color='#e6e9ed',alpha=.8,zorder=-2)
    axes[1].set_title(f'Near the Fermi level: pairing amplitude Δ = {gap*1000:.4f} meV',loc='left',fontsize=10)
    axes[1].set_xlabel('BCC high-symmetry path')
    if crossings:
        crossing = next((c for c in crossings if .05<c['fraction']<.95),crossings[0])
        center = crossing['distance_inv_angstrom']
        width = crossing['zoom_halfwidth_inv_angstrom']
        selection = (segment_ids==crossing['segment']) & (np.abs(distances-center)<=width*1.001)
        x = distances[selection]-center
        collection = draw(axes[2],x,energies[selection],weights[selection],[(0,len(x))],1000.)
        band = crossing['band_index']
        axes[2].plot(x,1000*normal[selection,band],'--',color='0.6',linewidth=.9,label='normal-state crossing')
        axes[2].plot(x,-1000*normal[selection,band],'--',color='0.6',linewidth=.9)
        axes[2].axhspan(-1000*gap,1000*gap,color='#e6e9ed',alpha=.8,zorder=-2)
        axes[2].set_xlim(-width,width); axes[2].set_ylim(-6,6)
        axes[2].set_title(f"Resolved avoided crossing on {crossing['start'].replace('G','Γ')}–{crossing['stop'].replace('G','Γ')}",loc='left',fontsize=10)
        axes[2].set_xlabel(r'Wavevector displacement from crossing ($\mathrm{\AA}^{-1}$)')
        axes[2].set_ylabel(r'$E-\mu$ (meV)')
        axes[2].legend(loc='upper right',fontsize=8,frameon=False)
        axes[2].axhline(0,color='0.4',linewidth=.6)
    bar = fig.colorbar(collection,ax=axes,shrink=.65,pad=.025)
    bar.set_label('Electron-sector weight')
    fig.savefig(out/'bands.png',dpi=200)
    fig.savefig(out/'bands.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results-gpu')
    parser.add_argument('--temperature',type=float,default=2.)
    parser.add_argument('--lattice',type=Path,default=Path('/home/tyler/Nb_DFPT/Nb.lattice'))
    parser.add_argument('--backend',choices=('cpu','cuda','auto'),default='cuda')
    parser.add_argument('--gpu-device',type=int,default=0)
    parser.add_argument('--points-per-segment',type=int,default=500)
    parser.add_argument('--chunk-size',type=int,default=256)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.points_per_segment<2 or args.chunk_size<1:
        parser.error('invalid sampling or batch size')
    source = args.results.resolve()
    config = json.loads((source/'spectral-summary.json').read_text())['configuration']
    config.update(output=str(source),backend=args.backend,gpu_device=args.gpu_device)
    check_saved_spectrum(config)
    model = model_from_config(config)
    source_file = source/f'bdg-T{args.temperature:g}.npz'
    with np.load(source_file) as archive:
        z0,phi0,mu,t = [float(archive[key]) for key in ('z0','phi0_ev','mu_ev','temperature_k')]
        if not np.isclose(mu,model.mu,atol=1e-10,rtol=0) or t!=args.temperature:
            raise ValueError('BdG archive and requested normal-state model do not match')
        sample = np.arange(0,len(archive['kpoints']),max(1,len(archive['kpoints'])//16))
        np.testing.assert_allclose(host(model.h(archive['kpoints'][sample])),
                                   archive['h_bare_orbital_ev'][sample],atol=1e-9,rtol=1e-10)
    me_summary = json.loads((source/'me-summary.json').read_text())
    me_config = dict(me_summary['configuration'],output=str(source))
    check_saved_spectrum(me_config)
    matches = [row for row in me_summary['temperatures'] if row['temperature_k']==t]
    if len(matches)!=1 or not np.allclose([z0,phi0],
            [matches[0]['z0'],matches[0]['phi0_ev']],atol=1e-14,rtol=0):
        raise ValueError('BdG archive does not match the saved ME solution')
    definition = path_definition(args.lattice)
    k,d,segments,ticks,breaks,crossings = sample_path(model,definition,args.points_per_segment,z0)
    values,weights,normal,checks = compute_bands(model,k,z0,phi0,args.chunk_size)
    gap = abs(phi0/z0)
    min_positive = float(values[:,values.shape[1]//2:].min())
    checks['minimum_positive_energy_ev'] = min_positive
    checks['minimum_minus_pairing_amplitude_ev'] = min_positive-gap
    if crossings and abs(min_positive-gap)>1e-8:
        raise ValueError('Fermi-crossing gap check failed')
    out = args.output.resolve() if args.output else source/f'bands-T{t:g}'
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'bands.npz',kpoints=k,distance_inv_angstrom=d,
        bdg_eigenvalues_ev=values,electron_weight=weights,normal_qp_eigenvalues_ev=normal,
        segment_ids=segments,temperature_k=t,z0=z0,phi0_ev=phi0,delta_qp_ev=phi0/z0,mu_ev=mu)
    np.savetxt(out/'bands.dat',np.column_stack((d,k,values)),
        header='distance[Angstrom^-1] k0 k1 k2 BdG eigenvalues E-mu [eV], ascending (24 spinful Nambu bands)')
    report = dict(temperature_k=t,z0=z0,phi0_ev=phi0,delta_qp_ev=phi0/z0,mu_ev=mu,
        source_bdg=str(source_file),source_configuration=config,backend=model.backend.info,
        source_bdg_mtime_ns=source_file.stat().st_mtime_ns,
        path=definition,kpoint_count=len(k),ticks=ticks,breaks=breaks,
        distance_units='Angstrom^-1',fermi_crossings=crossings,checks=checks,
        interpretation='static isotropic local-singlet coherent model; not real-axis ME spectral bands')
    write_json(out/'bands.json',report)
    plot(out,d,values,weights,normal,segments,ticks,breaks,crossings,gap,t)
    print(f'Saved {len(k)} k points, {values.shape[1]} BdG bands, {len(crossings)} Fermi crossings to {out}')
    print(f'Gap edge {min_positive*1000:.8g} meV; full gap {2*gap*1000:.8g} meV; pole error {checks["analytic_singlet_poles_ev"]:.3g} eV')


if __name__=='__main__':
    main()
