#!/usr/bin/env python3
"""Plot §16 local diagnostics of the converged 2 K Nb vacancy model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'jdftx/scripts'))
from me_postprocess.backend import Backend
from me_postprocess.cli import write_json
from me_postprocess.local_profiles import (onsite_and_bonds,correlation_blocks,
    uniform_bulk_blocks,shell_average,threshold_half_gap)
from plot_bands import path_definition
from vacancy import distances


def minimum_images(cells,lattice,shape):
    translations=np.indices((5,5,5)).reshape(3,-1).T-2
    vectors=(cells[:,None]+translations[None]*shape) @ lattice.T*.529177210903
    chosen=np.argmin(np.linalg.norm(vectors,axis=-1),axis=1)
    return vectors[np.arange(len(cells)),chosen]


def save_figure(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=190)
    fig.savefig(out/(name+'.pdf'))


def plots(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    radius=data['site_radius_angstrom']
    gap_ratio=data['onsite_gap_rms_ratio']
    pair_ratio=data['onsite_pair_norm_ratio']
    bond_r=data['bond_radius_angstrom'];bond_ratio=data['bond_pair_norm_ratio']
    energy=data['ldos_energy_ev']*1000
    ldos=data['ldos_per_spin_ev'];bulk_ldos=data['bulk_ldos_per_spin_ev']
    fig,axes=plt.subplots(3,1,figsize=(9,10.5),layout='constrained',sharex=True)
    def radial(ax,r,y,color,label):
        x,n,mean,low,high=shell_average(r,y)
        ax.scatter(r,y,s=16,alpha=.25,color=color)
        ax.errorbar(x,mean,yerr=np.array([mean-low,high-mean]),fmt='o-',color=color,
                    markersize=4,capsize=3,linewidth=1,label=label)
    radial(axes[0],radius,gap_ratio,'#28639a','onsite pairing field')
    axes[0].set_ylabel(r'$\|\Delta_i\|_F / (\sqrt{6}\Delta_{\rm bulk})$')
    axes[0].set_title('Nb vacancy at 2 K — local superconducting profiles',loc='left')
    radial(axes[1],radius,pair_ratio,'#28639a','onsite pair correlation')
    radial(axes[1],bond_r,bond_ratio,'#bb5731','nearest-neighbor pair correlation')
    axes[1].set_ylabel(r'$\|F\|_F / \|F_{\rm bulk}\|_F$')
    axes[1].legend(fontsize=8,frameon=False)
    for ax in axes[:2]:
        ax.axhline(1,color='0.45',linestyle='--',linewidth=.9)
    for j,(fraction,color) in enumerate(zip(data['threshold_fractions'],['#28639a','#bb5731','#6b4a86'])):
        radial(axes[2],radius,data['threshold_half_gap_ev'][:,j]*1000,color,f'{fraction*100:g}% LDOS threshold')
        axes[2].axhline(data['bulk_threshold_half_gap_ev'][j]*1000,color=color,linestyle='--',linewidth=.8)
    axes[2].set_ylabel('Apparent LDOS half-gap (meV)')
    axes[2].set_xlabel('Distance from vacancy / bond midpoint distance (Å)')
    axes[2].legend(fontsize=8,frameon=False)
    axes[2].set_title('Lorentzian η = 0.05 meV; dashed lines: bulk reference',loc='left',fontsize=9)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.supxlabel('Points: individual sites/bonds. Bars: shell min–max spread, not statistical uncertainty.',fontsize=8)
    save_figure(fig,out,'radial_profiles');plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(10,8),layout='constrained')
    shell_r,counts,mean,_,_=shell_average(radius,ldos)
    keep=np.abs(energy)<=4.00001
    image=axes[0].imshow(mean[:,keep],aspect='auto',origin='lower',interpolation='nearest',
        extent=[energy[keep][0],energy[keep][-1],-.5,len(shell_r)-.5],cmap='magma',vmin=0)
    axes[0].set_yticks(np.arange(len(shell_r)),[f'{r:.2f} ({n})' for r,n in zip(shell_r,counts)])
    axes[0].set_ylabel('Shell radius Å (Nb site count)')
    axes[0].set_xlabel('Energy from μ (meV)')
    axes[0].set_title('Shell-averaged coherent electron LDOS at 2 K',loc='left')
    fig.colorbar(image,ax=axes[0],label='States / spin / site / eV')
    nearest=np.isclose(radius,radius.min(),atol=1e-6)
    farthest=np.isclose(radius,radius.max(),atol=1e-6)
    axes[1].plot(energy,bulk_ldos,'--',color='black',label='homogeneous bulk')
    axes[1].plot(energy,ldos[nearest].mean(axis=0),color='#bb5731',label='nearest Nb neighbors')
    axes[1].plot(energy,ldos[farthest].mean(axis=0),color='#28639a',label='farthest Nb sites')
    axes[1].set_xlim(-4,4);axes[1].set_xlabel('Energy from μ (meV)')
    axes[1].set_ylabel('States / spin / site / eV')
    axes[1].legend(fontsize=9,frameon=False)
    axes[1].set_title('Active pairing window; Lorentzian η = 0.05 meV',loc='left',fontsize=10)
    save_figure(fig,out,'spectral_profiles');plt.close(fig)

    fig=plt.figure(figsize=(12,5.5),layout='constrained')
    positions=data['site_positions_angstrom']
    for panel,(values,title) in enumerate([(gap_ratio,'Onsite pairing-field norm / bulk'),
                                         (pair_ratio,'Onsite pair-correlation norm / bulk')],1):
        ax=fig.add_subplot(1,2,panel,projection='3d')
        norm=Normalize(min(values.min(),.7),max(values.max(),1.))
        obj=ax.scatter(*positions.T,c=values,cmap='viridis',norm=norm,s=65,edgecolor='0.3',linewidth=.3,depthshade=False)
        ax.scatter([0],[0],[0],color='black',marker='X',s=110,label='vacancy: orbitals removed')
        ax.set_xlabel('x (Å)');ax.set_ylabel('y (Å)');ax.set_zlabel('z (Å)')
        ax.set_title(title,fontsize=10);ax.set_box_aspect([1,1,1])
        ax.legend(loc='upper left',fontsize=7,frameon=False)
        fig.colorbar(obj,ax=ax,shrink=.6,pad=.05)
    fig.suptitle('Wannier-site maps — periodic 4³ vacancy array, T = 2 K')
    save_figure(fig,out,'spatial_profiles');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results-vacancy-T2-final')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--backend',choices=('cpu','cuda','auto'),default='cuda')
    parser.add_argument('--gpu-device',type=int,default=1)
    parser.add_argument('--lattice',type=Path,default=Path('/home/tyler/Nb_DFPT/Nb.lattice'))
    parser.add_argument('--reuse-correlations',action='store_true')
    args=parser.parse_args()
    source=args.results.resolve();out=args.output.resolve() if args.output else source/'local-profiles'
    out.mkdir(parents=True,exist_ok=True)
    backend=Backend(args.backend,args.gpu_device)
    a=np.load(source/'vacancy-T2.npz');basis=np.load(source/'vacancy-basis.npz')
    cells,shape=a['cells'],a['supercell'];nsite,norb=a['onsite_delta_ev'].shape
    t,z,gap=float(a['temperature_k']),float(a['z0']),float(a['bulk_gap_ev'])
    lattice=np.array(path_definition(args.lattice)['primitive_lattice_columns_bohr'])
    left,right,images,steps=onsite_and_bonds(cells,shape)
    nbonds=len(left)
    all_left=np.r_[np.arange(nsite),left,right]
    all_right=np.r_[np.arange(nsite),right,left]
    all_images=np.vstack((np.zeros((nsite,3),int),images,-images))
    cache=out/'correlations.npz'
    summary=json.loads((source/'summary.json').read_text())
    bulk_name='bulk-basis-'+'x'.join(map(str,summary['bulk_mesh']))+'.npz'
    bulk_file=source/bulk_name
    if not bulk_file.exists():
        bulk_file=source/'bulk-basis.npz'
    signature=hashlib.sha256((source/'vacancy-T2.npz').read_bytes()).hexdigest()
    if args.reuse_correlations and cache.exists():
        saved=np.load(cache)
        if str(saved['source_sha256'])!=signature:
            raise ValueError('correlation cache comes from a different converged solution')
        blocks,bulk_blocks,offsets=saved['blocks'],saved['bulk_blocks'],saved['offsets']
    else:
        print(f'Computing {nsite} onsite and {2*nbonds} directed nearest-neighbor blocks',flush=True)
        blocks=correlation_blocks(a,basis,all_left,all_right,all_images,t,int(basis['total_points']),
            norb,backend,progress=lambda size:print(f'Correlation blocks: active dimension {size}',flush=True))
        offsets=np.vstack((np.zeros((1,3),int),np.unique(steps,axis=0)))
        with np.load(bulk_file) as reference:
            bulk_blocks=uniform_bulk_blocks(reference,gap,offsets,t,backend)
        np.savez_compressed(cache,blocks=blocks,bulk_blocks=bulk_blocks,offsets=offsets,source_sha256=signature)
    onsite=blocks[:nsite]
    bond=blocks[nsite:nsite+nbonds]
    reverse=blocks[nsite+nbonds:]
    diagonal=np.diagonal(onsite,axis1=1,axis2=2)
    checks=dict(onsite_anomalous_density_error=float(np.max(np.abs(diagonal-a['anomalous_density']))),
        onsite_singlet_symmetry_error=float(np.max(np.abs(onsite-onsite.swapaxes(-1,-2)))),
        bond_singlet_symmetry_error=float(np.max(np.abs(bond-reverse.swapaxes(-1,-2)))),
        imaginary_correlation_error=float(np.max(np.abs(blocks.imag))),
        bulk_gap_calibration_error_ev=float(np.max(np.abs(np.diag(bulk_blocks[0])*a['attraction_ev']-gap))))
    if max(checks.values())>1e-8:
        raise ValueError(f'Local-correlation validation failed: {checks}')
    onsite=(onsite+onsite.swapaxes(-1,-2))/2
    bond=(bond+reverse.swapaxes(-1,-2))/2
    pair_norm=np.linalg.norm(onsite,axis=(-1,-2))
    bulk_pair=np.linalg.norm(bulk_blocks[0])
    offset_lookup={tuple(r):i for i,r in enumerate(offsets)}
    bond_reference=np.array([np.linalg.norm(bulk_blocks[offset_lookup[tuple(r)]]) for r in steps])
    pair_ratio=pair_norm/bulk_pair
    gap_rms=np.sqrt(np.mean(np.abs(a['onsite_delta_ev'])**2,axis=1))
    bond_ratio=np.linalg.norm(bond,axis=(-1,-2))/bond_reference
    bond_radius=distances(cells[left]+steps/2,lattice,shape)
    positions=minimum_images(cells,lattice,shape)
    np.testing.assert_allclose(np.linalg.norm(positions,axis=1),a['distance_angstrom'],atol=1e-8)
    fractions=np.array([.05,.10,.20]);ldos=a['ldos_electron_coherent_per_spin_ev']
    bulk_ldos=a['bulk_ldos_electron_coherent_per_spin_ev'];energy=a['ldos_energy_ev']
    thresholds=np.column_stack([threshold_half_gap(energy,ldos,f)[0] for f in fractions])
    bulk_thresholds=np.array([threshold_half_gap(energy,bulk_ldos,f)[0][0] for f in fractions])
    data=dict(site_cells=cells,site_positions_angstrom=positions,site_radius_angstrom=a['distance_angstrom'],
        onsite_delta_ev=a['onsite_delta_ev'],onsite_gap_rms_ratio=gap_rms/gap,
        onsite_anomalous_blocks=onsite,onsite_pair_norm_ratio=pair_ratio,
        onsite_coherent_electron_pair_blocks=onsite/z,bulk_onsite_anomalous_block=bulk_blocks[0],
        bond_left=left,bond_right=right,bond_image=images,bond_step=steps,bond_radius_angstrom=bond_radius,
        bond_anomalous_blocks=bond,bond_pair_norm_ratio=bond_ratio,
        threshold_fractions=fractions,threshold_half_gap_ev=thresholds,
        bulk_threshold_half_gap_ev=bulk_thresholds,
        ldos_energy_ev=energy,ldos_per_spin_ev=ldos,bulk_ldos_per_spin_ev=bulk_ldos)
    np.savez_compressed(out/'local-profiles.npz',**data)
    np.savetxt(out/'site-profiles.dat',np.column_stack((a['site_indices'],cells,a['distance_angstrom'],
        gap_rms*1000,gap_rms/gap,pair_norm,pair_ratio,thresholds*1000)),
        header='site cell0 cell1 cell2 radius_A gap_rms_meV gap_over_bulk F_frobenius F_over_bulk apparent_halfgap_meV[5%,10%,20%]')
    np.savetxt(out/'bond-profiles.dat',np.column_stack((left,right,images,steps,bond_radius,
        np.linalg.norm(bond,axis=(-1,-2)),bond_ratio)),
        header='left right image0 image1 image2 step0 step1 step2 midpoint_radius_A F_frobenius F_over_bulk_direction')
    pdf=HERE.parents[1]/'references/migdal_eliashberg_wannier_bdg_local_profiles.pdf'
    write_json(out/'profiles.json',dict(temperature_k=t,backend=backend.info,source=str(source),
        source_sha256=signature,workflow_pdf=str(pdf),workflow_pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        sites=nsite,nearest_neighbor_bonds=nbonds,bulk_pair_frobenius_norm=float(bulk_pair),
        checks=checks,ldos_lorentzian_eta_ev=.00005,ldos_spin_convention='per spin per surviving Nb site',
        threshold_fractions=fractions.tolist(),threshold_peak_window_ev=.0035,
        threshold_definition='average positive and negative first-onset energies at a fraction of the side-specific LDOS peak',
        bulk_threshold_half_gap_mev=(bulk_thresholds*1000).tolist(),
        apparent_halfgap_range_mev={str(f):[float(thresholds[:,i].min()*1000),float(thresholds[:,i].max()*1000)]
                                  for i,f in enumerate(fractions)},
        correlation_convention='canonical quasiparticle anomalous amplitude; coherent electron part F/Z0',
        geometry='4^3 periodic vacancy array; no Wannier orbitals at the missing site',
        limits=['site/bond Wannier diagnostics, not continuum F(r,r)',
            'local threshold gaps depend on eta, threshold and peak window',
            'finite-supercell and k-sampling limits inherited from the converged vacancy model']))
    plots(out,data)
    print(f'Saved local profiles to {out}; checks {checks}',flush=True)


if __name__=='__main__':
    main()
