#!/usr/bin/env python3
"""Fit a static pairing model and solve a periodic Nb vacancy self-consistently."""
import argparse
import json
import hashlib
import io
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'jdftx/scripts'))
from me_postprocess.cli import model_from_config,check_saved_spectrum,write_json
from me_postprocess.backend import host
from me_postprocess.selfconsistent import (ProjectedSpectrum,bulk_spectrum,vacancy_spectrum,
    supercell_matrix,solve_gap)


def snapshot_bulk(source,destination):
    """Freeze a completed bulk result before an independent vacancy run."""
    names = ('spectral-summary.json','me-summary.json','bdg-T2.npz','spectrum.npz')
    payload = {name:(source/name).read_bytes() for name in names}
    spectral,me = [json.loads(payload[name]) for name in names[:2]]
    for key in ('qmesh','kmesh','sigma_ev','mu_star','wannier_prefix','phonon_prefix'):
        if spectral['configuration'][key]!=me['configuration'][key]:
            raise ValueError('bulk stages are inconsistent; wait for the bulk run to finish')
    with np.load(io.BytesIO(payload['bdg-T2.npz'])) as data:
        row = next(row for row in me['temperatures'] if row['temperature_k']==2.)
        if not np.allclose([float(data['z0']),float(data['phi0_ev'])],
                            [row['z0'],row['phi0_ev']],rtol=0,atol=1e-14):
            raise ValueError('bulk ME and BdG stages do not match')
    destination.mkdir(parents=True,exist_ok=True)
    manifest={}
    for name,content in payload.items():
        (destination/name).write_bytes(content)
        manifest[name]=dict(source=str((source/name).resolve()),sha256=hashlib.sha256(content).hexdigest())
    pdf=HERE.parents[1]/'references/migdal_eliashberg_wannier_bdg_updated.pdf'
    manifest['workflow_pdf']=dict(path=str(pdf),sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    write_json(destination/'manifest.json',manifest)


def print_progress(label):
    previous = [-1]
    def progress(done,total):
        percent = int(10*done/total)
        if percent!=previous[0]:
            print(f'{label}: {done}/{total}',flush=True)
            previous[0] = percent
    return progress


def distances(cells,lattice,shape):
    translations = np.indices((5,5,5)).reshape(3,-1).T-2
    displacements = cells[:,None]+translations[None]*shape
    cart = displacements @ lattice.T*0.529177210903
    return np.min(np.linalg.norm(cart,axis=-1),axis=1)


def figures(out,cells,radius,delta,bulk_gap,density_change,energy,ldos,bulk_ldos):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(3,1,figsize=(9,10),layout='constrained')
    names = ['s','dz2','dxy','dxz','dx2-y2','dyz']
    for a,label in enumerate(names):
        axes[0].scatter(radius,delta[:,a]*1000,s=16,alpha=.65,label=label)
    axes[0].axhline(bulk_gap*1000,color='black',linestyle='--',linewidth=1,label='bulk calibration')
    axes[0].set_ylabel('Onsite pairing field (meV)')
    axes[0].set_xlabel('Distance from vacancy (Å)')
    axes[0].set_title('Periodic Nb vacancy — self-consistent onsite singlet fields at 2 K',loc='left')
    axes[0].legend(ncol=4,fontsize=8,frameon=False)
    axes[1].scatter(radius,density_change,color='#335e9c',s=22)
    axes[1].axhline(0,color='0.5',linewidth=.7)
    axes[1].set_xlabel('Distance from vacancy (Å)')
    axes[1].set_ylabel('Model occupation change / site')
    nearest = np.isclose(radius,radius.min(),atol=1e-6)
    farthest = np.isclose(radius,radius.max(),atol=1e-6)
    axes[2].plot(energy*1000,bulk_ldos,label='homogeneous bulk',color='black',linestyle='--')
    axes[2].plot(energy*1000,ldos[nearest].mean(axis=0),label='nearest-neighbor sites',color='#ae333b')
    axes[2].plot(energy*1000,ldos[farthest].mean(axis=0),label='farthest sites in supercell',color='#335e9c')
    axes[2].set_xlabel('Energy from bulk chemical potential (meV)')
    axes[2].set_ylabel('Coherent electron LDOS / spin / site / eV')
    axes[2].legend(fontsize=8,frameon=False)
    axes[2].set_title('Active pairing window; Lorentzian broadening 0.05 meV',loc='left',fontsize=10)
    fig.savefig(out/'vacancy-T2.png',dpi=180)
    fig.savefig(out/'vacancy-T2.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'results-vacancy-T2/input')
    parser.add_argument('--snapshot-from',type=Path,
                        help='freeze a completed bulk result into --input before running')
    parser.add_argument('--output',type=Path,default=HERE/'results-vacancy-T2')
    parser.add_argument('--supercell',type=int,nargs=3,default=[4,4,4])
    parser.add_argument('--twist-mesh',type=int,nargs=3,default=[8,8,8])
    parser.add_argument('--calibration-mesh',type=int,nargs=3,
                        help='independent primitive-cell mesh for fitting the bulk interaction')
    parser.add_argument('--fixed-calibration',type=Path,
                        help='reuse U_a from a pairing-calibration.json when changing geometry or twist mesh')
    parser.add_argument('--calibration-only',action='store_true')
    parser.add_argument('--backend',choices=('cpu','cuda','auto'),default='cuda')
    parser.add_argument('--gpu-device',type=int,default=1)
    parser.add_argument('--normal-batch-size',type=int,default=16)
    parser.add_argument('--lattice',type=Path,default=Path('/home/tyler/Nb_DFPT/Nb.lattice'))
    parser.add_argument('--cutoff-ev',type=float)
    parser.add_argument('--mixing',type=float,default=.4)
    parser.add_argument('--tolerance-ev',type=float,default=1e-9)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    source,out = args.input.resolve(),args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.snapshot_from:
        snapshot_bulk(args.snapshot_from.resolve(),source)
    spectral = json.loads((source/'spectral-summary.json').read_text())
    me = json.loads((source/'me-summary.json').read_text())
    config = dict(spectral['configuration'],output=str(source),backend=args.backend,gpu_device=args.gpu_device)
    check_saved_spectrum(config)
    model = model_from_config(config)
    centers = np.loadtxt(config['wannier_prefix']+'mlwfCenters',usecols=(3,4,5),ndmin=2)
    if model.nw!=6 or centers.shape!=(6,3) or np.max(np.abs(centers))>1e-6:
        raise ValueError('expected the six Nb-site-centered orbitals in the primitive cell')
    with np.load(source/'bdg-T2.npz') as data:
        z0,phi0,mu = [float(data[key]) for key in ('z0','phi0_ev','mu_ev')]
    gap = phi0/z0
    t = 2.
    shape,twist_mesh = np.array(args.supercell),np.array(args.twist_mesh)
    if np.any(shape<2) or np.any(twist_mesh<1):
        parser.error('supercell dimensions must be >=2 and twist dimensions >=1')
    with np.load(source/'spectrum.npz') as data:
        cutoff = float(data['omega_ev'].max()) if args.cutoff_ev is None else args.cutoff_ev
    if cutoff<=0 or not np.isfinite(cutoff):
        parser.error('invalid pairing cutoff')
    bulk_mesh = shape*twist_mesh if args.calibration_mesh is None else np.array(args.calibration_mesh)
    info = dict(supercell=shape.tolist(),twist_mesh=twist_mesh.tolist(),bulk_mesh=bulk_mesh.tolist(),
        z0=z0,bulk_gap_ev=gap,mu_ev=mu,cutoff_ev=cutoff,temperature_k=t,
        source_manifest=json.loads((source/'manifest.json').read_text()))
    bulk_cache_valid=True
    if args.resume and (out/'basis-settings.json').exists():
        previous=json.loads((out/'basis-settings.json').read_text())
        bulk_cache_valid=previous['bulk_mesh']==info['bulk_mesh']
        if {k:v for k,v in previous.items() if k!='bulk_mesh'}!={k:v for k,v in info.items() if k!='bulk_mesh'}:
            raise ValueError('saved basis does not match requested settings')
    write_json(out/'basis-settings.json',info)
    print(f'Backend {model.backend.info}; calibration k mesh {bulk_mesh}; cutoff {cutoff*1000:.6g} meV',flush=True)
    def clean():
        path = out/('bulk-basis-'+'x'.join(map(str,bulk_mesh))+'.npz')
        if args.resume and path.exists():
            return ProjectedSpectrum.load_basis(path,model.backend)
        legacy=out/'bulk-basis.npz'
        if args.resume and legacy.exists() and bulk_cache_valid:
            return ProjectedSpectrum.load_basis(legacy,model.backend)
        result = bulk_spectrum(model,bulk_mesh,z0,cutoff,t,progress=print_progress('Bulk normal basis'))
        result.save_basis(path)
        return result
    bulk = clean()
    fbulk = bulk.evaluate(np.full(model.nw,gap),t)['anomalous']
    attraction = gap/fbulk
    if args.fixed_calibration:
        fitted = json.loads(args.fixed_calibration.read_text())
        for key,value in [('z0',z0),('bulk_gap_ev',gap),('mu_ev',mu),('cutoff_ev',cutoff)]:
            if not np.isclose(fitted[key],value,rtol=1e-12,atol=1e-14):
                raise ValueError(f'fixed calibration has a different {key}')
        attraction = np.array(fitted['attraction_ev'])
    if np.any(attraction<=0) or not np.isfinite(attraction).all():
        raise ValueError('bulk pairing calibration failed')
    print('Fitted onsite attractions (eV):',attraction,flush=True)
    clean_result = solve_gap(bulk,attraction,np.full(model.nw,1.2*gap),t,tolerance_ev=args.tolerance_ev)
    clean_error = float(np.max(np.abs(clean_result['delta_ev']-gap)))
    if clean_error>1e-7 and not args.fixed_calibration:
        raise ValueError(f'clean self-consistent gap failed calibration: {clean_error:g} eV')
    print(f'Homogeneous SCF max difference from target gap: {clean_error:g} eV',flush=True)
    # Check the static model's temperature dependence and linearized Tc,
    # rather than asserting an exact ME mapping from one-temperature data.
    validation = []
    for row in me['temperatures']:
        temperature = row['temperature_k']
        solution = solve_gap(bulk,attraction,np.full(model.nw,gap),temperature,tolerance_ev=args.tolerance_ev)
        validation.append(dict(temperature_k=temperature,me_gap_ev=row['delta0_ev'],
            static_orbital_gaps_ev=solution['delta_ev'].tolist(),residual_ev=solution['residual_ev']))
    def eigenvalue(temperature):
        epsilon = 1e-7
        response = np.column_stack([bulk.evaluate(np.eye(model.nw)[a]*epsilon,temperature)['anomalous']/epsilon
                                    for a in range(model.nw)])
        return float(np.linalg.eigvals(attraction[:,None]*response).real.max())
    bounds = (2.,max(40.,2*me['tc_me_k']))
    rho_low,rho_high = eigenvalue(bounds[0]),eigenvalue(bounds[1])
    tc = float(brentq(lambda temperature:eigenvalue(temperature)-1,*bounds,xtol=.01)) if rho_low>1>rho_high else None
    write_json(out/'pairing-calibration.json',dict(**info,attraction_ev=attraction.tolist(),
        clean_gap_error_ev=clean_error,bulk_temperature_validation=validation,
        static_tc_k=tc,me_tc_k=me['tc_me_k'],
        pairing_channels='independent onsite intraorbital singlets; six fitted positive U_a',
        window='normal-state spectral projector |(epsilon-mu)/Z0| < cutoff; all projected BdG states retained',
        normal_sector='fixed bulk Z0 and mu; no Hartree or vacancy relaxation'))
    print(f'Static model Tc={tc} K; source ME Tc={me["tc_me_k"]:.6g} K',flush=True)
    if args.calibration_only:
        return
    all_cells = np.indices(shape).reshape(3,-1).T
    sites = np.arange(1,len(all_cells))
    basis_file = out/'vacancy-basis.npz'
    if args.resume and basis_file.exists():
        vacancy = ProjectedSpectrum.load_basis(basis_file,model.backend)
    else:
        vacancy,sites = vacancy_spectrum(model,shape,twist_mesh,z0,cutoff,t,
            chunk=args.normal_batch_size,progress=print_progress('Vacancy normal basis'))
        vacancy.save_basis(basis_file)
    # Validate the folding convention by recovering the pristine folded bands.
    check_twist = np.array([[.137,.219,.311]])
    pristine,_ = supercell_matrix(model,shape,check_twist,(),z0)
    folded_k = (check_twist[:,None]+all_cells)/shape
    expected = host(model.electrons(folded_k.reshape(-1,3))[0]).ravel()
    expected = np.sort((expected-mu)/z0)
    folding_error = float(np.max(np.abs(np.sort(host(model.xp.linalg.eigvalsh(pristine))[0])-expected)))
    if folding_error>1e-8:
        raise ValueError(f'supercell hopping folding check failed: {folding_error:g} eV')
    site_u = np.tile(attraction,len(sites))
    solutions = []
    for initial_scale in (.5,1.,1.5):
        def progress(iteration,residual,delta):
            if iteration==1 or iteration%25==0 or residual<args.tolerance_ev:
                print(f'Vacancy start {initial_scale:g}, iter {iteration}: residual {residual:.3g} eV; '
                      f'mean gap {delta.mean()*1000:.6g} meV',flush=True)
        solution = solve_gap(vacancy,site_u,np.full(vacancy.norbitals,initial_scale*gap),t,
            tolerance_ev=args.tolerance_ev,mixing=args.mixing,progress=progress)
        np.savez_compressed(out/f'solution-start-{initial_scale:g}.npz',**{
            k:v for k,v in solution.items() if k!='spectra'})
        solutions.append(solution)
    selected = min(solutions,key=lambda s:s['grand_potential_difference_ev'])
    delta = selected['delta_ev'].reshape(-1,model.nw)
    field_difference = max(float(np.max(np.abs(s['delta_ev']-selected['delta_ev']))) for s in solutions)
    energy = np.linspace(-.008,.008,1601)
    final = vacancy.evaluate(selected['delta_ev'],t,observables=True,ldos_grid=energy)
    next_field = site_u*final['anomalous']
    next_observables = vacancy.evaluate(next_field,t,observables=True)
    number_stability = float(abs(next_observables['density'].sum()-final['density'].sum()))
    reference = bulk.evaluate(np.full(model.nw,gap),t,observables=True,ldos_grid=energy)
    density = final['density'].reshape(-1,model.nw)
    occupation_change = density-reference['density'][None,:]
    # Read the BCC lattice from the same path utility used for bulk bands.
    import importlib.util
    spec = importlib.util.spec_from_file_location('nb_plot_bands',HERE/'plot_bands.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    lattice = np.array(module.path_definition(args.lattice)['primitive_lattice_columns_bohr'])
    radius = distances(all_cells[sites],lattice,shape)
    ldos = final['ldos_qp_per_spin_ev'].reshape(len(sites),model.nw,-1).sum(axis=1)/z0
    bulk_ldos = reference['ldos_qp_per_spin_ev'].sum(axis=0)/z0
    np.savez_compressed(out/'vacancy-T2.npz',cells=all_cells[sites],site_indices=sites,
        removed_cell=all_cells[0],removed_spatial_orbitals=np.arange(model.nw),
        onsite_delta_ev=delta,attraction_ev=attraction,anomalous_density=final['anomalous'].reshape(delta.shape),
        model_occupation=density,model_occupation_change=occupation_change,
        distance_angstrom=radius,ldos_energy_ev=energy,ldos_electron_coherent_per_spin_ev=ldos,
        bulk_ldos_electron_coherent_per_spin_ev=bulk_ldos,bulk_gap_ev=gap,z0=z0,mu_ev=mu,
        temperature_k=t,supercell=shape,twist_mesh=twist_mesh,cutoff_ev=cutoff,
        **final['spectra'])
    np.savetxt(out/'site-profile.dat',np.column_stack((sites,all_cells[sites],radius,delta*1000,
        density.sum(axis=1),occupation_change.sum(axis=1))),
        header='site r0 r1 r2 distance_Angstrom gaps_meV[s dz2 dxy dxz dx2-y2 dyz] occupation delta_occupation')
    nearest = np.isclose(radius,radius.min(),atol=1e-6)
    summary = dict(**info,backend=model.backend.info,removed_site=[0,0,0],removed_spatial_orbitals=model.nw,
        remaining_sites=len(sites),remaining_spatial_orbitals=vacancy.norbitals,
        spin_reduced_unprojected_dimension=2*vacancy.norbitals,
        max_projected_dimension=2*max(vacancy.groups),twist_points=vacancy.total_points,
        vacancy_concentration=1/int(np.prod(shape)),attraction_ev=attraction.tolist(),
        final_gap_residual_ev=float(np.max(np.abs(site_u*final['anomalous']-selected['delta_ev']))),
        independent_start_field_difference_ev=field_difference,pristine_folding_error_ev=folding_error,
        minimum_positive_bdg_energy_ev=final['minimum_energy_ev'],particle_hole_error_ev=final['particle_hole_error_ev'],
        grand_potential_difference_ev=selected['grand_potential_difference_ev'],
        gap_min_ev=float(delta.min()),gap_max_ev=float(delta.max()),gap_mean_ev=float(delta.mean()),
        nearest_site_average_gap_ev=float(delta[nearest].mean()),
        model_occupation_change_total=float(occupation_change.sum()),
        model_number_change_on_final_update=number_stability,
        iterations_by_start=[int(s['iterations']) for s in solutions],
        static_tc_k=tc,me_tc_k=me['tc_me_k'],elapsed_seconds=time.perf_counter()-started,
        limitations=['periodic vacancy array; supercell size convergence not established',
            'bulk-calibrated static onsite model; not defect-resolved dynamical Eliashberg',
            'fixed bulk chemical potential and residue; no Hartree, ionic relaxation or phonon changes',
            'occupation is canonical model occupation, not an ab initio defect charge'])
    write_json(out/'summary.json',summary)
    figures(out,all_cells[sites],radius,delta,gap,occupation_change.sum(axis=1),energy,ldos,bulk_ldos)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for scale,solution in zip((.5,1.,1.5),solutions):
        ax.semilogy(solution['history'][:,0],solution['history'][:,1],label=f'initial Δ = {scale:g} × bulk')
    ax.axhline(args.tolerance_ev,color='0.5',linestyle='--',linewidth=.8)
    ax.set_xlabel('Self-consistency iteration');ax.set_ylabel('Maximum gap residual (eV)')
    ax.legend(frameon=False);fig.savefig(out/'convergence.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
