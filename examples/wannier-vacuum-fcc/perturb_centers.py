#!/usr/bin/env python3
"""Test MV or TruncatedDC robustness to perturbed FCC Gaussian trial positions."""
import argparse
from itertools import permutations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
SITES = np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])

def fit_fcc(centers,a):
    """Fit translations, permutations and periodic images; vacuum has no origin."""
    candidates=[]
    for order in permutations(range(4)):
        delta=centers-SITES[list(order)]
        for anchor in delta:
            shift=anchor.copy()
            for _ in range(8):
                residual=delta-shift
                residual-=np.rint(residual)
                shift+=residual.mean(axis=0)
            residual=delta-shift
            residual-=np.rint(residual)
            shift-=np.rint(shift)
            distances=np.linalg.norm(residual*a,axis=1)
            candidates.append((np.sqrt(np.mean(distances**2)),np.linalg.norm(shift*a),order,shift,distances,residual))
    rms=min(x[0] for x in candidates)
    best=min((x for x in candidates if x[0]<=rms+1e-10),key=lambda x:x[1])
    return dict(fcc_rms_bohr=float(best[0]),fcc_max_bohr=float(best[4].max()),
                translation_bohr=(a*best[3]).tolist(),target_indices=list(best[2]),residual_bohr=(a*best[5]).tolist())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measure',choices=['MV','TruncatedDC'],default='TruncatedDC')
    parser.add_argument('--build',type=Path,default=ROOT.parent.parent/'build-gpu')
    parser.add_argument('--suffix',default='_gpu')
    parser.add_argument('--np',type=int,default=2)
    parser.add_argument('--amplitudes',type=float,nargs='+',default=[.1,.5,1.])
    parser.add_argument('--seeds',type=int,nargs='+',default=[23,71,181])
    args=parser.parse_args()
    out=ROOT/'results'/('mv-perturbations' if args.measure=='MV' else 'truncateddc-perturbations')
    if out.exists():
        raise SystemExit(f'{out} exists; move it aside before repeating.')
    out.mkdir()
    state=out/'state';state.mkdir()
    baseline=ROOT/'results/measure-comparison'/args.measure/'results'
    model=(ROOT/'model.in').read_text()
    a=float(re.search(r'^lattice Cubic\s+(\S+)',model,re.M).group(1))
    for f in ('scf.wfns','scf.fillings','scf.eigenvals','scf.out','pseudo.in','zero.Vscloc'):
        shutil.copy2(baseline/f,state/f)
    for f in ('common.in','model.in'):
        shutil.copy2(ROOT/f,state/f)
    windows=(baseline/'windows.in').read_text()
    windows=re.sub(r'localizationMeasure\s+\S+','localizationMeasure '+('FiniteDifference' if args.measure=='MV' else 'TruncatedDC'),windows)
    windows=re.sub(r'loadRotations\s+\S+','loadRotations no',windows)
    windows=re.sub(r'saveWfnsRealSpace\s+\S+','saveWfnsRealSpace no',windows)
    (state/'windows.in').write_text(windows)
    base_centers=np.loadtxt(baseline/'vac.mlwfCenters',ndmin=2)
    reference=json.loads((ROOT/'results/measure-comparison/summary.json').read_text())[args.measure]
    summary={'protocol':dict(measure=args.measure,lattice_bohr=a,amplitudes_rms_bohr=args.amplitudes,seeds=args.seeds,
        displacement='Independent normal draws, remove common translation, normalize RMS vector length per center.',
        comparison='Final centers registered to FCC modulo translations, permutations and periodic images.',
        fixed_settings='30 Ha, 8x8x8 mesh, sigma=1 bohr, four functions, same windows and SCF snapshot.'),
        'baseline':dict(objective=reference['final_objective_bohr2'],iterations=reference['iterations'],**fit_fcc(base_centers[:,3:6],a)), 'runs':[]}
    jobs=[('rigid',None,np.tile([.3,-.2,.1],(4,1)))]
    for amp in args.amplitudes:
        for seed in args.seeds:
            rng=np.random.default_rng(seed)
            displacement=rng.normal(size=(4,3));displacement-=displacement.mean(axis=0)
            displacement*=amp/np.sqrt(np.mean(np.sum(displacement**2,axis=1)))
            jobs.append((f'rms-{amp:g}-seed-{seed}',seed,displacement))
    env=dict(os.environ,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS','2'),OPENBLAS_NUM_THREADS='1')
    launcher=['mpirun','-np',str(args.np)] if args.np>1 else []
    for name,seed,displacement in jobs:
        work=out/name;work.mkdir();results=work/'results';results.mkdir()
        for f in ('common.in','model.in'):
            shutil.copy2(state/f,work/f)
        for f in ('scf.wfns','scf.fillings','scf.eigenvals','scf.out','pseudo.in','zero.Vscloc','windows.in'):
            (results/f).symlink_to(state/f)
        centers=SITES+displacement/a
        text='include common.in\nwannier-initial-state results/scf.$VAR\nwannier-dump-name results/vac.$VAR\ninclude results/windows.in\n'
        text+=''.join('wannier-center-vacuum Gaussian '+ ' '.join(f'{x:.16g}' for x in row)+' 1\n' for row in centers)
        text+='wannier-minimize nIterations 300 energyDiffThreshold 1e-8 knormThreshold 1e-7\n'
        (work/'wannier.in').write_text(text)
        np.savetxt(work/'initial-centers.dat',centers,header='Fractional Gaussian trial centers; sigma=1 bohr')
        print('Running',name,flush=True)
        start=time.monotonic()
        subprocess.run(launcher+[str(args.build.resolve()/('wannier'+args.suffix)),'-i','wannier.in','-o','results/wannier.out'],cwd=work,env=env,check=True,stdout=subprocess.DEVNULL)
        elapsed=time.monotonic()-start
        log=(results/'wannier.out').read_text()
        trace=re.findall(r'WannierMinimize: Iter:\s*(\d+)\s+Omega:\s*(\S+).*?\|grad\|_K:\s*(\S+).*?t\[s\]:\s*(\S+)',log)
        final=np.loadtxt(results/'vac.mlwfCenters',ndmin=2)
        entry=dict(name=name,seed=seed,initial_rms_displacement_bohr=float(np.sqrt(np.mean(np.sum(displacement**2,axis=1)))),
            initial_fractional_centers=centers.tolist(),final_fractional_centers=final[:,3:6].tolist(),spreads_bohr2=final[:,6].tolist(),
            converged='WannierMinimize: Converged' in log,iterations=int(trace[-1][0]),gradient_norm=float(trace[-1][2]),wall_seconds=elapsed,
            initial_objective=float(trace[0][1]),final_objective=float(trace[-1][1]),objective_change_from_baseline=float(trace[-1][1])-summary['baseline']['objective'],
            initial_fcc_fit=fit_fcc(centers,a),**fit_fcc(final[:,3:6],a))
        np.savetxt(results/'convergence.dat',np.array(trace,dtype=float),header='iteration objective gradient_norm elapsed_seconds')
        summary['runs'].append(entry)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(name, 'iterations',entry['iterations'],'objective',entry['final_objective'],'FCC RMS',entry['fcc_rms_bohr'],'converged',entry['converged'],flush=True)
    plot(out,summary)
    report(out,summary)

def report(out,summary):
    measure=summary['protocol'].get('measure','TruncatedDC')
    lines=[f'# {measure}: perturbing the initial FCC Gaussian positions','',
        'All runs start fresh from perturbed Gaussian projections at 30 Ha.',
        'Widths, energy windows, four-function model and 8x8x8 mesh are fixed.',
        'Random perturbations have zero mean and the listed RMS vector length',
        'per center. Seeds are independent. Centers are compared to an FCC array',
        'after fitting a common translation, a permutation and periodic images.',
        'This avoids treating homogeneous-vacuum translation as symmetry breaking.','',
        '| Start | Initial RMS kick (bohr) | Iterations | Converged | Final objective (bohr²) | Difference from FCC baseline | Final FCC RMS residual (bohr) |',
        '|---|---:|---:|---|---:|---:|---:|']
    for r in summary['runs']:
        lines.append(f'| {r["name"]} | {r["initial_rms_displacement_bohr"]:.4f} | {r["iterations"]} | {r["converged"]} | {r["final_objective"]:.9f} | {r["objective_change_from_baseline"]:+.3e} | {r["fcc_rms_bohr"]:.3e} |')
    random=[r for r in summary['runs'] if r['seed'] is not None]
    lines += ['', '| Initial RMS kick (bohr) | Iteration range | Max final FCC RMS residual (bohr) | Max absolute objective change (bohr²) |', '|---|---:|---:|---:|']
    for amp in summary['protocol']['amplitudes_rms_bohr']:
        group=[r for r in random if abs(r['initial_rms_displacement_bohr']-amp)<1e-9]
        if group:
            lines.append(f'| {amp:g} | {min(r["iterations"] for r in group)}–{max(r["iterations"] for r in group)} | {max(r["fcc_rms_bohr"] for r in group):.3e} | {max(abs(r["objective_change_from_baseline"]) for r in group):.3e} |')
    if random and all(r['converged'] and r['fcc_rms_bohr']<1e-3 and abs(r['objective_change_from_baseline'])<1e-5 for r in random):
        lines += ['', 'All tested random starts returned to the FCC arrangement (up to global', 'translation and periodic images) and the same spread within numerical tolerance.', 'The perturbed starts required more optimization than the symmetric baseline.']
    lines+=['',f'Unperturbed baseline objective: {summary["baseline"]["objective"]:.12f} bohr².',
        ('TruncatedDC centers are phase-based estimates. Spread changes and center' if measure=='TruncatedDC' else 'MV centers use the existing finite-difference definition. Spread changes and center'),
        'distortions test different aspects of the final orbitals. This finite set',
        'of starts cannot establish a global optimum. Native rotations and centers',
        'are saved in each run directory. No restart rotations were loaded.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

def plot(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for r in summary['runs']:
        if r['name']=='rigid':continue
        amp=r['initial_rms_displacement_bohr']
        axes[0].scatter(amp,r['objective_change_from_baseline'],label=r['name'])
        axes[1].scatter(amp,max(r['fcc_rms_bohr'],1e-10))
    axes[0].axhline(0,color='k',lw=.7)
    axes[0].set(xlabel='Initial RMS displacement per center (bohr)',ylabel='Final '+summary['protocol'].get('measure','TruncatedDC')+' spread minus FCC baseline (bohr²)')
    axes[1].set(xlabel='Initial RMS displacement per center (bohr)',ylabel='Final translation-aligned FCC RMS residual (bohr)',yscale='log')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'perturbation-summary.{suffix}',dpi=180)
    plot_centers(out,summary)


def plot_centers(out,summary):
    import matplotlib.pyplot as plt
    a=summary['protocol']['lattice_bohr']
    seed=summary['protocol']['seeds'][0]
    selected=[r for r in summary['runs'] if r['seed']==seed]
    fig=plt.figure(figsize=(5*len(selected),4.5))
    for n,r in enumerate(selected):
        ax=fig.add_subplot(1,len(selected),n+1,projection='3d')
        initial=np.array(r['initial_fractional_centers'])
        final=np.array(r['final_fractional_centers'])
        # Unwrap each final position near its original trial site for display.
        final=SITES+(final-SITES)-np.rint(final-SITES)
        for dim in range(3):
            others=[d for d in range(3) if d!=dim]
            for u in (0,1):
                for v in (0,1):
                    line=np.zeros((2,3));line[:,dim]=[0,1];line[:,others[0]]=u;line[:,others[1]]=v
                    ax.plot(*line.T,color='.8',lw=.6)
        ax.scatter(*SITES.T,s=110,facecolors='none',edgecolors='k',label='FCC sites')
        ax.scatter(*initial.T,s=45,color='#d47b16',marker='x',label='Initial Gaussians')
        ax.scatter(*final.T,s=25,color='#2166ac',label='Final centers')
        for i,f in zip(initial,final):
            ax.plot(*np.array([i,f]).T,color='.45',ls=':',lw=.8)
        ax.set(xlabel='x/a',ylabel='y/a',zlabel='z/a',title=f'RMS kick {r["initial_rms_displacement_bohr"]:g} bohr (seed {seed})')
        ax.set_box_aspect((1,1,1));ax.legend(fontsize=7,loc='upper right')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'center-relaxation.{suffix}',dpi=180)

if __name__=='__main__':main()
