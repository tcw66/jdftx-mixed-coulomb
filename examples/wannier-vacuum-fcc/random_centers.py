#!/usr/bin/env python3
"""Matched uniform random-cell Gaussian starts for MV and TruncatedDC."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
from perturb_centers import fit_fcc, SITES

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[23,71,181])
    p.add_argument('--build',type=Path,default=ROOT.parent.parent/'build-gpu')
    p.add_argument('--suffix',default='_gpu')
    p.add_argument('--np',type=int,default=2)
    args=p.parse_args()
    out=ROOT/'results/random-cell-starts'
    if out.exists():raise SystemExit(f'{out} exists; move it aside before repeating.')
    out.mkdir();state=out/'state';state.mkdir()
    base=ROOT/'results/measure-comparison/TruncatedDC/results'
    for f in ('scf.wfns','scf.fillings','scf.eigenvals','scf.out','pseudo.in','zero.Vscloc'):
        shutil.copy2(base/f,state/f)
    for f in ('common.in','model.in'):
        shutil.copy2(ROOT/f,state/f)
    windows=(base/'windows.in').read_text()
    windows=re.sub(r'loadRotations\s+\S+','loadRotations no',windows)
    windows=re.sub(r'saveWfnsRealSpace\s+\S+','saveWfnsRealSpace no',windows)
    a=float(re.search(r'^lattice Cubic\s+(\S+)',(state/'model.in').read_text(),re.M).group(1))
    baseline=json.loads((ROOT/'results/measure-comparison/summary.json').read_text())
    summary=dict(protocol=dict(distribution='Four independent uniform fractional positions in [0,1)^3; no minimum separation, no centering or symmetry constraint.',seeds=args.seeds,lattice_bohr=a,sigma_bohr=1.,cutoff_Ha=30,mesh=[8,8,8],max_iterations=300,energy_threshold=1e-8,gradient_threshold=1e-7),baseline={method:baseline[method]['final_objective_bohr2'] for method in ('MV','TruncatedDC')},runs=[])
    env=dict(os.environ,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS','2'),OPENBLAS_NUM_THREADS='1')
    launcher=['mpirun','-np',str(args.np)] if args.np>1 else []
    for seed in args.seeds:
        initial=np.random.default_rng(seed).random((4,3))
        seed_dir=out/f'seed-{seed}';seed_dir.mkdir()
        np.savetxt(seed_dir/'initial-centers.dat',initial,header='Uniform fractional Gaussian trial centers; sigma=1 bohr; identical for both methods')
        initial_fit=fit_fcc(initial,a)
        pairs=initial[:,None,:]-initial[None,:,:];pairs-=np.rint(pairs)
        min_sep=float(np.min(np.linalg.norm(pairs,axis=2)[np.triu_indices(4,1)])*a)
        for method,measure in [('MV','FiniteDifference'),('TruncatedDC','TruncatedDC')]:
            work=seed_dir/method;work.mkdir();result=work/'results';result.mkdir()
            for f in ('model.in','common.in'):shutil.copy2(state/f,work/f)
            for f in ('scf.wfns','scf.fillings','scf.eigenvals','scf.out','pseudo.in','zero.Vscloc'):(result/f).symlink_to(state/f)
            (result/'windows.in').write_text(re.sub(r'localizationMeasure\s+\S+',f'localizationMeasure {measure}',windows))
            text='include common.in\nwannier-initial-state results/scf.$VAR\nwannier-dump-name results/vac.$VAR\ninclude results/windows.in\n'
            text+=''.join('wannier-center-vacuum Gaussian '+' '.join(f'{x:.16g}' for x in row)+' 1\n' for row in initial)
            text+='wannier-minimize nIterations 300 energyDiffThreshold 1e-8 knormThreshold 1e-7\n'
            (work/'wannier.in').write_text(text)
            print('Running',seed,method,'initial FCC RMS',initial_fit['fcc_rms_bohr'],'minimum Gaussian separation',min_sep,flush=True)
            start=time.monotonic()
            subprocess.run(launcher+[str(args.build.resolve()/('wannier'+args.suffix)),'-i','wannier.in','-o','results/wannier.out'],cwd=work,env=env,check=True,stdout=subprocess.DEVNULL)
            elapsed=time.monotonic()-start
            log=(result/'wannier.out').read_text()
            trace=re.findall(r'WannierMinimize: Iter:\s*(\d+)\s+Omega:\s*(\S+).*?\|grad\|_K:\s*(\S+).*?t\[s\]:\s*(\S+)',log)
            final=np.loadtxt(result/'vac.mlwfCenters',ndmin=2)
            entry=dict(seed=seed,method=method,initial_fractional_centers=initial.tolist(),initial_fcc_fit=initial_fit,min_initial_separation_bohr=min_sep,final_fractional_centers=final[:,3:6].tolist(),spreads_bohr2=final[:,6].tolist(),converged='WannierMinimize: Converged' in log,iterations=int(trace[-1][0]),gradient_norm=float(trace[-1][2]),initial_objective=float(trace[0][1]),final_objective=float(trace[-1][1]),objective_change_from_baseline=float(trace[-1][1])-summary['baseline'][method],wall_seconds=elapsed,**fit_fcc(final[:,3:6],a))
            np.savetxt(result/'convergence.dat',np.array(trace,dtype=float),header='iteration objective gradient_norm elapsed_seconds')
            summary['runs'].append(entry)
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(seed,method,'iterations',entry['iterations'],'converged',entry['converged'],'objective',entry['final_objective'],'FCC RMS',entry['fcc_rms_bohr'],flush=True)
            report(out,summary)
    plot(out,summary)

def report(out,s):
    lines=['# Uniform random-cell Gaussian starts: MV versus TruncatedDC','',
        'Four trial centers are sampled independently and uniformly over [0,1)^3.',
        'No FCC offsets, common-translation removal, minimum separation, or',
        'position pinning is imposed. Both measures use the exact same coordinates',
        'for each seed and start fresh, without loading optimized rotations.',
        'The zero-electron, zero-potential calculation uses a=6 bohr, 30 Ha,',
        '8x8x8 k mesh, 40 bands, Gaussian sigma=1 bohr and unchanged windows.',
        'Stopping criteria and 300-iteration limit match the perturbation tests.',
        'FCC residuals allow translations, periodic images and site permutations.','',
        '| Seed | Measure | Initial FCC RMS residual (bohr) | Iterations | Converged | Final spread (bohr²) | Difference from own FCC baseline (bohr²) | Final FCC RMS residual (bohr) |',
        '|---|---|---:|---:|---|---:|---:|---:|']
    for r in s['runs']:
        lines.append(f'| {r["seed"]} | {r["method"]} | {r["initial_fcc_fit"]["fcc_rms_bohr"]:.6f} | {r["iterations"]} | {r["converged"]} | {r["final_objective"]:.9f} | {r["objective_change_from_baseline"]:+.3e} | {r["fcc_rms_bohr"]:.3e} |')
    expected=2*len(s['protocol']['seeds'])
    if len(s['runs'])==expected and all(r['converged'] and r['fcc_rms_bohr']<1e-3 and abs(r['objective_change_from_baseline'])<1e-5 for r in s['runs']):
        lines += ['', 'Both methods recovered a translated FCC array and their own symmetric',
            'FCC spread for every tested uniform random start. There was no detectable',
            'center symmetry breaking or alternative minimum in this sampled set.']
    lines += ['', '| Method | Converged starts | Iteration range | Maximum final FCC RMS residual (bohr) | Maximum absolute spread change from own FCC baseline (bohr²) |', '|---|---:|---:|---:|---:|']
    for method in ('MV','TruncatedDC'):
        rows=[r for r in s['runs'] if r['method']==method]
        if rows:
            lines.append(f'| {method} | {sum(r["converged"] for r in rows)}/{len(rows)} | {min(r["iterations"] for r in rows)}–{max(r["iterations"] for r in rows)} | {max(r["fcc_rms_bohr"] for r in rows):.3e} | {max(abs(r["objective_change_from_baseline"]) for r in rows):.3e} |')
    lines+=['','Raw spread values use different definitions for MV and TruncatedDC.',
        'Changes are relative to each measure\'s own symmetric FCC baseline.',
        'TruncatedDC center coordinates are phase-based estimates. These sampled',
        'starts do not establish the global optimum. Every final center set,',
        'rotation, native Hamiltonian, and convergence log is preserved.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

def plot(out,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    seeds=s['protocol']['seeds']
    for method,color,offset in [('MV','#2166ac',-.08),('TruncatedDC','#cf392f',.08)]:
        runs=[r for r in s['runs'] if r['method']==method]
        x=np.array([seeds.index(r['seed']) for r in runs])+offset
        axes[0].scatter(x,[r['iterations'] for r in runs],label=method,color=color)
        axes[1].scatter(x,[max(r['fcc_rms_bohr'],1e-10) for r in runs],label=method,color=color)
    axes[0].set_ylabel('Iterations to convergence / stopping')
    axes[1].set(ylabel='Final translation-aligned FCC RMS residual (bohr)',yscale='log')
    for ax in axes:ax.set_xticks(range(len(seeds)),seeds);ax.set_xlabel('Uniform random seed');ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'comparison.{ext}',dpi=180)
    plot_centers(out,s)


def plot_centers(out,s):
    import matplotlib.pyplot as plt
    seeds=s['protocol']['seeds'];a=s['protocol']['lattice_bohr']
    fig=plt.figure(figsize=(10,4*len(seeds)))
    for i,seed in enumerate(seeds):
        for j,method in enumerate(('MV','TruncatedDC')):
            r=next(x for x in s['runs'] if x['seed']==seed and x['method']==method)
            ax=fig.add_subplot(len(seeds),2,2*i+j+1,projection='3d')
            initial=np.array(r['initial_fractional_centers'])%1
            final=np.array(r['final_fractional_centers'])%1
            fitted=(SITES[list(r['target_indices'])]+np.array(r['translation_bohr'])/a)%1
            for dim in range(3):
                others=[d for d in range(3) if d!=dim]
                for u in (0,1):
                    for v in (0,1):
                        line=np.zeros((2,3));line[:,dim]=[0,1];line[:,others[0]]=u;line[:,others[1]]=v
                        ax.plot(*line.T,color='.8',lw=.6)
            ax.scatter(*initial.T,s=35,marker='x',color='#d47b16',label='Initial uniform positions')
            ax.scatter(*final.T,s=30,color='#2166ac' if method=='MV' else '#cf392f',label='Final centers')
            ax.scatter(*fitted.T,s=95,facecolors='none',edgecolors='k',label='Best translated FCC fit')
            ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xlabel='x/a',ylabel='y/a',zlabel='z/a',title=f'Seed {seed}: {method}')
            ax.set_box_aspect((1,1,1));ax.legend(fontsize=6,loc='upper left')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'centers.{ext}',dpi=180)

if __name__=='__main__':main()
