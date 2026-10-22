#!/usr/bin/env python3
"""Run matched MV/TDC platinum localizations and compare their band errors.
Default: two MPI ranks with the repository's GPU build. Outputs stay separate.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parent

def cross_evaluate(output, build, suffix, ranks, env):
    """Evaluate each final gauge using both measures, without optimization."""
    launcher = ['mpirun','-np',str(ranks)] if ranks>1 else []
    scores = {}
    for source, target, measure in [('MV','TruncatedDC','TruncatedDC'),('TruncatedDC','MV','FiniteDifference')]:
        original = output/source
        work = output/(source+'_evaluated_as_'+target)
        work.mkdir()
        results = work/'results'
        results.mkdir()
        for filename in ('common.in','model.in','02-wannier.in'):
            text = (original/filename).read_text()
            if filename == '02-wannier.in':
                text = re.sub(r'nIterations\s+\d+', 'nIterations 0',text)
            (work/filename).write_text(text)
        for filename in ('scf.wfns','scf.fillings','scf.eigenvals','scf.out','pseudo.in'):
            (results/filename).symlink_to(output/'state'/filename)
        for filename in ('pt.mlwfU','pt.mlwfState','pt.mlwfM0'):
            shutil.copy2(original/'results'/filename,results/filename)
        text = (original/'results/windows.in').read_text()
        text = re.sub(r'localizationMeasure\s+\S+',f'localizationMeasure {measure}',text)
        text = re.sub(r'loadRotations\s+\S+','loadRotations yes',text)
        (results/'windows.in').write_text(text)
        subprocess.run(launcher+[str(build/('wannier'+suffix)),'-i','02-wannier.in','-o','results/wannier.out'],cwd=work,env=env,check=True,stdout=subprocess.DEVNULL)
        log = (results/'wannier.out').read_text()
        if 'Reading Wannier rotations' not in log and 'Reading rotations' not in log:
            # Fail loudly if the checkpoint was rejected; fresh initialization is invalid here.
            if 'Reading' not in log or 'mlwfU' not in log or 'incompatible' in log:
                raise ValueError('Could not verify loaded final rotations: '+str(work))
        scores[source] = {target:float(re.findall(r'^\s*Omega:\s*(\S+)',log,re.M)[-1])}
    (output/'cross-objectives.json').write_text(json.dumps(scores,indent=2)+'\n')
    return scores

def plot_comparison(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    ref=output/'MV'/'results'
    dft=np.loadtxt(ref/'dft-bands.dat')
    axes[0].plot(dft[:,0],dft[:,1:],color='.65',lw=.8)
    axes[0].plot([],[],color='.65',label='DFT')
    for name,color in [('MV','#2166ac'),('TruncatedDC','#cf392f')]:
        data=np.loadtxt(output/name/'results/wannier-bands.dat')
        axes[0].plot(data[:,0],data[:,1:],color=color,lw=1,ls='--')
        axes[0].plot([],[],color=color,ls='--',label=name)
        trace=np.loadtxt(output/name/'results/convergence.dat')
        # Each curve is measured relative to its own final objective.
        axes[1].semilogy(trace[:,0],np.maximum(trace[:,1]-trace[-1,1],1e-10),color=color,label=name)
    path=np.loadtxt(ref/'bandpath.dat')
    meta=json.loads((ref/'bandpath.json').read_text())
    axes[0].set_xticks(path[meta['ticks'],0],meta['labels'])
    axes[0].set(xlim=(path[0,0],path[-1,0]),ylim=(-12,12),ylabel=r'$E-E_F$ (eV)',title='Platinum band interpolation')
    axes[0].axhline(0,color='k',lw=.5)
    axes[1].set(xlabel='Localization iteration',ylabel=r'$\Omega-\Omega_{final}$ (bohr$^2$)',title='Each measure relative to its own final value')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(output/f'comparison.{extension}',dpi=180)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT.parent.parent/'build-gpu')
    parser.add_argument('--suffix', default='_gpu')
    parser.add_argument('--np',type=int,default=2)
    args = parser.parse_args()
    output = ROOT/'results'/'measure-comparison'
    if output.exists():
        raise SystemExit(f'{output} exists; move it aside to keep previous comparisons.')
    output.mkdir()
    state = output/'state'
    state.mkdir()
    for name in ('scf.wfns','scf.fillings','scf.eigenvals','scf.out','pseudo.in'):
        shutil.copy2(ROOT/'results'/name,state/name)
    for filename in ('common.in','model.in','02-wannier.in','export-potential.in','03-bandstruct.in'):
        shutil.copy2(ROOT/filename,state/filename)
    env = dict(os.environ,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS','2'),OPENBLAS_NUM_THREADS='1')
    launcher = ['mpirun','-np',str(args.np)] if args.np>1 else []
    build = args.build.resolve()
    report = {}
    windows = (ROOT/'results/windows.in').read_text()
    for name,measure in [('MV','FiniteDifference'),('TruncatedDC','TruncatedDC')]:
        work = output/name
        work.mkdir()
        results = work/'results'
        results.mkdir()
        for filename in ('scf.wfns','scf.fillings','scf.eigenvals','scf.out','pseudo.in'):
            (results/filename).symlink_to(state/filename)
        for filename in ('common.in','model.in','02-wannier.in','export-potential.in','03-bandstruct.in'):
            shutil.copy2(state/filename,work/filename)
        (results/'windows.in').write_text(re.sub(r'localizationMeasure\s+\S+',f'localizationMeasure {measure}',windows))
        # Force fresh initialization for each objective, preserving all other options.
        p=results/'windows.in'
        p.write_text(re.sub(r'loadRotations\s+\S+','loadRotations no',p.read_text()))
        def run(exe,input_file,log):
            subprocess.run(launcher+[str(build/(exe+args.suffix)),'-i',input_file,'-o','results/'+log],cwd=work,env=env,check=True,stdout=subprocess.DEVNULL)
        print(f'Running {name} on {args.np} ranks',flush=True)
        start = time.monotonic()
        run('wannier','02-wannier.in','wannier.out')
        wall = time.monotonic()-start
        text = (results/'wannier.out').read_text()
        trace = re.findall(r'WannierMinimize: Iter:\s*(\d+)\s+Omega:\s*(\S+).*?t\[s\]:\s*(\S+)',text)
        report[name] = dict(wall_seconds=wall,iterations=int(trace[-1][0]),initial_objective_bohr2=float(trace[0][1]),final_objective_bohr2=float(trace[-1][1]),converged='WannierMinimize: Converged' in text)
        # Rebuild the direct DFT reference once from this exact SCF snapshot.
        spec=importlib.util.spec_from_file_location('bands',ROOT/'compare_bands.py')
        bands=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bands)
        bands.ROOT=work
        bands.prepare(20)
        if name=='MV':
            run('jdftx','export-potential.in','export-potential.out')
            run('jdftx','03-bandstruct.in','bandstruct.out')
        else:
            for filename in ('bandstruct.eigenvals','bandstruct.out'):
                (results/filename).symlink_to(output/'MV'/'results'/filename)
        bands.compare()
        report[name].update(json.loads((results/'band-errors.json').read_text()))
        np.savetxt(results/'convergence.dat',np.array(trace,dtype=float),header='iteration objective_bohr2 elapsed_seconds')
        print(name,report[name],flush=True)
    cross_evaluate(output,build,args.suffix,args.np,env)
    plot_comparison(output)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Comparison saved to',output,flush=True)

if __name__=='__main__':
    main()
