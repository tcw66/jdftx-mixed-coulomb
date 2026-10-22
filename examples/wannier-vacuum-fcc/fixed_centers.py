#!/usr/bin/env python3
"""Localize vacuum WFs with centers fixed at their random Gaussian positions."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT.parent.parent/'build-gpu')
    p.add_argument('--suffix',default='_gpu')
    p.add_argument('--np',type=int,default=2)
    p.add_argument('--seed',type=int,default=23)
    p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--out',type=Path,default=ROOT/'results/fixed-initial-centers')
    a=p.parse_args()
    if a.iterations<0:p.error('--iterations must be nonnegative')
    out=a.out.resolve()
    if out.exists():raise SystemExit(f'{out} exists; choose a new --out directory.')
    base=ROOT/'results/measure-comparison/TruncatedDC'
    for name in ('model.in','common.in'):
        if not (base/name).exists() or (base/name).read_text()!=(ROOT/name).read_text():
            raise SystemExit('Run ./run.sh first to generate SCF states matching model.in/common.in.')
    state=base/'results'
    for name in ('scf.wfns','scf.eigenvals','scf.fillings'):
        if not (state/name).is_file():raise SystemExit('Missing SCF state; run ./run.sh first.')
    out.mkdir(parents=True)
    initial=np.random.default_rng(a.seed).random((4,3))
    np.savetxt(out/'initial-centers.dat',initial,header='Fractional trial Gaussian centers; sigma=1 bohr')
    launcher=['mpirun','-np',str(a.np)] if a.np>1 else []
    env=dict(os.environ,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS','2'),OPENBLAS_NUM_THREADS='1')
    summary={}
    for label,measure in [('MV','FiniteDifference'),('TruncatedDC','TruncatedDC')]:
        work=out/label;work.mkdir();(work/'results').mkdir()
        for name in ('model.in','common.in'):shutil.copy2(ROOT/name,work/name)
        for name in ('scf.wfns','scf.eigenvals','scf.fillings'):
            (work/'results'/name).symlink_to(state/name)
        text='include common.in\nwannier-initial-state results/scf.$VAR\nwannier-dump-name results/vac.$VAR\n'
        text+=f'wannier localizationMeasure {measure} fixCenters yes innerWindow -0.000001 0.3 outerWindow -0.000001 1.5 saveWfns no loadRotations no\n'
        text+=''.join('wannier-center-vacuum Gaussian '+' '.join(f'{x:.16g}' for x in row)+' 1\n' for row in initial)
        text+=f'wannier-center-constraint-tolerance 1e-8\nwannier-minimize nIterations {a.iterations} energyDiffThreshold 1e-8 knormThreshold 1e-7 fdTest yes\n'
        (work/'wannier.in').write_text(text)
        print('Running',label,'with random seed',a.seed,flush=True)
        subprocess.run(launcher+[str(a.build.resolve()/('wannier'+a.suffix)),'-i','wannier.in','-o','results/wannier.out'],cwd=work,env=env,check=True,stdout=subprocess.DEVNULL)
        log=(work/'results/wannier.out').read_text()
        trace=re.findall(r'WannierMinimize: Iter:\s*(\d+)\s+Omega:\s*(\S+).*?centerError:\s*(\S+)',log)
        if not trace:raise RuntimeError(f'Missing optimization trace in {work}')
        i,omega,error=trace[-1]
        summary[label]=dict(iterations=int(i),objective_bohr2=float(omega),max_center_error_bohr=float(error),converged='WannierMinimize: Converged' in log)
        centers=np.loadtxt(work/'results/vac.mlwfCenters',ndmin=2)
        actual_error=np.max(np.linalg.norm(centers[:,:3]-6*initial,axis=1))
        summary[label]['max_displacement_from_initial_bohr']=float(actual_error)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(label,summary[label],flush=True)
        if float(error)>1e-8 or actual_error>1e-8:raise RuntimeError('Fixed-center tolerance not satisfied')
    print('Results:',out)

if __name__=='__main__':main()
