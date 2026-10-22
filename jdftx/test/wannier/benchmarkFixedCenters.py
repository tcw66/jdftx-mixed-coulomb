#!/usr/bin/env python3
"""Compare explicit and matrix-free hard centers on a 32-orbital vacuum problem.

Reports persistent derivative storage per rank (not total GPU/process memory),
matched final objectives, maximum center errors and measured solver times.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('build',type=Path)
p.add_argument('--suffix',default='')
p.add_argument('--np',type=int,default=1)
p.add_argument('--work',type=Path)
a=p.parse_args();build=a.build.resolve()
tmp=None
if a.work:
    root=a.work.resolve();root.mkdir(parents=True,exist_ok=False)
else:
    tmp=tempfile.TemporaryDirectory(prefix='center-benchmark-');root=Path(tmp.name)
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
launcher=['mpirun','-np',str(a.np),'--bind-to','none'] if a.np>1 else []
common='lattice Cubic 12\ncoords-type lattice\nspintype no-spin\nelec-cutoff 5\nfftbox 32 32 32\nelec-n-bands 96\nkpoint-folding 2 2 2\n'
(root/'common.in').write_text(common)
np.zeros((32,32,32)).tofile(root/'zero.Vscloc')
(root/'scf.in').write_text('include common.in\nfix-electron-potential zero.$VAR\nelectronic-minimize nIterations 200 energyDiffThreshold 1e-11\ndump-name scf.$VAR\ndump End State BandEigs Fillings\n')
def run(exe,name):
    start=time.monotonic()
    result=subprocess.run(launcher+[str(build/(exe+a.suffix)),'-c','2','-i',name+'.in','-o',name+'.out'],cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    out=(root/(name+'.out')).read_text()
    if result.returncode:raise RuntimeError(name+' failed\n'+out[-5000:]+'\n'+result.stderr)
    return out,time.monotonic()-start
run('jdftx','scf')
fcc=np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
positions=np.array([(fcc+[i,j,k])/2 for i in (0,1) for j in (0,1) for k in (0,1)]).reshape(-1,3)
positions+=.002*np.random.default_rng(23).normal(size=positions.shape)
summary={}
for measure in ('FiniteDifference','TruncatedDC'):
    for solver in ('Explicit','MatrixFree'):
        name=measure+'-'+solver
        text=f'''include common.in
wannier-initial-state scf.$VAR
wannier-dump-name {name}.$VAR
wannier localizationMeasure {measure} fixCenters yes innerWindow -0.000001 0.05 outerWindow -0.000001 1.5 saveWfns no loadRotations no
wannier-center-constraint-solver {solver}
wannier-minimize nIterations 0 energyDiffThreshold 1e-8 knormThreshold 1e-7
'''
        text+=''.join('wannier-center-vacuum Gaussian '+' '.join(map(str,row))+' 1\n' for row in positions)
        (root/(name+'.in')).write_text(text)
        out,wall=run('wannier',name)
        match=re.search(r'peak persistent derivative storage (\S+) MiB/rank; prepare (\S+) s; normal solves (\S+) s',out)
        assert match,out[-4000:]
        storage,prepare,solve=map(float,match.groups())
        objective=float(re.findall(r'WannierMinimize: Iter:.*?Omega:\s*(\S+)',out)[-1])
        centers=np.loadtxt(root/(name+'.mlwfCenters'),ndmin=2)
        error=float(np.max(np.linalg.norm(centers[:,:3]-12*positions,axis=1)))
        assert error<1e-8,(name,error)
        invariants=re.search(r'orthonormality error (\S+); inner-window inclusion error (\S+)',out)
        assert invariants and max(map(float,invariants.groups()))<1e-8
        summary[name]=dict(derivative_MiB_per_rank=storage,prepare_seconds=prepare,normal_seconds=solve,wall_seconds=wall,objective=objective,center_error_bohr=error)
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(name,summary[name],flush=True)
    explicit=summary[measure+'-Explicit'];mf=summary[measure+'-MatrixFree']
    assert abs(explicit['objective']-mf['objective'])<1e-5,(explicit,mf)
    assert mf['derivative_MiB_per_rank']<explicit['derivative_MiB_per_rank']/2
print('Matched benchmark passed:',root,flush=True)
