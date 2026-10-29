#!/usr/bin/env python3
"""Hard-center integration regression with a frozen inner window and free outer bands."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('build',type=Path)
p.add_argument('--suffix',default='')
p.add_argument('--np',type=int,default=1)
p.add_argument('--work',type=Path)
p.add_argument('--method',choices=['CG','LBFGS','SQP'],default='CG')
p.add_argument('--solver',choices=['MatrixFree','Explicit','Both'],default='Both')
a=p.parse_args();build=a.build.resolve()
temp=None
if a.work:
    root=a.work.resolve();root.mkdir(parents=True,exist_ok=False)
else:
    temp=tempfile.TemporaryDirectory(prefix='fixed-centers-');root=Path(temp.name)
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
launcher=['mpirun','-np',str(a.np)] if a.np>1 else []
np.zeros((24,24,24)).tofile(root/'zero.Vscloc')
(root/'common.in').write_text('lattice Cubic 6\ncoords-type lattice\nspintype no-spin\nelec-cutoff 6\nfftbox 24 24 24\nelec-n-bands 24\nkpoint-folding 2 2 2\n')
(root/'scf.in').write_text('include common.in\nfix-electron-potential zero.$VAR\nelectronic-minimize nIterations 100 energyDiffThreshold 1e-11\ndump-name scf.$VAR\ndump End State BandEigs Fillings\n')
def run(exe,name):
    result=subprocess.run(launcher+[str(build/(exe+a.suffix)),'-i',name+'.in','-o',name+'.out'],cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    text=(root/(name+'.out')).read_text()
    if result.returncode:raise RuntimeError(name+' failed in '+str(root)+'\n'+text[-4500:]+'\n'+result.stderr)
    return text
run('jdftx','scf')
targets=np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0]])
trials=targets+.02*np.random.default_rng(23).normal(size=(4,3))
(root/'cartesian.in').write_text((root/'common.in').read_text().replace('coords-type lattice','coords-type cartesian'))
solvers=['MatrixFree','Explicit'] if a.solver=='Both' else [a.solver]
for measure,partial,solver in [(m,p,s) for m,p in [('FiniteDifference',False),('TruncatedDC',False),('TruncatedDC',True)] for s in solvers]:
    name=measure.lower()+('-partial-cartesian' if partial else '')+('-explicit' if solver=='Explicit' else '')
    target_cart=6*(targets if partial else trials).copy()
    if partial: target_cart[0]+=[.01,.02,.03]
    selected=[0] if partial else range(4)
    common='cartesian.in' if partial else 'common.in'
    text=f'include {common}\nwannier-initial-state scf.$VAR\nwannier-dump-name {name}.$VAR\nwannier localizationMeasure {measure} outerWindow -0.000001 1.5 innerWindow -0.000001 0.3 saveWfns no loadRotations no\n'
    if not partial:text=text.replace('localizationMeasure '+measure,'localizationMeasure '+measure+' fixCenters yes')
    text+=''.join('wannier-center-vacuum Gaussian '+' '.join(map(str,row))+' 1\n' for row in (6*trials if partial else trials))
    if partial:text+=''.join(f'wannier-center-constraint {i} '+' '.join(map(str,target_cart[i]))+'\n' for i in selected)
    text+=f'wannier-center-constraint-solver {solver}\nwannier-center-constraint-method {a.method}\n'
    text+='wannier-minimize nIterations 500 energyDiffThreshold 0 knormThreshold 1e-6 fdTest yes\n'
    (root/(name+'.in')).write_text(text)
    out=run('wannier',name)
    centers=np.loadtxt(root/(name+'.mlwfCenters'),ndmin=2)
    error=np.max(np.linalg.norm(centers[list(selected),:3]-target_cart[list(selected)],axis=1))
    assert error<1e-8,(name,error)
    assert 'fdTest center Jacobian error' in out
    assert 'fdTest feasible derivative error' in out
    if a.method=='SQP':
        assert 'fdTest SQP Hessian symmetry error' in out
        assert 'fdTest global phase projection' in out
        assert 'SQP Hessian products' in out
    if a.method=='LBFGS':
        assert 'fdTest global phase projection' in out
        assert 'LBFGS curvature pairs' in out
    assert 'WannierMinimize: Converged' in out,out[-4000:]
    gradients=[float(x) for x in re.findall(r'\|grad\|_K:\s*(\S+)',out)]
    assert gradients[-1]<1e-6,gradients[-1]
    assert 'inner-window inclusion error' in out
    # Independently inspect the exported rectangular rotations. In vacuum the
    # source eigenvalues are the sorted |k+G|^2/2, so frozen-band rows are known.
    header=(root/(name+'.mlwfState')).read_text().splitlines()[0].split()
    points=np.array(list(map(float,header[17:]))).reshape(-1,3)
    nb=int(header[2]);nw=4
    U=np.fromfile(root/(name+'.mlwfU'),dtype=np.complex128,count=len(points)*nb*nw).reshape(len(points),nw,nb).swapaxes(1,2)
    grid=np.stack(np.meshgrid(np.arange(-4,5),np.arange(-4,5),np.arange(-4,5),indexing='ij'),axis=-1).reshape(-1,3)
    energies=np.sort(.5*(2*np.pi/6)**2*np.sum((points[:,None,:]+grid[None,:,:])**2,axis=2),axis=1)[:,:nb]
    inner_error=0.;orth_error=0.;counts=[]
    for k,W in enumerate(U):
        orth_error=max(orth_error,np.linalg.norm(W.conj().T@W-np.eye(nw)))
        frozen=np.flatnonzero(energies[k]<.3);counts.append(len(frozen))
        F=np.eye(nb)[:,frozen]
        inner_error=max(inner_error,np.linalg.norm(F-W@(W.conj().T@F)))
    assert min(counts)==0 and max(counts)==4,counts
    assert inner_error<1e-8 and orth_error<1e-8,(inner_error,orth_error)
    residuals=[float(x) for x in re.findall(r'centerError:\s*(\S+)',out)]
    assert residuals and max(residuals)<1e-8,residuals
    values=[float(x) for x in re.findall(r'WannierMinimize: Iter:.*?Omega:\s*(\S+)',out)]
    assert values[-1]<values[0]-1e-5,values
    print(name,'maximum target error',error,'spread',values[0],'->',values[-1],flush=True)
# Reject ambiguous duplicate targets during parsing, before optimization starts.
(root/'duplicate.in').write_text(text+'wannier-center-constraint 0 0 0 0\n')
result=subprocess.run(launcher+[str(build/('wannier'+a.suffix)),'-i','duplicate.in','-o','duplicate.out'],cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
assert result.returncode and 'Duplicate fixed-center target' in (root/'duplicate.out').read_text()+result.stderr
print('Hard-center regression passed in',root,flush=True)
