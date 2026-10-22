#!/usr/bin/env python3
"""Hard centers for explicit and automatically added atomic trial projections."""
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
a=p.parse_args();build=a.build.resolve()
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
launcher=['mpirun','-np',str(a.np)] if a.np>1 else []
with tempfile.TemporaryDirectory(prefix='fixed-atomic-centers-') as tmp:
    root=Path(tmp)
    common=f'''ion-species {build}/pseudopotentials/SG15/Si_ONCV_PBE.upf
lattice Face-Centered Cubic 10.26
coords-type lattice
ion Si 0 0 0 1
ion Si 0.25 0.25 0.25 1
elec-cutoff 5
elec-n-bands 12
kpoint-folding 2 2 2
'''
    def run(exe,name):
        result=subprocess.run(launcher+[str(build/(exe+a.suffix)),'-i',name+'.in','-o',name+'.out'],cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        out=(root/(name+'.out')).read_text()
        if result.returncode:raise RuntimeError(name+' failed\n'+out[-5000:]+'\n'+result.stderr)
        return out
    (root/'scf.in').write_text(common+'dump-name scf.$VAR\ndump End State BandEigs\nelectronic-minimize nIterations 70\n')
    run('jdftx','scf')
    for measure in ('FiniteDifference','TruncatedDC'):
        for auto in (False,True):
            name=measure.lower()+('-auto' if auto else '-explicit')
            text=common+f'''wannier-initial-state scf.$VAR
wannier-dump-name {name}.$VAR
wannier localizationMeasure {measure} fixCenters yes addAtomicOrbitals {'yes' if auto else 'no'} innerWindow -100 -0.15 outerWindow -100 100 loadRotations no
'''
            if not auto:
                text+=''.join(f'wannier-center Si {atom} {orbital}\n' for atom in (1,2) for orbital in ('s','px','py','pz'))
            text+='wannier-minimize nIterations 20 energyDiffThreshold 1e-8 knormThreshold 1e-7 fdTest yes\n'
            (root/(name+'.in')).write_text(text)
            out=run('wannier',name)
            centers=np.loadtxt(root/(name+'.mlwfCenters'),ndmin=2)
            expected=np.repeat([[0.,0.,0.],[.25,.25,.25]],4,axis=0)
            # Cartesian displacement in the primitive FCC lattice.
            R=5.13*np.array([[0.,1.,1.],[1.,0.,1.],[1.,1.,0.]])
            error=np.max(np.linalg.norm((centers[:,3:6]-expected)@R.T,axis=1))
            assert error<1e-8,(name,error)
            residuals=[float(x) for x in re.findall(r'centerError:\s*(\S+)',out)]
            assert residuals and max(residuals)<=1e-8,residuals
            errors=re.search(r'orthonormality error (\S+); inner-window inclusion error (\S+)',out)
            assert errors and max(map(float,errors.groups()))<1e-8
            assert 'fdTest center Jacobian error' in out
            values=[float(x) for x in re.findall(r'WannierMinimize: Iter:.*?Omega:\s*(\S+)',out)]
            assert values[-1]<=values[0]+1e-9,values
            print(name,'max displacement from atomic trial positions',error,flush=True)
print('Fixed atomic projection centers passed',flush=True)
