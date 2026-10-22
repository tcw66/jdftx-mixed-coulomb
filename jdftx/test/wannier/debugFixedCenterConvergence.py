"""Diagnose fixed-center convergence on small graphene plus vacuum projections.

This deliberately difficult coarse-mesh case may fail due to ill-conditioning.
The JSON report distinguishes convergence, iteration limits and solver failure.
"""
from pathlib import Path
import argparse,json,re
import os,subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('build',type=Path)
p.add_argument('--work',type=Path,required=True)
p.add_argument('--mesh',type=int,default=2)
p.add_argument('--iterations',type=int,default=150)
a=p.parse_args()
if a.mesh<1 or a.iterations<0:p.error('mesh must be positive and iterations nonnegative')
b=a.build.resolve();r=a.work.resolve();r.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
common=f'''ion-species {b}/pseudopotentials/SG15/C_ONCV_PBE.upf
lattice Hexagonal 4.651 24
coords-type lattice
ion C 0 0 0 0
ion C 0.3333333333333333 -0.3333333333333333 0 0
elec-cutoff 8
elec-n-bands 40
kpoint-folding {a.mesh} {a.mesh} 1
'''
(r/'scf.in').write_text(common+'dump-name scf.$VAR\ndump End State BandEigs\nelectronic-minimize nIterations 100 energyDiffThreshold 1e-10\n')
def run(exe,name):
 p=subprocess.run([str(b/exe),'-i',name+'.in','-o',name+'.out'],cwd=r,env=env,stdout=subprocess.DEVNULL)
 print(name,p.returncode,flush=True)
 return p.returncode
if run('jdftx','scf'):raise RuntimeError('SCF failed')
summary={}
for m,s in [('FiniteDifference','MatrixFree'),('TruncatedDC','MatrixFree'),('FiniteDifference','Explicit')]:
 name=m+'-'+s
 text=common+f'''wannier-initial-state scf.$VAR
wannier-dump-name {name}.$VAR
wannier localizationMeasure {m} fixCenters yes innerWindow -1 0.1 outerWindow -1 1.2 saveWfns no loadRotations no
wannier-center-constraint-solver {s}
wannier-minimize nIterations {a.iterations} energyDiffThreshold 0 knormThreshold 1e-6 fdTest yes
'''
 text+=''.join(f'wannier-center C {i} {o}\n' for i in (1,2) for o in ['s','px','py','pz'])
 for j in range(6):
  x,y=(0,0) if j%2==0 else (-1/3,1/3)
  text+=f'wannier-center Gaussian {x} {y} {(j+.5)/6} 1\n'
 (r/(name+'.in')).write_text(text)
 status=run('wannier',name)
 out=(r/(name+'.out')).read_text()
 iterations=re.findall(r'WannierMinimize: Iter:\s*(\d+).*?Omega:\s*(\S+).*?\|grad\|_K:\s*(\S+).*?centerError:\s*(\S+)',out)
 diagnostics=re.findall(r'stationarity rawNorm (\S+); tangentNorm (\S+); tangentResidual (\S+); phaseMargin (\S+); minOverlap (\S+)',out)
 summary[name]=dict(exit_code=status,converged='WannierMinimize: Converged' in out,last_iteration=iterations[-1] if iterations else None,last_diagnostics=diagnostics[-1] if diagnostics else None)
 (r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(summary[name],flush=True)
