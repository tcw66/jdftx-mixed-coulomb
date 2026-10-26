#!/usr/bin/env python3
"""Mixed hard/free Wannier centers, ordering, composite targets and conflicts."""
import argparse,json,os,re,subprocess,tempfile,itertools,importlib.util
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__);p.add_argument('build',type=Path);p.add_argument('--suffix',default='');p.add_argument('--np',type=int,default=1);p.add_argument('--work',type=Path);a=p.parse_args();build=a.build.resolve()
temp=None
if a.work:root=a.work.resolve();root.mkdir(parents=True,exist_ok=False)
else:temp=tempfile.TemporaryDirectory(prefix='selective-centers-');root=Path(temp.name)
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'};env.pop('PRB_FIXED_OPTIMIZER',None)
launch=['mpirun','-np',str(a.np)] if a.np>1 else []
common=f'''ion-species {build}/pseudopotentials/SG15/Si_ONCV_PBE.upf
lattice Face-Centered Cubic 10.26
coords-type lattice
symmetries none
ion Si 0 0 0 0
ion Si 0.25 0.25 0.25 0
elec-cutoff 5
elec-n-bands 12
kpoint-folding 2 2 2
'''
(root/'common.in').write_text(common)
def execute(exe,name,dry=False,fail=None):
    cmd=launch+[str(build/(exe+a.suffix)),'-c','1','-d']+(['-n'] if dry else [])+['-i',name+'.in','-o',name+'.out']
    result=subprocess.run(cmd,cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    text=(root/(name+'.out')).read_text()
    if fail:
        assert result.returncode and fail in text+result.stderr,(name,text[-3000:]);return text
    if result.returncode:raise RuntimeError(name+' failed\n'+text[-4000:]+result.stderr)
    return text
(root/'scf.in').write_text('include common.in\nconverge-empty-states yes\nelectronic-minimize nIterations 100 energyDiffThreshold 1e-10\ndump-name scf.$VAR\ndump End State BandEigs\n')
execute('jdftx','scf')
R=5.13*np.array([[0,1,1],[1,0,1],[1,1,0]],float)
trials=[f'Si {atom} {orb}' for atom in [1,2] for orb in ['s','px','py','pz']]
targets=np.repeat([[0.,0.,0.],[.25,.25,.25]],4,axis=0)
results={}
def run(name,selected,explicit=False,measure='FiniteDifference',all_fixed=False,composite=False,free_shift=False):
    text=f'''include common.in
wannier-initial-state scf.$VAR
wannier-dump-name {name}.$VAR
wannier localizationMeasure {measure} fixCenters {'yes' if all_fixed else 'no'} pinAtomicOrbitals no innerWindow -100 -0.15 outerWindow -100 100 loadRotations no
wannier-center-constraint-tolerance 1e-8
wannier-minimize nIterations 8 energyDiffThreshold 0 knormThreshold 1e-12 fdTest yes
'''
    expected=targets.copy()
    for i,t in enumerate(trials):
        if composite and i==0:
            t='Gaussian 0 0 0 1 s 1 Gaussian 0.01 0 0 1 s 2';expected[i]=[.008,0,0]
        if free_shift and i==4:t='Gaussian 0.27 0.25 0.25 1 s';expected[i]=[.27,.25,.25]
        cmd='wannier-center-fixed' if i in selected and not explicit else 'wannier-center'
        text+=cmd+' '+t+'\n'
    if explicit:
        for i in selected:text+=f'wannier-center-constraint {i} '+' '.join(str(x) for x in expected[i])+'\n'
    (root/(name+'.in')).write_text(text);log=execute('wannier',name)
    c=np.loadtxt(root/(name+'.mlwfCenters'));assert c.shape==(8,7)
    delta=c[:,3:6]-expected;delta-=np.rint(delta);dist=np.linalg.norm(delta@R.T,axis=1)
    fixed=list(range(8)) if all_fixed else selected
    if fixed:assert max(dist[fixed])<=1e-8,(name,dist)
    if not explicit and not all_fixed:
        for i in range(8):assert f'center {i}: '+('fixed at' if i in selected else 'free') in log
    hist=np.array(re.findall(r'WannierMinimize: Iter:\s*(\d+)\s+Omega:\s*([-+\d.eE]+).*?\|grad\|_K:\s*([-+\d.eE]+)',log),float)
    assert hist[-1,1]<=hist[0,1]+1e-9
    err=re.search(r'orthonormality error (\S+); inner-window inclusion error (\S+)',log)
    assert err and max(map(float,err.groups()))<1e-8
    results[name]=dict(selected=fixed,max_fixed_error_bohr=float(max(dist[fixed])) if fixed else None,free_offsets_bohr=[float(dist[i]) for i in range(8) if i not in fixed],final_spread_bohr2=float(hist[-1,1]))
    return c,hist,log

selected=[0,2,5,7]
c,h,log=run('interleaved',selected,free_shift=True)
ref,hh,_=run('indexed',selected,explicit=True,free_shift=True)
assert np.max(abs(h[:,1:]-hh[:,1:]))<1e-8
assert np.max(abs(c-ref))<1e-8
assert max(results['interleaved']['free_offsets_bohr'])>1e-3,'Free centers were inadvertently constrained'
assert 'wannier-center Gaussian 0.27 0.25 0.25' in log,'Echoed center list lost mixed-command ordering'
run('composite',[0,2,5,7],composite=True)
run('tdc',[0,2,5,7],measure='TruncatedDC')
run('all-fixed',[0,2,5,7],all_fixed=True)
base=(root/'interleaved.in').read_text()
(root/'conflict.in').write_text(base+'wannier-center-constraint 0 0 0 0\n')
execute('wannier','conflict',dry=True,fail='same Wannier function')
(root/'soft-mixed.in').write_text(base.replace('wannier-center Si 1 px','wannier-center-pinned Si 1 px'))
soft=execute('wannier','soft-mixed',dry=True)
assert 'center 1: soft-pinned' in soft
helper=Path(__file__).resolve().parents[2]/'scripts/selectWannierCenters.py'
spec=importlib.util.spec_from_file_location('select_centers',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
geometry=dict(lattice_bohr=R.tolist(),positions_fractional=[[0,0,0],[.25,.25,.25]],species=['Si','Si'])
(root/'geometry.json').write_text(json.dumps(geometry))
mask=module.generate(root/'all-fixed.in',root/'geometry.json',root/'radius-mask.in',1.,np.zeros(3))
assert mask['n_free']==4 and mask['n_fixed']==4
assert [x['fixed'] for x in mask['trials']]==[False]*4+[True]*4
assert 'fixCenters no' in (root/'radius-mask.in').read_text()
assert f'wannier-dump-name {root}/radius-mask.$VAR' in (root/'radius-mask.in').read_text()
region=execute('wannier','radius-mask',dry=True)
for i in range(8):assert f'center {i}: '+('free' if i<4 else 'fixed at') in region
skew=np.array([[6.,5.8,.2],[0.,1.2,.3],[0.,0.,5.]])
delta=np.array([.49,.49,.12])
expected=min(np.linalg.norm(skew@(delta-np.array(n)))*.529177210903 for n in itertools.product(range(-5,6),repeat=3))
assert abs(module.minimum_image_distance(delta,skew)-expected)<1e-12
assert abs(module.minimum_image_distance(delta+[3,-2,1],skew)-expected)<1e-12
(root/'hidden-extra.in').write_text('wannier-center Si 2 s\n')
(root/'hidden.in').write_text((root/'all-fixed.in').read_text()+'include hidden-extra.in\n')
try:module.generate(root/'hidden.in',root/'geometry.json',root/'hidden-mask.in',1.,np.zeros(3))
except ValueError as error:assert 'directly' in str(error)
else:raise AssertionError('Included Wannier trials were silently left outside the mask')
(root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2),flush=True)
print('Selective hard-center regressions passed',flush=True)
