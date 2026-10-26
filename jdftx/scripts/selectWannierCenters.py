#!/usr/bin/env python3
"""Select fixed atomic trials outside a periodic defect neighborhood."""
from pathlib import Path
import argparse,itertools,json,os,re
import numpy as np

BOHR=0.529177210903

def minimum_image_distance(delta,lattice):
    """Exact nearest image via a finite bound from the smallest singular value."""
    d=np.asarray(delta,float)-np.rint(delta)
    upper=float(np.linalg.norm(lattice@d));smallest=float(np.linalg.svd(lattice,compute_uv=False)[-1])
    if smallest<=1e-12:raise ValueError('Singular lattice')
    radius=upper/smallest+1e-12
    bounds=[range(int(np.ceil(x-radius)),int(np.floor(x+radius))+1) for x in d]
    return min(np.linalg.norm(lattice@(d-np.array(n))) for n in itertools.product(*bounds))*BOHR

def generate(input_path,geometry_path,output_path,free_radius,defect):
    source=Path(input_path).resolve();out=Path(output_path).resolve();g=json.loads(Path(geometry_path).read_text())
    if source==out:raise ValueError('Output must be a new input file')
    if not np.isfinite(free_radius) or free_radius<0:raise ValueError('Free radius must be finite and nonnegative')
    lattice=np.array(g['lattice_bohr'],float);positions=np.array(g['positions_fractional'],float)
    defect=np.asarray(defect,float)
    if lattice.shape!=(3,3) or positions.ndim!=2 or positions.shape[1]!=3 or defect.shape!=(3,) or not all(np.isfinite(x).all() for x in [lattice,positions,defect]):raise ValueError('Invalid lattice, positions or defect coordinates')
    species=g.get('species',['Cu']*len(positions))
    if len(species)!=len(positions):raise ValueError('Species/positions length mismatch')
    atoms={};counts={}
    for i,(element,pos) in enumerate(zip(species,positions)):
        counts[element]=counts.get(element,0)+1
        atoms[(element,counts[element])]=(i+1,pos,minimum_image_distance(pos-defect,lattice))
    text=source.read_text()
    visible='\n'.join(line.split('#',1)[0] for line in text.splitlines())
    if not re.search(r'^\s*wannier\s',visible,re.M) or not re.search(r'^\s*wannier-dump-name\s',visible,re.M):raise ValueError('List Wannier settings, dump prefix and trials directly in the input for this helper')
    def inspect_includes(contents,stack):
        for line in contents.splitlines():
            directive=line.split('#',1)[0].strip()
            if not directive.startswith('include '):continue
            path=Path(os.path.expandvars(directive.split(None,1)[1]))
            if not path.is_absolute():path=source.parent/path
            path=path.resolve()
            if path in stack:raise ValueError('Cyclic include')
            included=path.read_text()
            if re.search(r'^\s*wannier(?:\s|-)', '\n'.join(x.split('#',1)[0] for x in included.splitlines()),re.M):raise ValueError('List Wannier settings and trials directly, rather than inside includes')
            inspect_includes(included,stack+[path])
    inspect_includes(text,[source])
    if re.search(r'^\s*wannier-(?:center-constraint\s|frozen)',text,re.M):raise ValueError('Indexed targets/frozen WFs need an explicit manual mapping; use plain atomic trials for this helper')
    if re.search(r'addAtomicOrbitals\s+yes',text):raise ValueError('List atomic trials explicitly for radius selection')
    rows=[];lines=[]
    for line in text.splitlines():
        if re.match(r'\s*wannier-center(?:\s|-fixed\s|-pinned\s|-vacuum\s)',line):
            match=re.fullmatch(r'(\s*)wannier-center(?:-fixed)?\s+(\S+)\s+(\d+)\s+(\S+)\s*(?:#.*)?',line)
            if not match:raise ValueError('Radius helper accepts single atomic trials; use wannier-center-fixed directly for composites/Gaussians/numerical orbitals')
            indent,element,atom,orb=match.groups();atom=int(atom)
            if (element,atom) not in atoms:raise ValueError(f'Missing geometry for {element} {atom}')
            global_atom,pos,distance=atoms[(element,atom)];fixed=distance>free_radius+1e-12
            command='wannier-center-fixed' if fixed else 'wannier-center'
            lines.append(f'{indent}{command} {element} {atom} {orb}')
            rows.append(dict(trial_index=len(rows),species=element,species_atom=atom,global_atom=global_atom,orbital=orb,distance_from_defect_A=float(distance),fixed=bool(fixed),target_fractional=pos.tolist()))
        elif re.match(r'\s*(include|wannier-initial-state)\s+',line):
            command,value=line.split('#',1)[0].strip().split(None,1)
            if command=='include':value=os.path.expandvars(value)
            path=Path(value)
            lines.append(command+' '+str(path if path.is_absolute() else (source.parent/path).resolve()))
        elif re.match(r'\s*wannier-dump-name\s+',line):lines.append('wannier-dump-name '+str(out.parent/(out.stem+'.$VAR')))
        else:lines.append(line)
    if not rows:raise ValueError('No atomic trials found')
    result=re.sub(r'fixCenters\s+(yes|no)','fixCenters no','\n'.join(lines)+'\n')
    result=re.sub(r'loadRotations\s+(yes|no)','loadRotations no',result)
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and out.read_text()!=result:raise ValueError('Refusing to overwrite a different existing input')
    out.write_text(result)
    metadata=dict(source_input=str(source),geometry=str(Path(geometry_path).resolve()),free_radius_A=free_radius,defect_fractional=np.array(defect).tolist(),n_fixed=sum(r['fixed'] for r in rows),n_free=sum(not r['fixed'] for r in rows),trials=rows,note='Periodic nearest-image distances; centers on the radius boundary are free. Radius selects trials, not a confinement region. Trial indices exclude separately frozen WFs, which this helper rejects. New dump prefix prevents overwriting source outputs.')
    out.with_suffix('.mask.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--free-radius',type=float,required=True,help='Angstrom')
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--defect-atom',type=int,help='1-based global atom index in geometry JSON');group.add_argument('--defect-fractional',nargs=3,type=float)
    a=p.parse_args();g=json.loads(a.geometry.read_text())
    if a.defect_atom is not None:
        if not 1<=a.defect_atom<=len(g['positions_fractional']):p.error('Defect atom out of range')
        defect=np.array(g['positions_fractional'][a.defect_atom-1])
    else:defect=np.array(a.defect_fractional)
    m=generate(a.input,a.geometry,a.output,a.free_radius,defect)
    print(f"Fixed {m['n_fixed']}; free {m['n_free']}; input {a.output}")

if __name__=='__main__':main()
