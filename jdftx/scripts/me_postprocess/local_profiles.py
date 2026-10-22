"""Wannier-site and bond diagnostics for a saved projected singlet BdG state."""
import numpy as np
from . import KB_EV
from .backend import select_backend,host


def onsite_and_bonds(cells,shape):
    """BCC nearest neighbors, including the supercell image of each endpoint."""
    cells,shape=np.asarray(cells,int),np.asarray(shape,int)
    lookup={tuple(r):i for i,r in enumerate(cells)}
    steps=np.vstack((np.eye(3,dtype=int),-np.eye(3,dtype=int),[[1,1,1],[-1,-1,-1]]))
    bonds=[]
    for i,r in enumerate(cells):
        for step in steps:
            target=r+step
            image=np.floor_divide(target,shape)
            j=lookup.get(tuple(target % shape))
            if j is None:
                continue  # The deleted site has no orbital/pair amplitude.
            forward=(i,j,*image)
            reverse=(j,i,*(-image))
            if forward<reverse:
                bonds.append((i,j,image,step))
    left=np.array([b[0] for b in bonds],int)
    right=np.array([b[1] for b in bonds],int)
    image=np.array([b[2] for b in bonds],int)
    step=np.array([b[3] for b in bonds],int)
    return left,right,image,step


def correlation_blocks(archive,basis_archive,left,right,images,temperature,total_points,
                       norb=6,backend='cpu',chunk=128,progress=None):
    """F_ia,jb(n)=sum_q exp(-2πiq.n) sum_E>0 u_ia v_jb* tanh(E/2T)."""
    compute=select_backend(backend);xp=compute.xp
    left=xp.asarray(np.asarray(left)[:,None]*norb+np.arange(norb))
    right=xp.asarray(np.asarray(right)[:,None]*norb+np.arange(norb))
    images=xp.asarray(images)
    result=xp.zeros((len(left),norb,norb),complex)
    sizes=sorted(int(k.split('_')[2]) for k in archive.files if k.startswith('projected_bdg_'))
    for size in sizes:
        matrices=archive[f'projected_bdg_{size}_ev']
        normal=basis_archive[f'vectors_{size}']
        twists=archive[f'twists_{size}']
        if not np.allclose(twists,basis_archive[f'twists_{size}'],atol=1e-14,rtol=0):
            raise ValueError('saved BdG and normal bases use different twist order')
        for start in range(0,len(matrices),chunk):
            energy,states=xp.linalg.eigh(xp.asarray(matrices[start:start+chunk]))
            vectors=xp.asarray(normal[start:start+chunk])
            u=vectors @ states[:,:size,size:]
            v=vectors @ states[:,size:,size:]
            thermal=xp.tanh(energy[:,size:]/(2*KB_EV*temperature))
            phase=xp.exp(-2j*np.pi*xp.asarray(twists[start:start+chunk]) @ images.T)
            a=u[:,left,:]*thermal[:,None,None,:]*phase[:,:,None,None]
            b=v[:,right,:]
            # Batch over blocks, contracting both k points and positive states.
            a=a.transpose(1,2,0,3).reshape(len(left),norb,-1)
            b=b.transpose(1,2,0,3).reshape(len(right),norb,-1)
            result+=(a @ b.conj().swapaxes(-1,-2))/total_points
        if progress:
            progress(size)
    return host(result)


def uniform_bulk_blocks(basis_archive,gap_ev,offsets,temperature,backend='cpu',chunk=2048):
    """Homogeneous reference F_ab(R), with the identical pairing window."""
    compute=select_backend(backend);xp=compute.xp
    offsets=np.asarray(offsets)
    norb=len(basis_archive['normal_density'])
    result=xp.zeros((len(offsets),norb,norb),complex)
    total=int(basis_archive['total_points'])
    sizes=sorted(int(k.split('_')[1]) for k in basis_archive.files if k.startswith('xi_'))
    for size in sizes:
        xi=basis_archive[f'xi_{size}'];vectors=basis_archive[f'vectors_{size}']
        points=basis_archive[f'twists_{size}']
        for start in range(0,len(xi),chunk):
            e=xp.asarray(xi[start:start+chunk]);u=xp.asarray(vectors[start:start+chunk])
            energy=xp.sqrt(e**2+gap_ev**2)
            weight=gap_ev/(2*energy)*xp.tanh(energy/(2*KB_EV*temperature))
            phases=xp.exp(-2j*np.pi*xp.asarray(points[start:start+chunk]) @ xp.asarray(offsets).T)
            right=u.transpose(1,0,2).reshape(norb,-1)
            for index in range(len(offsets)):
                left=(u*weight[:,None,:]*phases[:,index,None,None]).transpose(1,0,2).reshape(norb,-1)
                result[index]+=left @ right.conj().T/total
    return host(result)


def shell_average(radius,values,tolerance=1e-6):
    radius,values=np.asarray(radius),np.asarray(values)
    keys=np.rint(radius/tolerance).astype(np.int64)
    unique=np.unique(keys)
    groups=[np.flatnonzero(keys==key) for key in unique]
    return (np.array([radius[g].mean() for g in groups]),
            np.array([len(g) for g in groups]),
            np.array([values[g].mean(axis=0) for g in groups]),
            np.array([values[g].min(axis=0) for g in groups]),
            np.array([values[g].max(axis=0) for g in groups]))


def threshold_half_gap(energy_ev,ldos,fraction=.1,peak_window_ev=.0035):
    """Average of positive/negative first-threshold onsets, in eV.

    Each side uses its own peak within |E|<=peak_window_ev. Lorentzian tails
    may make the threshold already exceed zero bias, giving a zero estimate.
    """
    energy_ev,ldos=np.asarray(energy_ev),np.atleast_2d(ldos)
    if not 0<fraction<1:
        raise ValueError('LDOS threshold fraction must be between zero and one')
    if ldos.shape[1]!=len(energy_ev) or not np.all(np.diff(energy_ev)>0):
        raise ValueError('LDOS and strictly increasing energy grid must match')
    onsets=[]
    for sign in (1,-1):
        keep=(sign*energy_ev>=-1e-14)&(sign*energy_ev<=peak_window_ev)
        order=np.argsort(sign*energy_ev[keep])
        x=(sign*energy_ev[keep])[order]
        y=ldos[:,keep][:,order]
        threshold=fraction*y.max(axis=1)
        first=np.argmax(y>=threshold[:,None],axis=1)
        onset=x[first].copy()
        for i,index in enumerate(first):
            if index:
                y0,y1=y[i,index-1:index+1]
                onset[i]=x[index-1]+(x[index]-x[index-1])*(threshold[i]-y0)/(y1-y0)
        onsets.append(onset)
    return (onsets[0]+onsets[1])/2,np.column_stack(onsets)
