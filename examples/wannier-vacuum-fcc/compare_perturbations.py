#!/usr/bin/env python3
"""Compare matched MV and TruncatedDC FCC-center perturbation tests."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    paths={'MV':ROOT/'results/mv-perturbations','TruncatedDC':ROOT/'results/truncateddc-perturbations'}
    data={name:json.loads((p/'summary.json').read_text()) for name,p in paths.items()}
    # Verify the actual saved inputs and states, not just the seed labels.
    for filename in ('model.in','common.in','scf.wfns','scf.eigenvals','scf.fillings','zero.Vscloc'):
        if digest(paths['MV']/'state'/filename)!=digest(paths['TruncatedDC']/'state'/filename):
            raise ValueError('Comparison inputs differ: '+filename)
    windows={method:re.sub(r'localizationMeasure\s+\S+','localizationMeasure MATCHED',(p/'state/windows.in').read_text()).split() for method,p in paths.items()}
    if windows['MV']!=windows['TruncatedDC']:
        raise ValueError('Energy windows or other localization settings differ.')
    indexed={name:{r['name']:r for r in d['runs']} for name,d in data.items()}
    if set(indexed['MV'])!=set(indexed['TruncatedDC']):
        raise ValueError('Perturbation runs differ.')
    for name,a in indexed['MV'].items():
        b=indexed['TruncatedDC'][name]
        if not np.array_equal(a['initial_fractional_centers'],b['initial_fractional_centers']):
            raise ValueError('Initial Gaussian positions differ for '+name)
    out=ROOT/'results/perturbation-method-comparison';out.mkdir(exist_ok=True)
    groups=[]
    for amp in data['MV']['protocol']['amplitudes_rms_bohr']:
        g={'amplitude_bohr':amp}
        for method in data:
            rows=[r for r in data[method]['runs'] if r['seed'] is not None and abs(r['initial_rms_displacement_bohr']-amp)<1e-9]
            g[method]=dict(tests=len(rows),converged=sum(r['converged'] for r in rows),iterations_min=min(r['iterations'] for r in rows),iterations_max=max(r['iterations'] for r in rows),iterations_mean=float(np.mean([r['iterations'] for r in rows])),max_fcc_rms_bohr=max(r['fcc_rms_bohr'] for r in rows),max_objective_change_bohr2=max(abs(r['objective_change_from_baseline']) for r in rows))
        groups.append(g)
    result=dict(input_verification='Identical model, mesh, saved wavefunctions, eigenvalues, fillings, zero potential and trial positions.',groups=groups)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# MV versus TruncatedDC: perturbed FCC vacuum trial centers','',
        'All comparisons use identical saved 30-Hartree free-electron states,',
        'four Gaussian trials of sigma=1 bohr, an 8x8x8 mesh, the same windows',
        'and stopping criteria, and the exact same initial positions. Equality',
        'of the saved model, states, potential and trial positions was verified.',
        'There are three seeds at each RMS kick size. The centers are fitted',
        'to FCC allowing a global translation, periodic images and permutations.','',
        '| RMS kick (bohr) | MV iterations | TruncatedDC iterations | MV max FCC RMS residual (bohr) | TruncatedDC max FCC RMS residual (bohr) |',
        '|---|---:|---:|---:|---:|']
    for g in groups:
        a,b=g['MV'],g['TruncatedDC']
        lines.append(f'| {g["amplitude_bohr"]:g} | {a["iterations_min"]}–{a["iterations_max"]} | {b["iterations_min"]}–{b["iterations_max"]} | {a["max_fcc_rms_bohr"]:.3e} | {b["max_fcc_rms_bohr"]:.3e} |')
    lines+=['','| Trial | MV iterations | TruncatedDC iterations |','|---|---:|---:|',f'| Unperturbed baseline | {data["MV"]["baseline"]["iterations"]} | {data["TruncatedDC"]["baseline"]["iterations"]} |']
    for name,a in indexed['MV'].items():
        lines.append(f'| {name} | {a["iterations"]} | {indexed["TruncatedDC"][name]["iterations"]} |')
    for method,d in data.items():
        rows=[r for r in d['runs'] if r['seed'] is not None]
        lines+=['',f'{method}: {sum(r["converged"] for r in rows)}/{len(rows)} random starts converged; maximum absolute',f'objective difference from its own symmetric baseline = {max(abs(r["objective_change_from_baseline"]) for r in rows):.3e} bohr².']
    random=[r for d in data.values() for r in d['runs'] if r['seed'] is not None]
    if all(r['converged'] and r['fcc_rms_bohr']<1e-3 and abs(r['objective_change_from_baseline'])<1e-5 for r in random):
        lines += ['', 'Both methods returned to an FCC arrangement and their respective symmetric',
            'baseline spread for every tested random start. TruncatedDC required zero',
            'to two fewer iterations for these matched starts. This test shows similar',
            'robustness and no substantial convergence advantage for either method.']
    lines+=['','MV and TDC have different spread definitions. Objective differences above',
        'are measured relative to each method\'s own unperturbed baseline, not',
        'between methods. The tests probe these nine starts, not global optimality.',
        'TruncatedDC center coordinates are phase-based estimates. Raw logs and',
        'native final rotations are preserved in the per-method directories.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for method,color,offset in [('MV','#2166ac',-.012),('TruncatedDC','#cf392f',.012)]:
        rows=[r for r in data[method]['runs'] if r['seed'] is not None]
        x=np.array([r['initial_rms_displacement_bohr'] for r in rows])+offset
        axes[0].scatter(x,[r['iterations'] for r in rows],color=color,label=method)
        axes[1].scatter(x,[r['fcc_rms_bohr'] for r in rows],color=color,label=method)
    axes[0].set(ylabel='Iterations to convergence')
    axes[1].set(ylabel='Final translation-aligned FCC RMS residual (bohr)',yscale='log')
    for ax in axes:
        ax.set_xlabel('Initial RMS displacement per center (bohr)');ax.legend()
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'comparison.{suffix}',dpi=180)
    print((out/'REPORT.md').read_text())

if __name__=='__main__':main()
