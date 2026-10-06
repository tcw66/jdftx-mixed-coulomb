#!/usr/bin/env python3
"""Generate a silicon fcc dispersion path in the reciprocal basis used by system.in."""
import argparse
import json
from pathlib import Path
import numpy as np

# Cartesian conventional reciprocal coordinates are q_frac @ (2*pi*inv(R)).
# For example, X=(.5,0,.5) maps to (0,1,0)*2*pi/a.
POINTS = {
    'G': [0., 0., 0.], 'X': [.5, 0., .5], 'W': [.5, .25, .75],
    'K': [.375, .375, .75], 'L': [.5, .5, .5], 'U': [.625, .25, .625],
}
PATHS = [['G', 'X', 'W','L','G']]


def generate(lattice, intervals):
    R = .5*lattice*np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    G = 2*np.pi*np.linalg.inv(R)
    qs, distance, ticks, breaks = [], [], [], []
    total = 0.
    for path in PATHS:
        breaks.append(len(qs))
        qs.append(POINTS[path[0]])
        distance.append(total)
        ticks.append({'index': len(qs)-1, 'label': path[0], 'distance': total})
        for start, end in zip(path[:-1], path[1:]):
            a, b = np.array(POINTS[start]), np.array(POINTS[end])
            step = np.linalg.norm((b-a)@G)/intervals
            for i in range(1, intervals+1):
                qs.append(a+(b-a)*i/intervals)
                total += step
                distance.append(total)
            ticks.append({'index': len(qs)-1, 'label': end, 'distance': total})
    return np.array(qs), np.array(distance), {
        'lattice_bohr': lattice, 'primitive_lattice_columns_bohr': R.tolist(),
        'points': POINTS, 'paths': PATHS, 'ticks': ticks, 'breaks': breaks,
        'distance_units': 'bohr^-1', 'points_per_segment': intervals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lattice', type=float, default=10.26)
    parser.add_argument('--points-per-segment', type=int, default=60)
    parser.add_argument('--out', type=Path, default=Path('.'))
    args = parser.parse_args()
    if args.lattice <= 0 or args.points_per_segment < 1:
        parser.error('lattice and points-per-segment must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    qs, distance, metadata = generate(args.lattice, args.points_per_segment)
    np.savetxt(args.out/'qpath.dat', qs, header='fractional q0 q1 q2')
    np.savetxt(args.out/'qpath-distance.dat', np.column_stack((distance, qs)),
               header='distance[bohr^-1] q0 q1 q2')
    (args.out/'qpath.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Generated {len(qs)} q points in {args.out / "qpath.dat"}')


if __name__ == '__main__':
    main()
