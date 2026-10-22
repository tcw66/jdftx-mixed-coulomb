# Settings shared by workflow.sh and run_all.sh. Environment overrides are allowed.
SOURCE_DIR="${SOURCE_DIR:-/home/tyler/koopmans}"
OUTPUT_DIR="${OUTPUT_DIR:-$SOURCE_DIR/archive/coarse-screening-4x4x4-20261014}"
COARSE_GRID="${COARSE_GRID:-4 4 4}"
BACKEND="${BACKEND:-gpu}"  # gpu or cpu
RANKS="${RANKS:-1}"       # One rank avoids assigning multiple ranks to one GPU.
THREADS="${THREADS:-4}"
# Empty means inherit the source ki-response.in value. Zero disables image corrections.
# A positive value must be a separately established electronic dielectric constant.
EPSILON_INFINITY="${EPSILON_INFINITY:-}"
MINIMUM_DENSITY_OVERLAP="${MINIMUM_DENSITY_OVERLAP:-0.9}"
SCF_LOG="${SCF_LOG:-$SOURCE_DIR/scf.out}"
ORBITAL_MAP="${ORBITAL_MAP:-}"  # Optional JSON valence/conduction permutations.
