# Rectangular periodic box revision

Baseline: `9ee7457`; 66 tests passed before revision.

Contract:
- New `domain.length` and `domain.depth` fix the dimensions to `(L, L, D)`.
- All three axes remain periodic. Legacy thickness/box-size configurations retain their original resolution behavior.
- Automatic particle counts account for actual sampled volumes; validation distinguishes count/distribution resolution from geometry or generator errors.
- Cubic algorithms keep their old scalar API; rectangular geometry uses three lengths internally.
- Unsupported crystal dimensions fail explicitly rather than straining a crystal silently.

Stages: baseline, geometry, resolution, basic generators/export, force-biased,
analysis/rendering, advanced generators, GUI/documentation and final acceptance.

## Completed verification (2026-10-02)

- Original baseline: 66 passing tests. These historical measurements precede
  the narrowed public release; see the release checklist for current validation.
- Independent explicit-image distance/overlap oracle covers cube, shallow,
  elongated and unequal-three-axis internal boxes.
- Fixed-size resolution covers monodisperse, quasi-monodisperse, lognormal and
  discrete-mixture distributions, manual-count conflict, units and repeatability.
- Basic generator tests replay exported configurations and verify JSON/HDF5 dimensions.
- Advanced tests cover unstrained FCC cells, explicit HCP/incommensurate rejection
  and event-driven periodic collision handling.
- A non-nearest periodic image collision test exposed an LS scheduling edge case
  in thin boxes. Fixed-box LS now predicts adjacent periodic images in both scalar
  and vectorized event scheduling, limits periodic recheck horizons using the
  shortest dimension, and passes the short-time backend trajectory comparison.
  Long chaotic trajectories are assessed by geometric validity, not bitwise identity.
- Streamlit AppTest performs a real Generate interaction with unequal dimensions,
  checks displayed dimensions, reusable configuration, default PNG output and
  the absence of default HDF5. Static plotting uses a headless backend.
- Full example: `2 x 2 x 1 um`, 200 nm mean diameter, phi target 0.50:
  477 particles; actual phi 0.4995132319207775; no excessive overlaps. Generation
  0.526 s, validation 0.058 s, analysis 0.069 s, plots 0.747 s, rendering 6.618 s.
- Stability matrix: 27/27 force-biased cases pass at N=500, phi=0.50/0.60/0.62,
  D/L=0.25/1/4 and seeds=123/456/789. Maximum overlaps remain within 1e-5.
- Scale matrix: 8/8 cases pass at N=1000/10000 for Boolean, Poisson-disk, RSA and
  force-biased (D/L=0.25). Analysis plots and rendering are disabled for these timings;
  nearest-neighbor analysis remains enabled.
- Legacy cubic comparison against original `9ee7457`: 8 paired cases at
  N=1000/10000 have exactly identical exported coordinates and radii (maximum
  absolute difference zero). No runtime regression was observed in this local
  single-run comparison; timings depend on machine load and are not speed guarantees.

Raw local reports (ignored generated outputs):
- `results/rectangular_validation/summary.json`
- `results/baseline_comparison/summary.json`
- `results/new_cube_comparison/summary.json`
- `results/new_cube_comparison/geometry_comparison.json`
- `results/revision_checkpoints/summary.json`

## Boundaries

All axes remain periodic; open-boundary slab generation is outside this revision.
Crystals require commensurate dimensions and density with complete unstrained
cells; fixed-box HCP is explicitly unsupported. High-density convergence above
validated cases (including phi=0.64) and extreme aspect ratios remain experimental.
Custom size files are archived into the portable run bundle for replay. These
historical measurements used a local Windows interpreter. The release checklist
records current package and multi-platform CI validation.


## Measured 10000-particle generation times

| Algorithm | Rectangular D/L=0.25 (s) | Original cube (s) | Revised legacy cube (s) |
|---|---:|---:|---:|
| marked_poisson_boolean | 0.000 | 0.001 | 0.000 |
| poisson_disk | 3.788 | 5.006 | 3.391 |
| rsa | 7.296 | 8.956 | 6.520 |
| force_biased | 23.419 | 23.394 | 20.197 |

The public release contains the final supported source tree. Local development
snapshots and internal review records are preserved privately.
