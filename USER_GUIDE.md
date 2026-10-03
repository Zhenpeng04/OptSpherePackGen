# OptSpherePackGen User Guide

English | [简体中文](USER_GUIDE.zh-CN.md) · [README](README.md)

This guide explains how to generate microsphere geometries, inspect their
quality and transfer them to optical simulation software. The Python package
and command-line executable are both named `spherepackgen`.

## 1. Install and choose a workspace

Use Python 3.10 or newer. Download or clone the repository, open a terminal
in its directory and install into a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\spherepackgen.exe gui --workdir .
```

macOS / Linux:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install .
./.venv/bin/spherepackgen gui --workdir .
```

Alternatively, install a wheel with
`python -m pip install /path/to/spherepackgen-0.2.0-py3-none-any.whl`.
Installation downloads dependencies; Anaconda is not required.

The commands above do not require environment activation. For the shorter
`spherepackgen` commands below, activate the environment with
`.\.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on macOS/Linux. If activation is blocked on Windows,
use the full path to `spherepackgen.exe` instead.

Choose a writable directory for your configurations, uploads and results:

```text
spherepackgen gui --workdir /path/to/workspace
spherepackgen examples --output examples
spherepackgen run examples/fcc.yaml
```

Without `--workdir`, the GUI uses the directory where it is launched.
Its upload cache and history are stored under `.spherepackgen/` in that
workspace. Restarting with the same workspace restores the saved history.
Example export refuses to overwrite existing example files.

Optional PyVista rendering can be installed from the source directory using
`python -m pip install ".[visualization]"`; matplotlib rendering remains
available without it.

## 2. Choose a structure and size distribution

| Spatial order | Geometry | Typical choices |
|---|---|---|
| `periodic_crystal` | Repeated crystal unit cells | SC, BCC, FCC, diamond cubic; HCP with automatically resolved cubic dimensions |
| `hard_core_random` | Random spheres subject to periodic non-overlap | Poisson-disk, RSA, force-biased, LS |
| `overlapping_random` | Independent random centers, with sphere overlap allowed | Marked Poisson Boolean media |

| Size class | Configuration |
|---|---|
| `monodisperse` | Equal radii |
| `quasi_monodisperse` | Truncated normal sampling; `CV_radius` defaults to 0.03 |
| `continuous_polydisperse` | Choose `lognormal`, `truncated_normal`, `uniform`, `gamma`, `weibull` or `custom_file` |
| `discrete_mixture` | Specify relative sizes and number fractions through YAML/API |
| `imported_distribution` | Sample from a particle-size file through YAML/API |

The GUI offers monodisperse, quasi-monodisperse and continuous polydisperse
sizes. Its crystal option requires monodisperse particles. SHU, external
RCPGenerator, generic hyperuniform, quasicrystal and custom correlation
generation are not available.

## 3. Configure physical dimensions and density

Internal lengths are dimensionless, with the mean particle diameter equal
to 1. Physical input lengths accept units such as `200 nm` or `0.2 um`.
Real-coordinate export uses `um` by default.

| Field | Meaning |
|---|---|
| `physical.mean_diameter` | Mean diameter of the generated sample |
| `physical.packing_fraction` | Target sphere volume fraction for non-overlapping structures |
| `physical.covered_volume_fraction` | Target expected Boolean coverage for overlapping media; between 0 and 1 |
| `domain.length` | Fixed lateral length, shared by x and y |
| `domain.depth` | Fixed z length |
| `particles.num_particles` | `auto` or a positive integer |
| `runtime.random_seed` | Non-negative integer, or `null` to generate a recorded seed |

With `domain.type: periodic_box`, provide both length and depth. The box is
`(L, L, D)`, and **all three axes are periodic**. Do not combine these fields
with `physical.medium_thickness` or `domain.box_size`.

Automatic count selection preserves the configured mean diameter and fixed
box dimensions while choosing an integer count from sampled particle volumes.
For equal spheres:

```text
N ≈ round(phi × L² × D / (pi × mean_diameter³ / 6))
```

A small density mismatch can arise from integer counts. The report distinguishes
this count-resolution allowance from validation tolerance; the allowance is
half the largest sampled sphere volume divided by box volume.
A conflicting manual count or a sphere overlapping its own periodic image
causes an error.

Cubic configurations without fixed length/depth use `domain.type: periodic_cube`.
They can derive the side length from particle count and density, or use
`physical.medium_thickness` as an approximate target side length. The resulting
side may differ because counts and crystal cells are discrete.

### Complete random-packing configuration

Save as `my_case.yaml`:

```yaml
project:
  name: my_case
structure:
  size_distribution: monodisperse
  spatial_order: hard_core_random
physical:
  mean_diameter: 200 nm
  packing_fraction: 0.15
domain:
  type: periodic_box
  length: 1.2 um
  depth: 0.6 um
particles:
  num_particles: auto
size_distribution:
  type: monodisperse
algorithm:
  name: default
analysis:
  compute_g2: true
  compute_Sk: true
  bins: 80
  k_max_index: 3
validation:
  require_non_overlap: true
  tolerance_phi: 1.0e-6
  tolerance_overlap: 1.0e-5
output:
  path: results/my_case
  coordinate_unit: um
  formats: [csv, json]
  make_plots: true
  overwrite: false
runtime:
  random_seed: 12345
```

Run `spherepackgen run my_case.yaml`.

### Overlapping Boolean media

In the complete configuration above, replace the structure and physical
sections with the following, and set `validation.require_non_overlap: false`:

```yaml
structure:
  size_distribution: monodisperse
  spatial_order: overlapping_random
physical:
  mean_diameter: 200 nm
  covered_volume_fraction: 0.70
```

For independent sphere centers, the model relates covered fraction `c` and
nominal fraction `eta` by `eta = -ln(1 - c)`. Nominal fraction is the summed
sphere volume divided by box volume; it may exceed 1 because overlaps count
multiple times. The reported `1 - exp(-eta)` is expected coverage, not a
measurement of the union volume of the finite generated sample.

## 4. Configure particle sizes

To use a continuous distribution, replace both the structure size class and
the size-distribution section:

```yaml
structure:
  size_distribution: continuous_polydisperse
  spatial_order: hard_core_random
size_distribution:
  type: continuous_polydisperse
  distribution: lognormal
  CV_radius: 0.20
  min_radius_factor: 0.40
  max_radius_factor: 1.80
```

| Distribution | Controls |
|---|---|
| `lognormal` | `CV_radius`; optional radius-factor bounds |
| `truncated_normal` | `CV_radius` and radius-factor bounds |
| `uniform` | Lower and upper radius factors |
| `gamma` | `CV_radius` or `parameters.shape` |
| `weibull` | `parameters.shape` |
| `custom_file` | `size_file`, `file_values`, optional `file_column` |

Radius factors are relative to the nominal mean radius. Sampling or clipping
is followed by normalization to the requested sample mean diameter, so final
cutoffs and measured CV can differ from the input settings. Inspect the exported
size statistics when bounds or distribution width matter.

### Custom particle-size files

For `continuous_polydisperse`, use:

```yaml
size_distribution:
  type: continuous_polydisperse
  distribution: custom_file
  size_file: particle_sizes.csv
  file_values: diameter
  file_column: diameter_um
```

An example CSV:

```csv
diameter_um
0.20
0.20
0.40
0.40
```

Files can be CSV, TXT or DAT with comma- or whitespace-separated values.
Use a clean numeric column of positive finite sizes; header names can select
the column, or use a zero-based integer index. Omit `file_column` to select
the first numeric column. File paths are resolved relative to the YAML file.

The file supplies a size population to sample from. Sampling uses replacement
when the requested number exceeds the number of input rows. The generated
sample is then normalized to `physical.mean_diameter`; the input rows are not
a fixed per-particle list, and group counts are not enforced. No material mapping
is inferred from a column name or size value.

In the GUI, choose **Continuous distribution → custom_file**, upload the file
and choose whether its values represent diameters or radii.

### Discrete mixtures through YAML

Set `structure.size_distribution: discrete_mixture` and use:

```yaml
size_distribution:
  type: discrete_mixture
  species:
    - radius_factor: 1.0
      number_fraction: 0.5
    - radius_factor: 2.0
      number_fraction: 0.5
```

The factors specify relative sizes. Number fractions are sampling probabilities,
not exact quotas. All sampled radii are normalized together to the requested
mean diameter. These settings specify geometry, not optical materials.

## 5. Select a generator

| Algorithm | Behavior |
|---|---|
| `crystal` | Deterministic repeated unit cells |
| `marked_poisson_boolean` | Independent random sphere centers; overlap allowed |
| `poisson_disk` | Variable-radius placement using local non-overlap checks |
| `rsa` | Random sequential insertion until all particles are placed or the attempt budget is exhausted |
| `force_biased` | Growth and repulsive relaxation toward target sizes |
| `lubachevsky_stillinger` | Event-driven motion and elastic collisions during radius growth |
| `metropolis` | RSA initialization followed by accepted non-overlapping random moves; YAML/API only |

With `algorithm.name: default`:

| Spatial order / fraction | Selected algorithm |
|---|---|
| `periodic_crystal` | `crystal` |
| `overlapping_random` | `marked_poisson_boolean` |
| `hard_core_random`, `phi <= 0.18` | `poisson_disk` |
| `hard_core_random`, `0.18 < phi <= 0.34` | `rsa` |
| `hard_core_random`, `phi > 0.34` | `force_biased` |

If automatic Poisson-disk or RSA placement fails, the workflow can fall back
to force-biased generation. Explicitly selecting Poisson-disk or RSA preserves
that choice and reports placement failure.

Force-biased controls include `initial_radius_fraction`, `stages`,
`relaxation_steps_per_stage` and `contraction_rate`. Stronger default
relaxation/cleanup budgets apply at `phi >= 0.60`. LS controls include
`initialization`, `initial_radius_scale`, `compression_rate`,
`velocity_scale`, `max_events`, `event_backend` and `final_cleanup_steps`.
LS automatic initialization selects a force-biased warm start at
`phi >= 0.60`. Consult the
[force-biased example](configs/examples/force_biased.yaml) and
[LS example](configs/examples/lubachevsky_stillinger.yaml) for complete settings.

Higher density or broader particle sizes can require more time and tuning.
Neither force-biased relaxation nor LS output alone proves equilibrium,
random close packing or maximally random jamming.

### Crystals

For a crystal with automatically determined cubic dimensions:

```yaml
project:
  name: fcc_case
structure:
  size_distribution: monodisperse
  spatial_order: periodic_crystal
physical:
  mean_diameter: 200 nm
  packing_fraction: 0.50
domain:
  type: periodic_cube
particles:
  num_particles: auto
algorithm:
  name: crystal
  parameters:
    lattice_type: FCC
    unit_cells: 2
output:
  path: results/fcc_case
  make_plots: false
  overwrite: false
```

For fixed boxes, use complete, unstrained cells. A three-element
`unit_cells: [nx, ny, nz]` can specify the tiling, but each cell must have
the same side length and the resulting density must match the target.
The GUI's **Apply compatible crystal dimensions** button computes compatible
length/depth from your chosen cell counts. Fixed-box HCP is rejected.

Maximum non-overlapping lattice fractions are approximately SC 0.5236,
BCC 0.6802, FCC/HCP 0.7405 and diamond cubic 0.3401. Changing a random seed
does not change a deterministic crystal.

## 6. Operate the GUI

Start with `spherepackgen gui --workdir /path/to/workspace`.
The server binds to localhost. Use `--port 8502` if the default port is busy,
or `--headless` to skip automatically opening a browser.

1. In **Run**, select size distribution, structure, dimensions and density.
2. Review the particle-count preview and any input errors.
3. In **Advanced**, adjust distribution, algorithm, analysis, validation and
   export settings. The configuration preview reflects the current settings.
4. Select **Generate** and watch the stage progress.
5. Inspect validation and downloads in **Results**; use **History** for past runs.

The GUI limits the resolved count to 100,000 particles. Its default run time
limit is 600 seconds. **Cancel run** or a timeout stops the background job and
preserves earlier completed results. Each generation reserves its own output
directory, and only files listed in that run's manifest are displayed.

Quick, Standard and Detailed `S(k)` presets use maximum reciprocal indices
5, 10 and 20. Increasing this index increases computation substantially.
Disabling plots does not disable analysis; the `Compute g2(r)` and
`Compute S(k)` controls do that separately.

The GUI defaults to CSV/JSON, enabled plots and one perspective packing view.
HDF5 is optional. Preview images use at most 1,000 particles and label the
displayed subset; exported geometry and validation include all particles.

## 7. Interpret validation and output

Finite coordinates, positive radii, periodic boundaries, density consistency,
size statistics and required non-overlap are checked. Review errors, warnings,
generator diagnostics and the `export_ready` readiness flag before simulation.

| CLI exit code | Meaning |
|---|---|
| 0 | Geometry valid and generator status successful; or a valid candidate explicitly accepted with `--allow-candidate` |
| 1 | Input, file or generation error |
| 2 | Geometry validation failed |
| 3 | Geometry valid but generator warning or target not met |

Strict behavior is the default. `spherepackgen run my_case.yaml --allow-candidate`
changes batch-job acceptance of valid candidates; it does not change their
recorded readiness flags or permit failed geometry validation.

`tolerance_phi` is an absolute dimensionless fraction tolerance.
`tolerance_overlap` is a length tolerance in internal dimensionless units.
Increasing a tolerance does not fix invalid geometry.

| File | Contents |
|---|---|
| `particles_real_units.csv` | Centers, radii and diameters in `output.coordinate_unit` (default `um`) |
| `particles_dimensionless.csv` | Geometry with mean diameter normalized to 1 |
| `metadata.json` | Dimensions, units, resolved parameters, diagnostics, validation and readiness |
| `validation_report.json` | Detailed validation metrics |
| `config_input.yaml` / `config_resolved.yaml` | Original input and resolved settings |
| `config_replay.yaml` | Effective seed, archived input paths and a fresh output destination |
| `particles_snapshot.npz` | Exact dimensionless geometry and labels |
| `inputs/*` | Archived custom particle-size input |
| `environment.json` | Software/source information and dependency versions |
| `run_manifest.json` | Output list, checksums and readiness |
| `run_bundle.zip` | Portable configuration, input and result bundle |
| `analysis/*.csv` | Nearest-neighbor distances, `g2(r)` and `S(k)`, when enabled |
| `particles.h5` | Optional dimensionless geometry and analysis |
| `figures/*.png` | Size histogram, analysis plots and packing preview, when enabled |
| `logs/run.log` | Run summary |

Coordinate CSV columns are
`particle_id,x,y,z,radius,diameter,species_id,material_id`.
Use exported diameters rather than assuming every sphere equals the mean.
The label fields do not provide refractive indices or a material database;
generated material IDs default to zero.

Metadata records box lengths in dimensionless units and meters. HDF5 geometry,
`g2` distances and nearest-neighbor distances use internal lengths; `S(k)`
wavevectors use inverse internal lengths. Convert length by multiplying by
the physical mean diameter, and convert wavevector by dividing by it.
For rectangular boxes, `g2(r)` extends to half the shortest side.

Relative output paths normally use the launch directory. Set
`output.relative_to_config: true` to use the YAML directory instead.
The CLI defaults to `output.overwrite: true`; choose `false` to preserve
previous runs and reserve a fresh directory. GUI runs are already isolated.

## 8. Transfer and reproduce a structure

Download **Download complete result bundle**, extract it on another computer
with the package installed and run:

```text
spherepackgen run /path/to/extracted/config_replay.yaml
```

The input size file travels with the bundle and is checksum-verified.
Replay writes to `replay/` next to the configuration, adding a unique suffix
if needed. Original results are preserved.

A null seed generates a recorded seed for that run. Repeating the original
unseeded input creates another sample; use the replay configuration for its
effective settings. Seed-based regeneration requires matching software and
numerical dependencies. Identical optimized trajectories across different
platforms are not guaranteed.

Load the original coordinates exactly without regeneration:

```python
from spherepackgen import load_snapshot, run_generation

bundle = run_generation("my_case.yaml")
print(bundle.readiness)
particles = load_snapshot(bundle.config.output.path)
centers_um = particles.positions * (bundle.resolved.mean_diameter_m / 1e-6)
radii_um = particles.radii * (bundle.resolved.mean_diameter_m / 1e-6)
```

`load_snapshot` verifies the snapshot checksum. Its coordinates and radii
remain dimensionless until you convert them.

## 9. Connect to an optical solver

Import `particles_real_units.csv` and the dimensions from `metadata.json`.
Create each sphere at `(x, y, z)` using its `radius`. Assign materials and
optical properties in your solver, including any size-based material mapping.
Custom-file input does not perform this assignment.

A periodic sphere may cross a box face. Represent the periodic images or
appropriate clipping in your solver; do not discard it merely because its
surface extends outside the cell. The generated depth direction is also
periodic. If you model a finite slab with open entrance/exit faces, construct
that slab and its boundaries explicitly and check the resulting interfaces.

Geometry validity and structural statistics do not establish an optical
response. Set mesh resolution, illumination, wavelength and boundaries in
your optical solver and check numerical convergence there.

## 10. Troubleshooting

| Symptom | Action |
|---|---|
| `spherepackgen` command not found | Activate the environment or use its full executable path |
| Windows blocks activation | Run `.\.venv\Scripts\spherepackgen.exe` directly |
| Port is busy | Start with `--port 8502` |
| Particle count conflicts with density | Select `auto`, or make count, size, dimensions and density consistent |
| Own periodic-image overlap | Increase the shortest box dimension or reduce particle size |
| Crystal dimensions do not fit | Apply compatible crystal dimensions in the GUI |
| Placement fails or generation times out | Review the density/distribution, choose another algorithm or increase the appropriate budget |
| Missing analysis files | Enable the corresponding computation; plot settings are separate |
| Custom sizes differ from input values | Check sampling, `file_values` and normalization to the requested mean |
| Replay checksum fails | Re-extract the intact bundle and avoid editing archived inputs |

For unresolved problems, open a
[GitHub issue](https://github.com/Zhenpeng04/OptSpherePackGen/issues) with the
configuration, seed, versions, operating system and error/validation report.
Inspect result bundles for private paths or data before sharing.

See [README](README.md), [example configurations](configs/examples/) and
[the MIT License](LICENSE).
