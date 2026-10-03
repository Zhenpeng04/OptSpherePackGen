# OptSpherePackGen

English | [简体中文](README.zh-CN.md)

OptSpherePackGen provides the `spherepackgen` Python package for generating,
validating, analyzing and exporting three-dimensional microsphere structures
for optical scattering and electromagnetic simulation.

Choose the particle-size distribution, mean diameter, volume fraction and
periodic box dimensions, then generate structures through a local graphical
interface, YAML configuration or Python API.

## Supported structures

| Structure | Available options |
|---|---|
| Periodic crystals | SC, BCC, FCC and diamond cubic in compatible fixed boxes; HCP through cubic configurations with automatically resolved dimensions |
| Non-overlapping random packings | Poisson-disk placement, random sequential adsorption (RSA), force-biased relaxation and native Lubachevsky–Stillinger (LS) compression |
| Overlapping random media | Marked Poisson Boolean spheres with independent centers |
| Particle sizes | Monodisperse, quasi-monodisperse, lognormal, truncated normal, uniform, gamma, Weibull, custom size files and discrete mixtures |

The GUI provides the main size distributions and structure choices. Discrete
mixtures, imported distributions and the Metropolis sampling route are
configured through YAML or the Python API. SHU and external RCPGenerator
are not included.

## Install and start

Use **Python 3.10 or newer** on Windows, macOS or Linux. Download or clone the
repository and open a terminal in its directory. Installation downloads the
required dependencies; Anaconda is not required.

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

These commands work without activating the environment. To use the shorter
commands below, activate it with `.\.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on macOS/Linux. If Windows blocks script activation,
continue using the full executable paths above.

You can also install a supplied wheel:

```text
python -m pip install /path/to/spherepackgen-0.2.0-py3-none-any.whl
```

## Generate your first structure

In the GUI, select the particle-size distribution and structure type, enter
the mean diameter, volume fraction, lateral length and depth, then select
**Generate**. Results include geometry validation, structural statistics,
coordinate downloads and a complete result bundle.

For a command-line run:

```text
spherepackgen examples --output examples
spherepackgen run examples/fcc.yaml
```

Examples are included in the installed package. To generate a fixed rectangular
random packing, save this as `my_case.yaml`:

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
algorithm:
  name: default
analysis:
  compute_g2: true
  compute_Sk: true
  k_max_index: 3
output:
  path: results/my_case
  coordinate_unit: um
  formats: [csv, json]
  make_plots: true
  overwrite: false
runtime:
  random_seed: 12345
```

Run `spherepackgen run my_case.yaml`. For non-overlapping random structures,
automatic selection uses Poisson-disk at `phi <= 0.18`, RSA at
`0.18 < phi <= 0.34`, and force-biased relaxation at higher fractions.
You can explicitly select another supported algorithm.

## Use the results in optical simulation

`particles_real_units.csv` contains particle centers, radii and diameters
in micrometers by default. `metadata.json` records the box dimensions, actual
volume fraction, units and validation results. Optional HDF5 output and
`particles_snapshot.npz` store dimensionless geometry.

Use those dimensions and coordinates to construct spheres in your simulation
software. Assign materials, refractive indices, wavelength, illumination and
solver boundary conditions there. The package generates geometry and structural
statistics; optical spectra are calculated by your electromagnetic solver.

## Save and reproduce a run

The GUI maintains results and history in the chosen workspace. Each GUI run
has its own directory; changing parameters preserves the previous result.
Generation supports cancellation and a time limit. Use `--port 8502` to select
another port or `--headless` to start without opening a browser.

Every run includes a recorded seed, archived custom size inputs, checksums,
an exact geometry snapshot and `run_bundle.zip`. Extract the bundle and run:

```text
spherepackgen run /path/to/extracted/config_replay.yaml
```

Replay writes to a fresh output directory. To read the stored geometry directly:

```python
from spherepackgen import load_snapshot, run_generation

bundle = run_generation("my_case.yaml")
print(bundle.readiness)
particles = load_snapshot(bundle.config.output.path)
```

Seed-based regeneration requires matching software and numerical dependencies.
Use the snapshot when you need the original coordinates exactly.

## Modeling and validation

- All three box axes are periodic, including depth. The fixed box is
  `Lx = Ly = length`, `Lz = depth`; it represents a repeating medium.
- Fixed dimensions are preserved. Integer particle counts can produce a small,
  reported difference between requested and actual volume fractions.
- Crystals require complete, unstrained unit cells. The GUI can apply compatible
  dimensions; fixed-box HCP is not supported.
- For overlapping media, covered fraction and summed sphere volume fraction
  are different quantities. Reported Boolean coverage is a model expectation,
  rather than a measured union volume.
- Near-jamming convergence depends on the distribution, algorithm and budget.
  Valid geometry alone does not establish isotropy, equilibrium or maximally
  random jamming.
- Three-dimensional preview images display at most 1,000 particles. Coordinate
  exports and validation use the complete structure.

CLI exit code 0 indicates valid geometry and successful generator status.
Codes 1, 2 and 3 distinguish input/generation errors, validation failure and
valid candidates requiring review. See the guide before using
`--allow-candidate` in automated workflows.

## Documentation and support

- [User guide (English)](USER_GUIDE.md)
- [用户指南（简体中文）](USER_GUIDE.zh-CN.md)
- [Example configurations](configs/examples/)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Report a problem](https://github.com/Zhenpeng04/OptSpherePackGen/issues)

Distributed under the [MIT License](LICENSE).
