# SpherePackGen

SpherePackGen is a Python toolkit for generating, validating, analyzing, and exporting three-dimensional microsphere packing structures for optical scattering media and downstream electromagnetic simulation.

The public 0.2.0 release supports these capabilities:

- configuration-driven runs through YAML files and a Python API
- dimensionless internal geometry with real-unit export in micrometers by default
- periodic rectangular domains `(L, L, D)` with independent lateral length and depth; legacy cubic configurations remain supported
- monodisperse, quasi-monodisperse, continuous polydisperse, custom-file, and discrete mixture radius distributions
- macro-parameter input: mean particle diameter, target packing fraction or Boolean covered volume fraction, lateral length, and depth
- automatic density-class inference from target packing fraction, with a separate nominal state for overlapping Boolean media
- fixed user dimensions with automatic particle-count selection and explicit density-resolution reporting; legacy cubic inputs retain automatic box sizing
- default generator selection based on spatial order and target volume fraction, with optional user override
- Generators: periodic crystals, marked Poisson Boolean overlapping spheres, variable-radius Poisson-disk hard-core placement, RSA hard-core random packing, force-biased dense relaxation, native Lubachevsky-Stillinger event-driven growth, and a developer/reference Metropolis baseline
- validation: boundary checks, volume fraction, radius statistics, and periodic non-overlap checks for hard-core models
- analysis: periodic-tree nearest-neighbor distances, pair correlation `g2(r)`, and shell-averaged structure factor `S(k)`
- export: CSV, JSON metadata, YAML resolved config, optional HDF5, and optional basic plots when matplotlib is available
- default output: CSV and JSON; HDF5 is written only when explicitly selected in `output.formats` or the GUI
- 3D packing-view PNGs: one default perspective view with light-blue particles on a white background

## Rectangular Periodic Boxes

The GUI defaults to a fixed periodic box with two independent inputs: lateral
length (`Lx = Ly`) and depth (`Lz`). The cube is the special case `length = depth`.
For YAML runs use:

```yaml
physical:
  mean_diameter: 200 nm
  packing_fraction: 0.50
domain:
  type: periodic_box
  length: 2 um
  depth: 1 um
particles:
  num_particles: auto
```

See `configs/examples/periodic_rectangular.yaml` for a complete example. All three
axes remain periodic. Fixed dimensions are never adjusted to match density;
integer particle counts may cause a small reported target-density difference.
Manual counts inconsistent with the target are rejected. Old `periodic_cube`
configurations using `medium_thickness` or `box_size` retain their old behavior.

The supported generator routes accept rectangular dimensions. Crystals require complete
unstrained unit cells; fixed-box HCP is explicitly unsupported. Extreme density and
aspect ratios still require convergence checks. Rectangular `g2(r)` is limited to
half the shortest side; reciprocal-space vectors use each axis length.

Outputs include three-axis dimensions in metadata and HDF5, a loadable
`config_replay.yaml`, and separate workflow timings. For cubes, legacy scalar
metadata/HDF5 attributes remain available; rectangular HDF5 uses `box_lengths`.

## Installation

Use Python 3.10 or newer. In a downloaded source checkout, create a virtual
environment and install the package. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
spherepackgen gui
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.
Anaconda is not required. Alternatively install a supplied wheel with
`python -m pip install /path/to/spherepackgen-0.2.0-py3-none-any.whl`.

Once installed, run from any user-owned working directory:

```text
spherepackgen examples --output examples
spherepackgen run examples/fcc.yaml
spherepackgen gui --workdir .
```

Examples are included in the wheel and source distribution. The GUI writes
results under the selected workspace, caches uploads and remembers history in
its `.spherepackgen/` directory. It serves only on localhost; add `--headless`
to skip opening a browser, or `--port 8502` to choose another port.

For development, install the package in editable mode with the test dependency:

```powershell
python -m pip install -e ".[dev]"
```

After this, the command-line entry point is available as:

```powershell
spherepackgen run configs\examples\fcc.yaml
```

After the repository is available on GitHub, users can install directly from it:

```powershell
python -m pip install git+https://github.com/Zhenpeng04/OptSpherePackGen.git
```

PyVista is optional and improves the 3D packing-view rendering path when it is
available. Install it with:

```powershell
python -m pip install ".[visualization]"
```

## Development Checks

Run the test suite with:

```powershell
python -m pytest
```

Run an example configuration without installing the package:

```powershell
$env:PYTHONPATH = "src"
python -m spherepackgen.cli run configs\examples\fcc.yaml
```

For user-facing instructions, see [USER_GUIDE.md](USER_GUIDE.md).

Run the Streamlit GUI prototype:

```powershell
spherepackgen gui --workdir .
```

The GUI groups structure and physical inputs in **Run**, optional settings in
**Advanced**, saved results in **Results**, and past runs in **History**. Particle
count and density previews use the same sampled radii as generation. Each run
uses a separate directory and an output manifest, so changing options does not
mix new coordinates with old images. Results and downloads remain available
when parameters change.

Generation runs in a background process with stage progress, a Cancel button,
and a configurable time limit (600 seconds by default). Quick, Standard and
Detailed S(k) presets use maximum indices 5, 10 and 20 respectively; disabling
plots does not disable analysis. Fixed-box GUI crystals support monodisperse particles
and complete unit cells; use **Apply compatible crystal dimensions** to fit
the chosen cell counts.

After editable installation, the explicit `PYTHONPATH` assignment is not needed.

## Validation and Reproduction

Non-finite coordinates/radii, empty geometry, malformed particle labels and
invalid validation tolerances are rejected before analysis/export.
CLI generation is strict by default: exit 0 means valid geometry and a
successful generator target, exit 2 means failed geometry validation, exit 3
means a valid candidate that did not meet its structure target, and exit 1
means an input or generation error. Use `--allow-candidate` to return exit 0 for
geometrically valid candidates. Their readiness flags remain false in metadata.

Every run records its actual random seed, software/environment information,
checksums, an exact dimensionless `particles_snapshot.npz` and a portable
`run_bundle.zip`. Custom size files are copied into `inputs/` before sampling.
Download the complete result bundle in the GUI to transfer these files.
Extract it and run `spherepackgen run /path/to/config_replay.yaml`; output goes
to a new `replay/` directory next to the configuration, with a unique suffix
on subsequent runs. Original results are preserved.

Load the stored geometry without regeneration with
`spherepackgen.load_snapshot('/path/to/extracted/bundle')`. Its checksum is
verified before loading. Seed-based reruns require matching software and
numerical dependencies.

## Current Defaults and Performance Paths

- Monodisperse non-overlapping random structures now use the unified `hard_core_random` route. Former separate random-structure labels are no longer user-facing structure types; different random states are controlled primarily by packing fraction and algorithm parameters.
- With `algorithm.name: default`, `hard_core_random` selects `poisson_disk` at low packing fraction, `rsa` at moderate packing fraction, and `force_biased` for denser targets.
- The low-density Poisson-disk route uses cell-list local candidate checks instead of scanning every existing particle for each candidate.
- The dense `force_biased` route uses a vectorized Verlet pair cache, dynamic cutoff shrinkage, and vectorized pair-force accumulation; it switches to stronger high-density defaults at `packing_fraction >= 0.60`.
- The ordinary GUI route exposes `force_biased` as the internal middle/high-density non-overlapping random generator. `metropolis` remains available only as an explicit YAML/developer comparison algorithm.
- The ordinary force-biased GUI shows only the major controls: initial radius fraction, radius-growth stages, relaxation steps per stage, and contraction rate. Initializer budgets, outer-shell ratio, force scale, displacement cap, temporary overlap tolerance, Verlet skin, and final cleanup budget stay under advanced controls.
- The native `lubachevsky_stillinger` route is an advanced/manual dynamic-compression generator: reduced non-overlapping spheres move ballistically in a fixed periodic box, radii grow linearly, and elastic hard-sphere collisions are processed as events. Its `initialization: auto` setting uses direct reduced-radius Poisson-disk placement below `packing_fraction = 0.60` and a `force_biased_warm_start` above that point. Its `event_backend: auto` setting uses exhaustive pair scanning for small systems and a Verlet-neighbor priority queue for larger systems.
- Crystal generation uses a vectorized unit-cell grid and validates that fixed particle counts are compatible with complete unit-cell tiling.
- Nearest-neighbor and `g2(r)` analysis use periodic `cKDTree` backends. Packing-view PNG rendering uses a deterministic preview subset of at most 1000 particles and lower mesh resolution by default, while CSV, JSON, validation, and analysis data still use the full particle set.
- In the GUI and config defaults, analysis plots and the single perspective 3D packing-view PNG are enabled by default; HDF5 output remains disabled unless explicitly selected. The default overlap tolerance is `1.0e-5`.

## P0 Boundary

The P0 implementation is intended to establish a reliable vertical workflow: configuration input -> parameter resolution -> structure generation -> validation -> analysis -> export. The force-biased dense route is a simplified Bezrukov/Jodrey-Tory-style relaxation, not a complete replica of the reference C++ implementation. The native Lubachevsky-Stillinger route follows the literature-level LS idea of event-driven hard-sphere dynamics with linear particle growth in a fixed volume, but it is positioned here as an advanced configurable generator rather than the ordinary default path. Its Python implementation prioritizes practical efficiency through an auto-selected initialization path and an auto-selected exhaustive or Verlet-neighbor event backend. Local N=500 checks support `force_biased_warm_start` for `packing_fraction = 0.60` and `0.62`; `0.64` remains experimental and is not treated as production-ready for LS. Advanced algorithms such as event-chain Monte Carlo and Torquato-Jiao are not claimed as complete in P0.

## Open-Source Release Notes

Version 0.2.0 excludes SHU and external RCPGenerator. Old configurations for
those routes are rejected. Native Lubachevsky-Stillinger remains available.
See the [public release checklist](docs/public-release-checklist.md) for scope,
package checks and the required private-repository CI gate.

Generated outputs under `results/`, local scratch folders, Word planning
documents, and optional third-party executables are intentionally excluded from
the first public source release. They can be regenerated locally or documented
separately when needed.

SpherePackGen is distributed under the MIT License. See [LICENSE](LICENSE).
