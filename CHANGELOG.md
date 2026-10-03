# Changelog

All notable changes to SpherePackGen will be recorded in this file.

The format follows the spirit of Keep a Changelog, and this project uses
semantic versioning once public releases begin.

## [0.2.0] - 2026-10-02

### Removed
- Exploratory SHU generation and external RCPGenerator integration, including
  their GUI controls and example configurations, from the public release.
  Native Lubachevsky-Stillinger and all other implemented routes are retained.

### Added
- Finite geometry and label integrity checks at construction and validation,
  including rejection of empty structures and invalid validation tolerances.
- Strict CLI success semantics with `--allow-candidate` and shared readiness
  flags in summaries, metadata and manifests.
- Installed-package GUI launcher, user-owned workspace/cache/history and three
  packaged examples available through `spherepackgen examples`.
- Recorded actual seeds, archived/hash-verified size inputs, software/environment
  provenance, exact NPZ snapshots and downloadable portable ZIP bundles.
- Portable replay configuration with fresh output directories, plus a checksum
  verified `load_snapshot` API.
- GUI input checks before sampling, bounded truncated-normal sampling and exact
  particle-count previews shared with the backend resolver.
- Separate GUI run directories, explicit output manifests, persistent results
  and downloads, background progress, cancellation and whole-run time limits.
- Compatible crystal-size
  controls, readable algorithm labels and analysis detail presets.
- Responsive parameter groups, a periodic-box schematic, complete historical
  result views and explicit particle-subset labels for rendering.
- Fixed periodic rectangular boxes with independent lateral length and depth;
  legacy cubic configuration behavior is preserved.
- Rectangular geometry for all retained generator routes, periodic neighbor indexing,
  validation, statistical analysis, metadata, HDF5 and rendering.
- Explicit integer-count density allowance, resolved-volume consistency checks,
  and self-periodic-image overlap checks.
- Complete unstrained crystal-cell checks; fixed-box HCP fails explicitly.
- Reusable configuration exports, GUI downloads and separate workflow timings.
- Independent image-based geometry tests, advanced-generator integration checks,
  Streamlit end-to-end tests and bounded stability/scale benchmark tooling.

## [0.1.0] - 2026-07-05

### Added

- P0 / MVP workflow from YAML configuration to parameter resolution, generation,
  validation, analysis, and export.
- Command-line entry point: `spherepackgen run <config.yaml>`.
- Python API entry point through `spherepackgen.api.run_generation`.
- Streamlit GUI prototype for configuring, running, and viewing generated
  packings.
- Periodic crystal, overlapping Boolean, hard-core random, Metropolis,
  force-biased, native Lubachevsky-Stillinger and dense baseline generator routes.
- Radius distributions for monodisperse, quasi-monodisperse, continuous
  polydisperse, discrete mixture, imported, and custom-file use cases.
- Validation reports for boundary checks, packing fraction, radius statistics,
  and periodic non-overlap checks.
- Analysis outputs for nearest-neighbor distances, pair correlation `g2(r)`,
  and shell-averaged structure factor `S(k)`.
- Export of dimensionless and real-unit coordinates, metadata, resolved config,
  HDF5 data when available, analysis CSV files, run logs, and PNG figures.
- Focused test suite covering geometry, radii, workflow, GUI preview helpers,
  generator routing, and selected generator runs.

### Notes

- This release is a P0 / MVP implementation. Some algorithms are exploratory or
  simplified and are documented as such in `README.md` and `USER_GUIDE.md`.
