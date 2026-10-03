# GUI robustness and layout revision

Completed on 2026-10-02 following the GUI layout/case audit. Changes were applied
in priority order: input safety, run isolation and persistent results, algorithm
status semantics, then layout and analysis controls.

## Implemented behavior

1. Truncated-normal sampling checks finite ordered bounds and zero CV, retains
   the normal deterministic sampling path, and uses a bounded fallback for rare
   intervals. Missing quasi-monodisperse CV still defaults to 0.03; explicit zero
   now means identical radii.
2. GUI preflight reuses the backend configuration parser and parameter resolver.
   Automatic count previews use the exact radii/count for the selected seed.
   The GUI caps resolved counts at 100,000 with bounded preview allocation.
   Inconsistent manual counts and self-periodic-image overlaps
   block generation before starting a job. Algorithm-density and
   extreme-aspect-ratio concerns produce advisory messages.
3. Each GUI job reserves a separate directory. A run manifest lists only that
   run's exported files and the number of particles used in the rendering.
   Manifest paths use portable separators; readers also accept previous Windows
   separators. Figures from earlier or unlisted exports are not displayed.
4. Results persist across parameter edits and downloads. Current inputs and the
   saved run snapshot are distinguished. Historical runs use the same result
   component with geometry, validation, parameters, figures and downloads.
   Legacy directories lacking a manifest have an explicit warning.
5. Background jobs report stages, have a configurable whole-run time limit,
   and support cancellation. A supervisor process bounds generation independently
   of Python computation in the child. Earlier completed results remain available
   after cancellation/failure. Progress updates tolerate temporary Windows rename
   locks and cannot misclassify an otherwise completed run solely because a final
   progress write was unavailable.
6. Geometry validation remains separately available. Unconverged
   candidates return warning status and are never displayed as complete success.
   Failed validation uses an error indicator.
7. Fixed-input GUI crystals require monodisperse particles and exclude HCP.
   Cell-count controls can apply compatible dimensions.
8. Project/history remain in the sidebar. Main inputs, geometry schematic and
   the generation summary occupy Run. Advanced settings use full-width groups,
   Results and History have separate tabs, and internal algorithm identifiers
   have readable labels. Standard S(k) uses index 10, Quick 5, Detailed 20; the
   actual particle–wavevector workload is shown.
9. Disabled plots do not create missing-figure cards. Particle-subset rendering
   is labeled explicitly. Distribution previews and exported histograms use
   particle diameter and percentage; result histograms use the output physical
   unit. Requested HDF5 export fails explicitly if its dependency is unavailable.

## Verification

- Final full local suite: **127 passed in 65.75 seconds** on Windows/Python 3.13.
- Added regressions cover invalid/rare truncation, missing versus zero CV,
  transient Windows progress locks, exact previews, result retention, unique
  output paths, exclusion of stale figures, unsupported settings, compatible
  crystal controls, candidate status, actual cancellation and supervisor timeout.
- GUI-driven representative runs passed geometry validation:
  Poisson-disk (phi 0.12), RSA (0.25), force-biased (0.60), LS (0.15),
  Boolean (covered fraction 0.70), and SC crystal.
  Preview counts matched generated counts for every route.
- Actual browser acceptance at approximately 903 px width checked the revised
  layout, a 1.2 x 1.2 x 0.6 um polydisperse case, changing seed from 12345 to
  98765, separate run destinations, persistent prior results, rendered figures,
  metadata download without clearing the page, and historical result viewing.
- Existing rectangular geometry/generator and legacy cubic tests remained green.

Evidence from this local run is stored under
`results/gui_audit_20261002/`: `full_regression.log`, `revised_routes.json`,
`revised_advanced.png` and `revised_results.png`. The original audit also remains
there. Generated evidence/results are ignored by Git.

This verification covers representative cases, not every density or aspect
ratio. Near-jamming convergence remains algorithm-dependent. Linux/macOS process
handling is implemented but was not exercised on this Windows host. Remote
multi-platform CI has not been run. Actual custom-file browser upload was not
part of this acceptance run; its parser/backend regression coverage remains.
