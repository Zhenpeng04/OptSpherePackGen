# Public release 0.2.0

## Scope

The public version generates periodic crystals, overlapping Boolean media and
non-overlapping random packings. It retains Poisson-disk, RSA, force-biased,
native Lubachevsky-Stillinger and the YAML Metropolis reference route, together
with all supported size distributions, analysis, exports and portable replay.
SHU and external RCPGenerator implementations, controls and examples are removed.
Their old configuration values fail explicitly instead of selecting another route.

All three box axes are periodic. Fixed-box HCP and incommensurate crystals are
rejected. High-density convergence is algorithm-dependent; geometry validation
does not establish MRJ or isotropy. Optical material assignment remains the
downstream user's responsibility, including assignment by size for custom files.

## Validation gates

The final local source suite passes 173 tests on Windows. The installed wheel
passes CLI, snapshot/replay, localhost server and GUI generation/download/history
checks outside the checkout. Wheel and source archives exclude the removed
implementations and all private working files.

Run `python -m pytest` on the final source tree. Build both wheel and source
distribution using `python -m build` in a clean build directory. Install the wheel
in a separate environment and run `python scripts/check_installed_package.py`.
That check exercises packaged examples, CLI generation, snapshot/replay, the
localhost GUI server, GUI generation, downloads and history outside the checkout.

The private repository's GitHub Actions matrix checks Python 3.10 and 3.12 on
Windows, Linux and macOS. Every job must pass tests, build both distributions,
and pass the installed-package check before changing repository visibility.
Use the Actions result for the exact final commit as the release gate.

## Public files and history

Include source, tests, supported example configurations, package metadata,
license, README, user guide, developer scripts and CI configuration.
Exclude generated outputs, local environments, input caches, assistant/session
records, internal reviews, planning documents and third-party binaries.

The remote repository initially contains only its LICENSE commit. The public
source snapshot is based on that commit. Preserve local development history
privately; do not push local development branches or tags with `--all`/`--mirror`.
Before publication, check remote branches/tags and recursively inspect the
source tree and both distribution archives for excluded files and credentials.

Once all gates pass, publish from the verified main branch and use the 0.2.0
version for the source archive and wheel. Add a short repository description and
topics so users can find the project. Changing visibility and publishing a
GitHub Release are separate actions from preparing and synchronizing the source.
