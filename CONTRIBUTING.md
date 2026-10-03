# Contributing

Use Python 3.10 or newer in a virtual environment. Install development
dependencies with `python -m pip install -e ".[dev]"` and run
`python -m pytest` before proposing changes.

Report problems through GitHub Issues. Include the configuration, random seed,
package/Python versions, operating system and complete error message. For
geometry or convergence issues, attach the validation report and diagnostics.
Remove private paths or proprietary input data before sharing run bundles.

New generators should preserve the requested physical dimensions and size
distribution, report convergence truthfully, and pass periodic geometry
validation. Include tests that compare geometry against independent references
where practical, and document algorithm limits and parameters in USER_GUIDE.md.

Pull requests are tested on Windows, Linux and macOS with Python 3.10 and 3.12,
including installation and GUI checks outside the source checkout.
