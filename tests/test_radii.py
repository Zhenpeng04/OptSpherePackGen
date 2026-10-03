import numpy as np
import pytest

from spherepackgen.api import load_config
from spherepackgen.generators.radii import sample_radii


def _write_continuous_config(tmp_path, distribution, extra=""):
    cfg_path = tmp_path / f"{distribution}.yaml"
    cfg_path.write_text(
        f"""
project:
  name: test_{distribution}
structure:
  size_distribution: continuous_polydisperse
  spatial_order: hard_core_random
physical:
  mean_diameter: 0.3 um
  packing_fraction: 0.10
  medium_thickness: 2.0 um
particles:
  num_particles: 32
size_distribution:
  type: continuous_polydisperse
  distribution: {distribution}
  CV_radius: 0.25
  min_radius_factor: 0.20
  max_radius_factor: 2.50
{extra}
algorithm:
  name: default
output:
  path: results/test_{distribution}
  formats: [csv, json]
  make_plots: false
runtime:
  random_seed: 7
""",
        encoding="utf-8",
    )
    return cfg_path


@pytest.mark.parametrize("distribution", ["lognormal", "truncated_normal", "uniform", "gamma", "weibull"])
def test_continuous_polydisperse_distributions_sample_positive_normalized_radii(tmp_path, distribution):
    extra = ""
    if distribution == "gamma":
        extra = "  parameters:\n    shape: 4.0\n"
    if distribution == "weibull":
        extra = "  parameters:\n    shape: 2.5\n"

    config = load_config(_write_continuous_config(tmp_path, distribution, extra))
    radii, species_id = sample_radii(config, 128, np.random.default_rng(123))

    assert radii.shape == (128,)
    assert species_id.shape == (128,)
    assert np.all(np.isfinite(radii))
    assert np.all(radii > 0)
    assert np.mean(radii) == pytest.approx(0.5)


def test_continuous_polydisperse_custom_file_reads_uploaded_particle_sizes(tmp_path):
    size_file = tmp_path / "uploaded_sizes.csv"
    size_file.write_text("diameter_um\n0.20\n0.30\n0.60\n", encoding="utf-8")
    cfg_path = _write_continuous_config(
        tmp_path,
        "custom_file",
        """
  size_file: uploaded_sizes.csv
  file_values: diameter
  file_column: diameter_um
""",
    )

    config = load_config(cfg_path)
    radii, _ = sample_radii(config, 32, np.random.default_rng(123))

    assert np.all(radii > 0)
    assert np.mean(radii) == pytest.approx(0.5)
    assert str(tmp_path) in str(config.size_distribution.size_file)
