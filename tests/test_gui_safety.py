import numpy as np
import pytest

from spherepackgen.generators.radii import _truncated_normal
from spherepackgen.gui.streamlit_app import _preview_truncated_normal


@pytest.mark.parametrize("sampler", [_truncated_normal, _preview_truncated_normal])
def test_invalid_truncation_fails_without_sampling(sampler):
    with pytest.raises(ValueError, match="min < max"):
        sampler(np.random.default_rng(1), .5, .01, .6, .4, 20)
    with pytest.raises(ValueError, match="CV=0"):
        sampler(np.random.default_rng(1), .5, 0, .6, .7, 20)
    assert np.all(sampler(np.random.default_rng(1), .5, 0, .4, .6, 20) == .5)


def test_rare_truncation_uses_bounded_fallback():
    values = _truncated_normal(np.random.default_rng(1), .5, .01, .7, .71, 20)
    assert np.all((values >= .7) & (values <= .71))


def test_missing_and_explicit_zero_quasi_cv_have_distinct_meaning():
    from spherepackgen.config.loader import config_from_mapping
    from spherepackgen.generators.radii import sample_radii
    raw = {"structure": {"size_distribution": "quasi_monodisperse"},
           "physical": {"mean_diameter": "0.2 um", "packing_fraction": .1},
           "particles": {"num_particles": 1000}, "size_distribution": {}}
    config = config_from_mapping(raw)
    radii, _ = sample_radii(config, 1000, np.random.default_rng(1))
    assert np.std(radii) > 0
    raw["size_distribution"]["CV_radius"] = 0
    radii, _ = sample_radii(config_from_mapping(raw), 1000, np.random.default_rng(1))
    assert np.all(radii == .5)


def test_progress_update_tolerates_windows_reader_lock(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from spherepackgen.gui.worker import report
    original = Path.replace
    calls = []

    def replace(path, target):
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError("Reader holds a temporary lock")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    report(tmp_path, "generation")
    assert json.loads((tmp_path / "gui_progress.json").read_text())["stage"] == "generation"
    assert len(calls) == 3
