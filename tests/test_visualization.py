import numpy as np

from spherepackgen.domain.particles import ParticleSet
from spherepackgen.visualization.packing_views import (
    DEFAULT_MAX_RENDER_PARTICLES,
    DEFAULT_RENDER_VIEWS,
    PYVISTA_PHI_RESOLUTION,
    PYVISTA_THETA_RESOLUTION,
    _normalized_views,
    _remove_stale_view_files,
    _sample_particles_for_render,
)


def test_render_particle_sampling_preserves_original_particle_set():
    particles = ParticleSet(
        positions=np.arange(30, dtype=float).reshape(10, 3),
        radii=np.linspace(0.1, 1.0, 10),
        box_length=10.0,
        species_id=np.arange(10),
        material_id=np.arange(10) + 100,
    )

    sampled = _sample_particles_for_render(particles, max_render_particles=4)

    assert particles.n_particles == 10
    assert sampled.n_particles == 4
    assert np.array_equal(sampled.positions[0], particles.positions[0])
    assert np.array_equal(sampled.positions[-1], particles.positions[-1])
    assert np.array_equal(sampled.species_id, np.array([0, 3, 6, 9]))


def test_render_defaults_use_single_lower_resolution_perspective_view():
    assert DEFAULT_MAX_RENDER_PARTICLES == 1000
    assert DEFAULT_RENDER_VIEWS == ("perspective",)
    assert PYVISTA_THETA_RESOLUTION == 12
    assert PYVISTA_PHI_RESOLUTION == 8
    assert _normalized_views(("diagonal", "perspective")) == ("perspective",)

def test_render_cleanup_removes_stale_unrequested_views(tmp_path):
    for name in ["packing_front.png", "packing_top.png", "packing_diagonal.png", "packing_perspective.png"]:
        (tmp_path / name).write_text("stale", encoding="utf-8")

    _remove_stale_view_files(tmp_path, "packing", ("perspective",))

    assert (tmp_path / "packing_perspective.png").exists()
    assert not (tmp_path / "packing_front.png").exists()
    assert not (tmp_path / "packing_top.png").exists()
    assert not (tmp_path / "packing_diagonal.png").exists()
