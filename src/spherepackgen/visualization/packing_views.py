"""Render packing view PNGs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from spherepackgen.domain.particles import ParticleSet

SPHERE_COLOR = "#ADD8E6"
DEFAULT_MAX_RENDER_PARTICLES = 1000
DEFAULT_RENDER_VIEWS = ("perspective",)
PYVISTA_THETA_RESOLUTION = 12
PYVISTA_PHI_RESOLUTION = 8
MATPLOTLIB_SPHERE_U_POINTS = 12
MATPLOTLIB_SPHERE_V_POINTS = 6
PACKING_VIEW_NAMES = ("front", "top", "diagonal", "perspective")


def render_packing_views(
    particles: ParticleSet,
    output_dir: Path,
    *,
    basename: str = "packing",
    window_size: tuple[int, int] = (1000, 1000),
    max_render_particles: int = DEFAULT_MAX_RENDER_PARTICLES,
    views: Iterable[str] = DEFAULT_RENDER_VIEWS,
) -> list[str]:
    """Render default perspective packing-view PNGs.

    Additional views can be requested explicitly with ``views=("front", "top", "perspective")``.
    The legacy view name ``diagonal`` is accepted as an alias for ``perspective``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    render_particles = _sample_particles_for_render(particles, max_render_particles)
    view_names = _normalized_views(views)
    _remove_stale_view_files(output_dir, basename, view_names)
    try:
        return _render_with_pyvista(render_particles, output_dir, basename=basename, window_size=window_size, views=view_names)
    except Exception:
        return _render_with_matplotlib(render_particles, output_dir, basename=basename, views=view_names)


def _normalized_views(views: Iterable[str]) -> tuple[str, ...]:
    aliases = {"diagonal": "perspective"}
    normalized: list[str] = []
    for view in views:
        key = aliases.get(str(view).strip().lower(), str(view).strip().lower())
        if key not in {"front", "top", "perspective"}:
            raise ValueError(f"Unknown packing view {view!r}. Allowed views: front, top, perspective.")
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized) or DEFAULT_RENDER_VIEWS


def _remove_stale_view_files(output_dir: Path, basename: str, requested_views: tuple[str, ...]) -> None:
    requested = {f"{basename}_{view}.png" for view in requested_views}
    for view in PACKING_VIEW_NAMES:
        path = output_dir / f"{basename}_{view}.png"
        if path.name not in requested:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _sample_particles_for_render(particles: ParticleSet, max_render_particles: int) -> ParticleSet:
    max_render_particles = int(max_render_particles)
    if max_render_particles <= 0 or particles.n_particles <= max_render_particles:
        return particles
    indices = np.linspace(0, particles.n_particles - 1, max_render_particles, dtype=int)
    return ParticleSet(
        positions=particles.positions[indices],
        radii=particles.radii[indices],
        box_length=particles.box_length,
        species_id=particles.species_id[indices],
        material_id=particles.material_id[indices],
    )


def _view_camera_positions(particles: ParticleSet) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[int, int, int]]]:
    center = particles.box_lengths / 2.0
    distance = float(np.max(particles.box_lengths)) * 2.4
    return {
        "front": ((center[0], center[1] - distance, center[2]), tuple(center), (0, 0, 1)),
        "top": ((center[0], center[1], center[2] + distance), tuple(center), (0, 1, 0)),
        "perspective": ((center[0] + distance, center[1] - distance, center[2] + distance), tuple(center), (0, 0, 1)),
    }


def _render_with_pyvista(
    particles: ParticleSet,
    output_dir: Path,
    *,
    basename: str,
    window_size: tuple[int, int],
    views: tuple[str, ...],
) -> list[str]:
    import pyvista as pv

    pv.OFF_SCREEN = True
    camera_positions = _view_camera_positions(particles)
    files: list[str] = []
    for view_name in views:
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background("white")
        plotter.enable_lightkit()
        for position, radius in zip(particles.positions, particles.radii):
            mesh = pv.Sphere(
                radius=float(radius),
                center=tuple(float(x) for x in position),
                theta_resolution=PYVISTA_THETA_RESOLUTION,
                phi_resolution=PYVISTA_PHI_RESOLUTION,
            )
            plotter.add_mesh(mesh, color=SPHERE_COLOR, smooth_shading=True, specular=0.18, roughness=0.65)
        plotter.camera_position = camera_positions[view_name]
        if view_name == "perspective":
            plotter.camera.parallel_projection = False
            plotter.camera.view_angle = 28.0
        else:
            plotter.camera.parallel_projection = True
            plotter.camera.parallel_scale = float(np.max(particles.box_lengths)) * 0.72
        plotter.hide_axes()
        path = output_dir / f"{basename}_{view_name}.png"
        plotter.screenshot(str(path), transparent_background=False)
        plotter.close()
        files.append(str(path))
    return files


def _render_with_matplotlib(particles: ParticleSet, output_dir: Path, *, basename: str, views: tuple[str, ...]) -> list[str]:
    import matplotlib.pyplot as plt

    view_angles = {
        "front": (0, 0),
        "top": (90, -90),
        "perspective": (28, -45),
    }
    sphere_u = np.linspace(0, 2 * np.pi, MATPLOTLIB_SPHERE_U_POINTS)
    sphere_v = np.linspace(0, np.pi, MATPLOTLIB_SPHERE_V_POINTS)
    files = []
    for view_name in views:
        elev, azim = view_angles[view_name]
        fig = plt.figure(figsize=(7, 7), facecolor="white")
        ax = fig.add_subplot(111, projection="3d", facecolor="white")
        for center, radius in zip(particles.positions, particles.radii):
            x = center[0] + radius * np.outer(np.cos(sphere_u), np.sin(sphere_v))
            y = center[1] + radius * np.outer(np.sin(sphere_u), np.sin(sphere_v))
            z = center[2] + radius * np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
            ax.plot_surface(x, y, z, color=SPHERE_COLOR, linewidth=0, antialiased=False, shade=True)
        ax.set_xlim(0, particles.box_lengths[0])
        ax.set_ylim(0, particles.box_lengths[1])
        ax.set_zlim(0, particles.box_lengths[2])
        ax.set_box_aspect(particles.box_lengths)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_proj_type("persp" if view_name == "perspective" else "ortho")
        path = output_dir / f"{basename}_{view_name}.png"
        plt.tight_layout(pad=0)
        plt.savefig(path, dpi=140, facecolor="white", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        files.append(str(path))
    return files
