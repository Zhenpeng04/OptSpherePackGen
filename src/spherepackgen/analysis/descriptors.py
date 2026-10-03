"""P0 structural descriptors."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from spherepackgen.config.schema import AnalysisConfig
from spherepackgen.domain.particles import ParticleSet
from spherepackgen.domain.result import StructuralDescriptors


def _periodic_tree(particles: ParticleSet) -> cKDTree:
    positions = np.mod(np.asarray(particles.positions, dtype=float), particles.box_length)
    return cKDTree(positions, boxsize=particles.box_lengths)


def _query_nearest(tree: cKDTree, positions: np.ndarray) -> np.ndarray:
    try:
        distances, _ = tree.query(positions, k=2, workers=-1)
    except TypeError:
        distances, _ = tree.query(positions, k=2)
    return np.asarray(distances, dtype=float)


def nearest_neighbor_distribution(particles: ParticleSet) -> dict:
    n = particles.n_particles
    if n == 0:
        nearest = np.empty(0, dtype=float)
    elif n == 1:
        nearest = np.full(1, np.inf, dtype=float)
    else:
        positions = np.mod(np.asarray(particles.positions, dtype=float), particles.box_length)
        distances = _query_nearest(cKDTree(positions, boxsize=particles.box_lengths), positions)
        nearest = distances[:, 1]
    return {
        "distances": nearest,
        "min": float(np.min(nearest)) if n else float("nan"),
        "mean": float(np.mean(nearest)) if n else float("nan"),
        "max": float(np.max(nearest)) if n else float("nan"),
        "backend": "periodic_ckdtree",
    }


def pair_correlation(particles: ParticleSet, bins: int = 80) -> dict:
    r_max = float(np.min(particles.box_lengths)) / 2.0
    edges = np.linspace(0.0, r_max, int(bins) + 1)
    n = particles.n_particles
    if n < 2:
        counts = np.zeros(int(bins), dtype=int)
    else:
        tree = _periodic_tree(particles)
        count_edges = edges.copy()
        if len(count_edges) > 2:
            count_edges[1:-1] = np.nextafter(count_edges[1:-1], -np.inf)
        raw_counts = np.asarray(tree.count_neighbors(tree, count_edges, cumulative=False), dtype=np.int64)
        counts = raw_counts[1:].astype(np.float64) * 0.5
        zero_distance_duplicate_counts = max(int(raw_counts[0]) - n, 0)
        if len(counts):
            counts[0] += 0.5 * zero_distance_duplicate_counts
        counts = np.rint(counts).astype(np.int64)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = 4.0 * np.pi * r**2 * np.diff(edges)
    rho = particles.n_particles / particles.domain.volume
    expected = 0.5 * particles.n_particles * rho * shell_volumes
    g2 = np.divide(counts, expected, out=np.zeros_like(r, dtype=float), where=expected > 0)
    return {"r": r, "g2": g2, "counts": counts, "backend": "periodic_ckdtree_count_neighbors"}


def structure_factor(particles: ParticleSet, k_max_index: int = 5) -> dict:
    positions = particles.positions
    box = particles.box_lengths
    n = particles.n_particles
    ks = []
    values = []
    for nx in range(-k_max_index, k_max_index + 1):
        for ny in range(-k_max_index, k_max_index + 1):
            for nz in range(-k_max_index, k_max_index + 1):
                if nx == ny == nz == 0:
                    continue
                vec_int = np.array([nx, ny, nz], dtype=float)
                k_vec = 2.0 * np.pi * vec_int / box
                phase = positions @ k_vec
                rho_k = np.sum(np.exp(-1j * phase))
                sk = (abs(rho_k) ** 2) / n
                ks.append(float(np.linalg.norm(k_vec)))
                values.append(float(sk.real))
    ks = np.asarray(ks, dtype=float)
    values = np.asarray(values, dtype=float)
    shells = np.round(ks, decimals=10)
    unique_shells = np.unique(shells)
    shell_k = []
    shell_sk = []
    shell_count = []
    for shell in unique_shells:
        mask = shells == shell
        shell_k.append(float(np.mean(ks[mask])))
        shell_sk.append(float(np.mean(values[mask])))
        shell_count.append(int(np.sum(mask)))
    return {
        "k": np.asarray(shell_k, dtype=float),
        "Sk": np.asarray(shell_sk, dtype=float),
        "shell_count": np.asarray(shell_count, dtype=int),
        "k_min": float(np.min(ks)) if len(ks) else float("nan"),
    }


def analyze_particle_set(particles: ParticleSet, config: AnalysisConfig) -> StructuralDescriptors:
    nearest = nearest_neighbor_distribution(particles)
    g2 = pair_correlation(particles, bins=config.bins) if config.compute_g2 else {}
    sk = structure_factor(particles, k_max_index=config.k_max_index) if config.compute_Sk else {}
    return StructuralDescriptors(nearest_neighbor=nearest, g2=g2, Sk=sk)
