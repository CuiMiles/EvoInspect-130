from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True)
class DirectionBank:
    vectors: torch.Tensor
    real_count: int
    synthetic_count: int


def spatial_grid(height: int, width: int, *, device: torch.device | None = None) -> torch.Tensor:
    if height < 1 or width < 1:
        raise ValueError("spatial dimensions must be positive")
    rows = torch.linspace(0.0, 1.0, height, device=device)
    columns = torch.linspace(0.0, 1.0, width, device=device)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((row_grid, column_grid), dim=-1).reshape(-1, 2)


def conditioned_nearest(
    queries: torch.Tensor,
    memory: torch.Tensor,
    query_coordinates: torch.Tensor,
    memory_coordinates: torch.Tensor,
    spatial_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if queries.ndim != 2 or memory.ndim != 2 or queries.shape[1] != memory.shape[1]:
        raise ValueError("queries and memory must be aligned two-dimensional feature tensors")
    if query_coordinates.shape != (queries.shape[0], 2):
        raise ValueError("query coordinate shape mismatch")
    if memory_coordinates.shape != (memory.shape[0], 2):
        raise ValueError("memory coordinate shape mismatch")
    if spatial_weight < 0:
        raise ValueError("spatial_weight must be non-negative")
    feature_distance = torch.cdist(queries.unsqueeze(0), memory.unsqueeze(0)).squeeze(0)
    if spatial_weight:
        spatial_distance = torch.cdist(
            query_coordinates.unsqueeze(0), memory_coordinates.unsqueeze(0)
        ).squeeze(0)
        feature_distance = feature_distance + spatial_weight * spatial_distance
    return feature_distance.min(dim=1)


def build_direction_bank(
    anomaly_patch_sets: list[torch.Tensor],
    memory: torch.Tensor,
    memory_coordinates: torch.Tensor,
    patch_coordinates: torch.Tensor,
    *,
    top_k_per_image: int,
    spatial_weight: float,
    synthetic_count: int,
    jitter_strength: float,
    seed: int,
) -> DirectionBank:
    if not anomaly_patch_sets:
        raise ValueError("at least one anomaly patch set is required")
    if top_k_per_image < 1 or synthetic_count < 0 or jitter_strength < 0:
        raise ValueError("invalid direction-bank configuration")
    real_directions: list[torch.Tensor] = []
    for patches in anomaly_patch_sets:
        distances, nearest_indices = conditioned_nearest(
            patches,
            memory,
            patch_coordinates,
            memory_coordinates,
            spatial_weight,
        )
        selected = distances.topk(min(top_k_per_image, len(distances))).indices
        deltas = patches[selected] - memory[nearest_indices[selected]]
        real_directions.append(functional.normalize(deltas, dim=1))
    real = torch.cat(real_directions, dim=0)
    if synthetic_count == 0:
        return DirectionBank(real, len(real), 0)

    generator = torch.Generator(device=real.device).manual_seed(seed)
    first = torch.randint(len(real), (synthetic_count,), generator=generator, device=real.device)
    second = torch.randint(len(real), (synthetic_count,), generator=generator, device=real.device)
    mixing = torch.rand(
        synthetic_count, 1, generator=generator, device=real.device, dtype=real.dtype
    )
    mixing = 0.25 + 0.5 * mixing
    mixed = mixing * real[first] + (1.0 - mixing) * real[second]
    if jitter_strength:
        noise = torch.randn(
            mixed.shape, generator=generator, device=real.device, dtype=real.dtype
        )
        mixed = mixed + jitter_strength * functional.normalize(noise, dim=1)
    synthetic = functional.normalize(mixed, dim=1)
    return DirectionBank(torch.cat((real, synthetic), dim=0), len(real), len(synthetic))


def heteromemory_scores(
    patches: torch.Tensor,
    memory: torch.Tensor,
    memory_coordinates: torch.Tensor,
    patch_coordinates: torch.Tensor,
    directions: torch.Tensor,
    *,
    spatial_weight: float,
    direction_weight: float,
    query_top_k: int | None = None,
) -> tuple[float, float, float]:
    if direction_weight < 0:
        raise ValueError("direction_weight must be non-negative")
    if query_top_k is not None and query_top_k < 1:
        raise ValueError("query_top_k must be positive when provided")
    distances, nearest_indices = conditioned_nearest(
        patches,
        memory,
        patch_coordinates,
        memory_coordinates,
        spatial_weight,
    )
    selected = (
        distances.topk(min(query_top_k, len(distances))).indices
        if query_top_k is not None
        else torch.arange(len(distances), device=distances.device)
    )
    deltas = functional.normalize(patches[selected] - memory[nearest_indices[selected]], dim=1)
    affinity = torch.relu(deltas @ directions.T).max(dim=1).values
    normal_score = distances.max()
    directed_score = (distances[selected] * affinity).max()
    fused_score = normal_score + direction_weight * directed_score
    return float(normal_score.item()), float(directed_score.item()), float(fused_score.item())
