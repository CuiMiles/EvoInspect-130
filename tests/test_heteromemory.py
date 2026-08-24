from __future__ import annotations

import unittest

import torch

from evoinspect.heteromemory import (
    build_direction_bank,
    conditioned_nearest,
    heteromemory_scores,
    spatial_grid,
)


class HeteroMemoryTest(unittest.TestCase):
    def test_spatial_condition_changes_nearest_prototype(self) -> None:
        queries = torch.tensor([[0.0, 0.0]])
        memory = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
        query_coordinates = torch.tensor([[1.0, 1.0]])
        memory_coordinates = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        _, index_without_position = conditioned_nearest(
            queries, memory, query_coordinates, memory_coordinates, 0.0
        )
        _, index_with_position = conditioned_nearest(
            queries, memory, query_coordinates, memory_coordinates, 1.0
        )
        self.assertEqual(index_without_position.item(), 0)
        self.assertEqual(index_with_position.item(), 1)

    def test_direction_expansion_is_deterministic_and_normalized(self) -> None:
        memory = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        memory_coordinates = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        coordinates = spatial_grid(1, 2)
        anomaly_patches = [torch.tensor([[0.0, 1.0], [2.0, 1.0]])]
        first = build_direction_bank(
            anomaly_patches,
            memory,
            memory_coordinates,
            coordinates,
            top_k_per_image=2,
            spatial_weight=0.0,
            synthetic_count=4,
            jitter_strength=0.05,
            seed=130,
        )
        second = build_direction_bank(
            anomaly_patches,
            memory,
            memory_coordinates,
            coordinates,
            top_k_per_image=2,
            spatial_weight=0.0,
            synthetic_count=4,
            jitter_strength=0.05,
            seed=130,
        )
        self.assertEqual(first.real_count, 2)
        self.assertEqual(first.synthetic_count, 4)
        self.assertTrue(torch.equal(first.vectors, second.vectors))
        self.assertTrue(torch.allclose(first.vectors.norm(dim=1), torch.ones(6)))

    def test_matching_anomaly_direction_increases_fused_score(self) -> None:
        memory = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        memory_coordinates = spatial_grid(1, 2)
        patches = torch.tensor([[0.0, 0.5], [1.0, 0.0]])
        directions = torch.tensor([[0.0, 1.0]])
        normal, directed, fused = heteromemory_scores(
            patches,
            memory,
            memory_coordinates,
            spatial_grid(1, 2),
            directions,
            spatial_weight=0.0,
            direction_weight=0.5,
        )
        self.assertAlmostEqual(normal, 0.5)
        self.assertAlmostEqual(directed, 0.5)
        self.assertAlmostEqual(fused, 0.75)

    def test_query_top_k_restricts_direction_comparison(self) -> None:
        memory = torch.tensor([[0.0, 0.0]])
        memory_coordinates = torch.tensor([[0.0, 0.0]])
        patches = torch.tensor([[0.0, 0.5], [1.0, 0.0]])
        directions = torch.tensor([[0.0, 1.0]])
        _, all_directed, _ = heteromemory_scores(
            patches,
            memory,
            memory_coordinates,
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            directions,
            spatial_weight=0.0,
            direction_weight=0.5,
        )
        _, top_one_directed, _ = heteromemory_scores(
            patches,
            memory,
            memory_coordinates,
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            directions,
            spatial_weight=0.0,
            direction_weight=0.5,
            query_top_k=1,
        )
        self.assertAlmostEqual(all_directed, 0.5)
        self.assertAlmostEqual(top_one_directed, 0.0)


if __name__ == "__main__":
    unittest.main()
