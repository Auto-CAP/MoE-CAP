# SPDX-License-Identifier: Apache-2.0
# vLLM MoE Expert Selection Hooks
#
# This module provides hooks and utilities for MoE modules to report
# expert selections to the expert distribution recorder.

from typing import Optional

import torch

from expert_distribution_recorder import (
    get_global_expert_distribution_recorder,
)

logger = None  # Will be set by importing module


class MoEExpertSelectionHookMixin:
    """Mixin for MoE modules to report expert selections to the recorder.

    Usage:
        class MyMoELayer(MoEExpertSelectionHookMixin, nn.Module):
            def __init__(self, ...):
                super().__init__()
                self.layer_idx = layer_idx  # REQUIRED: set by parent

            def forward(self, hidden_states, ...):
                ...
                topk_ids = top_k_gating(hidden_states)  # shape: (num_tokens, topk)
                self.report_expert_selection(topk_ids)
                ...
    """

    # Subclasses should set this
    layer_idx: Optional[int] = None

    def report_expert_selection(self, topk_ids: torch.Tensor) -> None:
        """Report expert selection to the global recorder.

        Args:
            topk_ids: Selected expert indices (shape: [num_tokens, topk])
        """
        if self.layer_idx is None:
            return

        recorder = get_global_expert_distribution_recorder()
        with recorder.with_current_layer(self.layer_idx):
            recorder.on_select_experts(layer_idx=self.layer_idx, topk_ids=topk_ids)

    def report_expert_dispatch(self, num_tokens_per_expert: torch.Tensor) -> None:
        """Report expert dispatch information to the global recorder.

        This is useful for distributed expert scenarios (e.g., DeepEP).

        Args:
            num_tokens_per_expert: Token count per expert
        """
        if self.layer_idx is None:
            return

        recorder = get_global_expert_distribution_recorder()
        with recorder.with_current_layer(self.layer_idx):
            recorder.on_expert_dispatch(
                layer_idx=self.layer_idx,
                num_tokens_per_expert=num_tokens_per_expert,
            )


# Helper function for easy integration into existing MoE modules
def report_expert_selection(
    layer_idx: int,
    topk_ids: torch.Tensor,
) -> None:
    """Convenience function to report expert selection.

    Can be called from any MoE module without inheritance.

    Args:
        layer_idx: Index of the current layer
        topk_ids: Selected expert indices (shape: [num_tokens, topk])
    """
    if layer_idx is None:
        return

    recorder = get_global_expert_distribution_recorder()
    with recorder.with_current_layer(layer_idx):
        recorder.on_select_experts(layer_idx=layer_idx, topk_ids=topk_ids)


# torch.compile compatible recording function
def record_expert_selection_atomic(
    layer_idx: int,
    topk_ids: torch.Tensor,
) -> None:
    """Record expert selection using torch.compile-compatible operations.

    This function can be called directly from MoE layers during forward pass
    without Python callbacks that break torch.compile.

    Args:
        layer_idx: Index of the current layer
        topk_ids: Selected expert indices (shape: [num_tokens, topk])
    """
    from expert_distribution_recorder import (
        get_global_expert_counts_buffer,
        record_expert_selections_atomic,
    )

    buffer = get_global_expert_counts_buffer()
    if buffer is not None:
        record_expert_selections_atomic(buffer, layer_idx, topk_ids)


def report_expert_dispatch(
    layer_idx: int,
    num_tokens_per_expert: torch.Tensor,
) -> None:
    """Convenience function to report expert dispatch information.

    Can be called from any MoE module without inheritance.

    Args:
        layer_idx: Index of the current layer
        num_tokens_per_expert: Token count per expert
    """
    if layer_idx is None:
        return

    recorder = get_global_expert_distribution_recorder()
    with recorder.with_current_layer(layer_idx):
        recorder.on_expert_dispatch(
            layer_idx=layer_idx,
            num_tokens_per_expert=num_tokens_per_expert,
        )