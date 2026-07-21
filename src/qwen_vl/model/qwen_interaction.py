"""Geometry interaction modules for VGGT and DA3-based variants."""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import is_flash_attn_2_available

from .da3_adapter import DA3Projector
from .geometry_bank import BankFusionBlock, BankRouter
from .geometry_bank.router_stats import get_router_stats_collector
from .stage2 import CompetitiveBridgeGate, LocalLevelRouter
from .msgf_memory import BiDirectionalMemoryBank, HierarchicalMemoryBank, MemoryRefiner, _similarity
from .msgf_utils import FrameLayout, compute_stage_ranges, infer_frame_layout, mean_pool_tokens, safe_topk, split_by_layout
from .mmr_memory import FrameMemoryBank, RegionMemoryBank
from .mmr_retriever import QueryDrivenMMRRetriever
from .mmr_utils import compute_mmr_stage_ranges

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func
else:
    flash_attn_func = None


class QwenVGGTInteractionv1(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.geo_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        self.layer_idx = layer_idx

    def forward(self, image_hidden_states, vggt_features, **kwargs):
        q_input = self.input_layernorm(image_hidden_states)
        kv_input = self.geo_layernorm(vggt_features)

        q = self.q_proj(q_input).view(q_input.shape[0], q_input.shape[1], self.num_heads, self.head_dim)
        k = self.k_proj(kv_input).view(kv_input.shape[0], kv_input.shape[1], self.num_heads, self.head_dim)
        v = self.v_proj(kv_input).view(kv_input.shape[0], kv_input.shape[1], self.num_heads, self.head_dim)

        if flash_attn_func is not None:
            attn_output = flash_attn_func(q, k, v, causal=False)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(q, k, v)
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(image_hidden_states.shape)
        attn_output = self.o_proj(attn_output)
        return torch.tanh(self.gate) * attn_output


class _FramewiseGeometryInteraction(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        use_spatial_bias: bool = True,
        use_importance_gate: bool = True,
        geo_learn_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = 0 if layer_idx is None else int(layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.use_spatial_bias = use_spatial_bias
        self.learnable_bias = geo_learn_bias
        self.use_importance_gate = use_importance_gate
        self.msgf_debug = bool(getattr(config, "msgf_debug", False))

        if hasattr(config, "vision_config"):
            self.pooling_stride = config.vision_config.spatial_merge_size
        else:
            self.pooling_stride = 2

        if kwargs.pop("depart_smi_token", False):
            self.pooling_stride *= kwargs.pop("smi_downsample_rate", 2)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.geo_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

        if self.use_importance_gate:
            self.importance_net = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 4, 1),
                nn.Sigmoid(),
            )
        else:
            self.importance_net = None

        self.bias_gate = nn.Sequential(
            nn.Linear(config.hidden_size, max(config.num_attention_heads // 2, 1)),
            nn.Sigmoid(),
        )

        memory_heads = max(1, min(self.num_heads, 8))
        self.memory_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=memory_heads,
            batch_first=True,
        )
        self.memory_gate = nn.Parameter(torch.zeros(1))
        self.geo_projector = DA3Projector(config.hidden_size, config.hidden_size)
        self.query_mixer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def get_spatial_bias(self, h_feat: int, w_feat: int, device, dtype) -> torch.Tensor:
        y = torch.arange(h_feat, device=device, dtype=dtype)
        x = torch.arange(w_feat, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        dist_matrix = torch.cdist(coords, coords, p=2)
        diag_len = math.sqrt(h_feat**2 + w_feat**2) + 1e-6
        return -(dist_matrix / diag_len)

    def _build_layout(self, total_tokens: int, grid_thw: Optional[torch.Tensor]) -> FrameLayout:
        return infer_frame_layout(total_tokens=total_tokens, grid_thw=grid_thw, pooling_stride=self.pooling_stride)

    def _split_frames(self, hidden_states: torch.Tensor, layout: FrameLayout) -> List[torch.Tensor]:
        return split_by_layout(hidden_states, layout)

    def _build_attn_bias(
        self,
        q_input: torch.Tensor,
        k_input: torch.Tensor,
        frame_shape: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        q_len = q_input.shape[1]
        k_len = k_input.shape[1]
        attn_bias = None

        if self.use_spatial_bias and q_len == k_len and frame_shape[0] * frame_shape[1] == q_len:
            spatial_bias = self.get_spatial_bias(frame_shape[0], frame_shape[1], q_input.device, q_input.dtype)
            gate_score = self.bias_gate(k_input).transpose(1, 2).unsqueeze(-1)
            head_bias = torch.zeros(
                q_input.shape[0], self.num_heads, q_len, k_len, device=q_input.device, dtype=q_input.dtype
            )
            split_idx = max(self.num_heads // 2, 1)
            head_bias[:, :split_idx] = spatial_bias.unsqueeze(0).unsqueeze(0) * gate_score[:, :split_idx]
            attn_bias = head_bias

        if self.use_importance_gate and self.importance_net is not None:
            importance = self.importance_net(k_input)
            importance_logit = torch.log(importance + 0.1).view(k_input.shape[0], 1, 1, k_len)
            attn_bias = importance_logit if attn_bias is None else attn_bias + importance_logit

        return attn_bias

    def _frame_attention(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        frame_shape: Tuple[int, int],
    ) -> torch.Tensor:
        if q_tokens.numel() == 0:
            return q_tokens
        if kv_tokens.numel() == 0:
            return torch.zeros_like(q_tokens)

        q_input = self.input_layernorm(q_tokens).unsqueeze(0)
        k_input = self.geo_layernorm(kv_tokens).unsqueeze(0)

        bsz = q_input.shape[0]
        q_len = q_input.shape[1]
        k_len = k_input.shape[1]

        q = self.q_proj(q_input).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_input).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(k_input).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_bias = self._build_attn_bias(q_input, k_input, frame_shape)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        output = output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        output = self.o_proj(output).squeeze(0)
        return torch.tanh(self.gate) * output

    def _local_frame_fusion(
        self,
        semantic_hidden: torch.Tensor,
        geo_hidden: torch.Tensor,
        grid_thw: Optional[torch.Tensor],
    ):
        image_layout = self._build_layout(semantic_hidden.shape[1], grid_thw)
        geo_layout = self._build_layout(geo_hidden.shape[1], grid_thw)
        image_frames = self._split_frames(semantic_hidden, image_layout)
        geo_frames = self._split_frames(geo_hidden, geo_layout)

        if len(image_frames) != len(geo_frames):
            image_layout = FrameLayout([semantic_hidden.shape[1]], [(semantic_hidden.shape[1], 1)])
            geo_layout = FrameLayout([geo_hidden.shape[1]], [(geo_hidden.shape[1], 1)])
            image_frames = self._split_frames(semantic_hidden, image_layout)
            geo_frames = self._split_frames(geo_hidden, geo_layout)

        outputs = []
        for frame_idx, image_tokens in enumerate(image_frames):
            geo_tokens = geo_frames[frame_idx] if frame_idx < len(geo_frames) else geo_frames[-1]
            frame_shape = image_layout.frame_shapes[min(frame_idx, len(image_layout.frame_shapes) - 1)]
            outputs.append(self._frame_attention(image_tokens, geo_tokens, frame_shape))

        if outputs:
            local_delta = torch.cat(outputs, dim=0).unsqueeze(0)
        else:
            local_delta = torch.zeros_like(semantic_hidden)
        return local_delta, image_layout, image_frames, geo_frames

    def _build_frame_query(
        self,
        frame_tokens: torch.Tensor,
        text_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        frame_summary = mean_pool_tokens(frame_tokens)
        if text_hidden_states is None or text_hidden_states.numel() == 0:
            return frame_summary

        if text_hidden_states.dim() == 3:
            text_hidden_states = text_hidden_states.squeeze(0)
        text_summary = mean_pool_tokens(text_hidden_states)
        return self.query_mixer(torch.cat([frame_summary, text_summary], dim=-1))

    def _select_frame_atoms(
        self,
        frame_tokens: torch.Tensor,
        geo_tokens: Optional[torch.Tensor],
        top_r: int,
    ) -> torch.Tensor:
        if frame_tokens.numel() == 0:
            return frame_tokens

        if self.use_importance_gate and self.importance_net is not None:
            scores = self.importance_net(self.input_layernorm(frame_tokens).unsqueeze(0)).squeeze(0).squeeze(-1)
        else:
            scores = frame_tokens.norm(dim=-1)
        _, token_indices = safe_topk(scores, top_r)
        if token_indices.numel() == 0:
            return frame_tokens[:0]

        atoms = frame_tokens[token_indices]
        if geo_tokens is not None and geo_tokens.numel() > 0:
            if geo_tokens.shape[0] == frame_tokens.shape[0]:
                geo_atoms = geo_tokens[token_indices.clamp(max=geo_tokens.shape[0] - 1)]
            else:
                geo_atoms = mean_pool_tokens(geo_tokens).expand_as(atoms)
            atoms = 0.5 * (atoms + self.geo_projector(geo_atoms))
        return atoms

    def _memory_update(self, frame_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        if context_tokens is None or context_tokens.numel() == 0:
            return torch.zeros_like(frame_tokens)

        query = self.memory_norm(frame_tokens).unsqueeze(0)
        context = self.memory_norm(context_tokens).unsqueeze(0)
        update, _ = self.memory_attn(query, context, context)
        return torch.tanh(self.memory_gate) * update.squeeze(0)

    def _log_memory_stats(self, tag: str, available_frames: int, available_atoms: int, frame_topk: int, atom_topk: int):
        if not self.msgf_debug:
            return
        print(
            f"[MSGF] layer={self.layer_idx} stage={tag} "
            f"available_frames={available_frames} available_atoms={available_atoms} "
            f"frame_topk={frame_topk} atom_topk={atom_topk}"
        )


class QwenVGGTInteractionv2(_FramewiseGeometryInteraction):
    def forward(self, semantic_hidden, vggt_features, grid_thw=None, **kwargs):
        local_delta, _, _, _ = self._local_frame_fusion(semantic_hidden, vggt_features, grid_thw)
        return local_delta


class QwenVGGTInteractionv2Flash(QwenVGGTInteractionv2):
    pass


class QwenDA3SGFBaseline(QwenVGGTInteractionv2Flash):
    """DA3 + SGF baseline. Kept as a named alias for script switching."""


class QwenDA3NewInteraction(QwenDA3SGFBaseline):
    """DA3-new: uses blocks_to_take (local+global) features with SGF interaction.

    The feature extraction change (aux -> blocks_to_take, 1536 -> 3072 dim)
    is handled by DA3NewEncoder + GeometryFeatureMerger. After merger projection
    to hidden_size, the cross-attention sees the same dimensionality.
    """


class QwenDA3MSGFBase(_FramewiseGeometryInteraction):
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        self.stage_ranges = compute_stage_ranges(config.num_hidden_layers, "msgf", config)
        self.msgf_topr = int(getattr(config, "msgf_topr", 32))
        self.msgf_frame_topk_max = int(getattr(config, "msgf_frame_topk_max", 3))
        self.msgf_atom_topk_max = int(getattr(config, "msgf_atom_topk_max", 8))

    def forward(self, semantic_hidden, vggt_features, grid_thw=None, text_hidden_states=None, **kwargs):
        local_delta, layout, _, geo_frames = self._local_frame_fusion(semantic_hidden, vggt_features, grid_thw)
        fused_hidden = semantic_hidden + local_delta
        fused_frames = self._split_frames(fused_hidden, layout)

        if self.layer_idx <= self.stage_ranges.warmup_end:
            return local_delta

        frame_atoms = [
            self._select_frame_atoms(frame_tokens, geo_tokens, self.msgf_topr)
            for frame_tokens, geo_tokens in zip(fused_frames, geo_frames)
        ]
        bank = BiDirectionalMemoryBank.from_frame_atoms(frame_atoms)

        memory_updates = []
        frame_topk = 0
        atom_topk = 0
        for frame_idx, frame_tokens in enumerate(fused_frames):
            if self.stage_ranges.write_start <= self.layer_idx <= self.stage_ranges.write_end:
                context = frame_atoms[frame_idx] if frame_idx < len(frame_atoms) else frame_tokens[:0]
                frame_topk = 1 if context.numel() > 0 else 0
                atom_topk = int(context.shape[0]) if context.numel() > 0 else 0
            else:
                query = self._build_frame_query(frame_tokens, text_hidden_states)
                retrieved = bank.retrieve(query, self.msgf_frame_topk_max, self.msgf_atom_topk_max)
                context = retrieved.context
                frame_topk = max(frame_topk, retrieved.frame_topk)
                atom_topk = max(atom_topk, retrieved.atom_topk)
            memory_updates.append(self._memory_update(frame_tokens, context))

        self._log_memory_stats(
            tag="msgf",
            available_frames=len(frame_atoms),
            available_atoms=bank.total_atoms,
            frame_topk=frame_topk,
            atom_topk=atom_topk,
        )
        return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)


class QwenDA3HMSGF(_FramewiseGeometryInteraction):
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        self.stage_ranges = compute_stage_ranges(config.num_hidden_layers, "hmsgf", config)
        self.hmsgf_frame_topk_max = int(getattr(config, "hmsgf_frame_topk_max", 3))
        self.hmsgf_region_topr = int(getattr(config, "hmsgf_region_topr", 32))
        self.hmsgf_region_topk_max = int(getattr(config, "hmsgf_region_topk_max", 8))

    def forward(self, semantic_hidden, vggt_features, grid_thw=None, text_hidden_states=None, **kwargs):
        local_delta, layout, _, geo_frames = self._local_frame_fusion(semantic_hidden, vggt_features, grid_thw)
        fused_hidden = semantic_hidden + local_delta
        fused_frames = self._split_frames(fused_hidden, layout)

        if self.layer_idx <= self.stage_ranges.warmup_end:
            return local_delta

        region_atoms = [
            self._select_frame_atoms(frame_tokens, geo_tokens, self.hmsgf_region_topr)
            for frame_tokens, geo_tokens in zip(fused_frames, geo_frames)
        ]
        bank = HierarchicalMemoryBank.from_frame_atoms(region_atoms)

        memory_updates = []
        frame_topk = 0
        region_topk = 0
        for frame_idx, frame_tokens in enumerate(fused_frames):
            if self.stage_ranges.write_start <= self.layer_idx <= self.stage_ranges.write_end:
                own_atoms = region_atoms[frame_idx] if frame_idx < len(region_atoms) else frame_tokens[:0]
                own_summary = mean_pool_tokens(own_atoms) if own_atoms.numel() > 0 else frame_tokens[:1]
                context = torch.cat([own_summary, own_atoms], dim=0) if own_atoms.numel() > 0 else own_summary
                frame_topk = 1
                region_topk = max(region_topk, int(own_atoms.shape[0]))
            else:
                query = self._build_frame_query(frame_tokens, text_hidden_states)
                retrieved = bank.retrieve(query, self.hmsgf_frame_topk_max, self.hmsgf_region_topk_max)
                context = retrieved.context
                frame_topk = max(frame_topk, retrieved.frame_topk)
                region_topk = max(region_topk, retrieved.atom_topk)
            memory_updates.append(self._memory_update(frame_tokens, context))

        total_regions = sum(int(atoms.shape[0]) for atoms in region_atoms)
        self._log_memory_stats(
            tag="hmsgf",
            available_frames=len(region_atoms),
            available_atoms=total_regions,
            frame_topk=frame_topk,
            atom_topk=region_topk,
        )
        return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)


class QwenDA3RMSGF(_FramewiseGeometryInteraction):
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        self.stage_ranges = compute_stage_ranges(config.num_hidden_layers, "rmsgf", config)
        self.rmsgf_topr = int(getattr(config, "rmsgf_topr", 32))
        self.rmsgf_atom_topk_max = int(getattr(config, "rmsgf_atom_topk_max", 8))
        self.refiner = MemoryRefiner(
            hidden_size=config.hidden_size,
            use_gate=bool(getattr(config, "rmsgf_refine_gate", True)),
            residual=bool(getattr(config, "rmsgf_refine_residual", True)),
        )

    def forward(self, semantic_hidden, vggt_features, grid_thw=None, text_hidden_states=None, **kwargs):
        local_delta, layout, _, geo_frames = self._local_frame_fusion(semantic_hidden, vggt_features, grid_thw)
        fused_hidden = semantic_hidden + local_delta
        fused_frames = self._split_frames(fused_hidden, layout)

        if self.layer_idx <= self.stage_ranges.warmup_end:
            return local_delta

        init_atoms = [
            self._select_frame_atoms(frame_tokens, geo_tokens, self.rmsgf_topr)
            for frame_tokens, geo_tokens in zip(fused_frames, geo_frames)
        ]

        if self.stage_ranges.init_start <= self.layer_idx <= self.stage_ranges.init_end:
            memory_updates = [self._memory_update(frame_tokens, atoms) for frame_tokens, atoms in zip(fused_frames, init_atoms)]
            total_atoms = sum(int(atoms.shape[0]) for atoms in init_atoms)
            self._log_memory_stats("rmsgf_init", len(init_atoms), total_atoms, 1, self.rmsgf_topr)
            return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)

        refined_atoms = []
        for frame_tokens, geo_tokens, atoms in zip(fused_frames, geo_frames, init_atoms):
            if atoms.numel() == 0:
                refined_atoms.append(atoms)
                continue
            frame_summary = self._build_frame_query(frame_tokens, text_hidden_states)
            geo_summary = mean_pool_tokens(geo_tokens) if geo_tokens.numel() > 0 else frame_summary
            refined_atoms.append(self.refiner(atoms, frame_summary + geo_summary))

        bank = BiDirectionalMemoryBank.from_frame_atoms(refined_atoms)
        memory_updates = []
        atom_topk = 0
        for frame_tokens in fused_frames:
            query = self._build_frame_query(frame_tokens, text_hidden_states)
            retrieved = bank.retrieve(query, frame_topk_max=len(refined_atoms), atom_topk_max=self.rmsgf_atom_topk_max)
            atom_topk = max(atom_topk, retrieved.atom_topk)
            memory_updates.append(self._memory_update(frame_tokens, retrieved.context))

        self._log_memory_stats(
            tag="rmsgf_refine",
            available_frames=len(refined_atoms),
            available_atoms=bank.total_atoms,
            frame_topk=len(refined_atoms),
            atom_topk=atom_topk,
        )
        return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)


class QwenDA3MMRInteraction(_FramewiseGeometryInteraction):
    """Candidate D: retain DA3 cross-attention and replace SGF gating with explicit retrieval."""

    def __init__(self, config, layer_idx=None, **kwargs):
        kwargs.pop("use_spatial_bias", None)
        kwargs.pop("use_importance_gate", None)
        super().__init__(
            config,
            layer_idx=layer_idx,
            use_spatial_bias=False,
            use_importance_gate=False,
            **kwargs,
        )
        self.stage_ranges = compute_mmr_stage_ranges(config.num_hidden_layers, config)
        self.mmr_debug = bool(getattr(config, "mmr_debug", False))
        self.mmr_use_region_memory = bool(getattr(config, "mmr_use_region_memory", False))
        self.mmr_frame_topk_max = int(getattr(config, "mmr_frame_topk_max", 3))
        self.mmr_region_topk_max = int(getattr(config, "mmr_region_topk_max", 8))
        self.mmr_use_view_continuity = bool(getattr(config, "mmr_use_view_continuity", True))
        self.mmr_use_temporal_continuity = bool(getattr(config, "mmr_use_temporal_continuity", True))
        self.mmr_region_atoms_per_frame = int(getattr(config, "mmr_region_atoms_per_frame", 8))
        self.mmr_query_use_text = bool(getattr(config, "mmr_query_use_text", True))
        self.mmr_query_use_visual_summary = bool(getattr(config, "mmr_query_use_visual_summary", True))
        self.mmr_memory_dim = int(getattr(config, "mmr_memory_dim", config.hidden_size) or config.hidden_size)
        self.mmr_query_proj = nn.Sequential(
            nn.Linear(config.hidden_size * 2, self.mmr_memory_dim),
            nn.GELU(),
            nn.Linear(self.mmr_memory_dim, config.hidden_size),
        )
        self.retriever = QueryDrivenMMRRetriever(
            frame_topk_max=self.mmr_frame_topk_max,
            region_topk_max=self.mmr_region_topk_max,
            use_view_continuity=self.mmr_use_view_continuity,
            use_temporal_continuity=self.mmr_use_temporal_continuity,
        )

    def _mmr_query(self, frame_tokens: torch.Tensor, text_hidden_states: Optional[torch.Tensor]) -> torch.Tensor:
        query_parts = []
        if self.mmr_query_use_visual_summary:
            query_parts.append(mean_pool_tokens(frame_tokens))
        else:
            query_parts.append(frame_tokens[:1].new_zeros((1, frame_tokens.shape[-1])))

        if self.mmr_query_use_text and text_hidden_states is not None and text_hidden_states.numel() > 0:
            if text_hidden_states.dim() == 3:
                text_hidden_states = text_hidden_states.squeeze(0)
            query_parts.append(mean_pool_tokens(text_hidden_states))
        else:
            query_parts.append(query_parts[0].new_zeros(query_parts[0].shape))

        return self.mmr_query_proj(torch.cat(query_parts, dim=-1))

    def _log_mmr(self, available_frames: int, available_regions: int, frame_topk: int, region_topk: int):
        if not (self.mmr_debug or self.msgf_debug):
            return
        print(
            f"[MMR] layer={self.layer_idx} "
            f"available_frames={available_frames} available_regions={available_regions} "
            f"frame_topk={frame_topk} region_topk={region_topk}"
        )

    def forward(
        self,
        semantic_hidden,
        vggt_features,
        grid_thw=None,
        text_hidden_states=None,
        memory_bank=None,
        frame_ids=None,
        view_ids=None,
        **kwargs,
    ):
        local_delta, layout, _, geo_frames = self._local_frame_fusion(semantic_hidden, vggt_features, grid_thw)
        fused_hidden = semantic_hidden + local_delta
        fused_frames = self._split_frames(fused_hidden, layout)

        if self.layer_idx <= self.stage_ranges.warmup_end:
            return local_delta

        num_frames = len(fused_frames)
        if frame_ids is None:
            frame_ids = list(range(num_frames))
        elif torch.is_tensor(frame_ids):
            frame_ids = [int(v) for v in frame_ids.view(-1).tolist()]
        else:
            frame_ids = [int(v) for v in frame_ids]
        if len(frame_ids) < num_frames:
            frame_ids = frame_ids + list(range(len(frame_ids), num_frames))

        if view_ids is not None and torch.is_tensor(view_ids):
            view_ids = [int(v) for v in view_ids.view(-1).tolist()]

        frame_summaries = [mean_pool_tokens(frame_tokens) for frame_tokens in fused_frames]
        frame_bank = FrameMemoryBank.from_summaries(frame_summaries, frame_ids, view_ids=view_ids)

        region_bank = None
        if self.mmr_use_region_memory:
            region_atoms = [
                self._select_frame_atoms(frame_tokens, geo_tokens, self.mmr_region_atoms_per_frame)
                for frame_tokens, geo_tokens in zip(fused_frames, geo_frames)
            ]
            region_bank = RegionMemoryBank.from_atoms(region_atoms, frame_ids, view_ids=view_ids)

        if self.stage_ranges.write_start <= self.layer_idx <= self.stage_ranges.write_end:
            return local_delta

        memory_updates = []
        max_frame_topk = 0
        max_region_topk = 0
        for frame_idx, frame_tokens in enumerate(fused_frames):
            query = self._mmr_query(frame_tokens, text_hidden_states)
            current_view_id = None if view_ids is None or frame_idx >= len(view_ids) else view_ids[frame_idx]
            retrieved = self.retriever.retrieve(
                query=query,
                frame_bank=frame_bank,
                region_bank=region_bank,
                current_frame_id=frame_ids[frame_idx],
                current_view_id=current_view_id,
                use_region_memory=self.mmr_use_region_memory,
            )
            max_frame_topk = max(max_frame_topk, retrieved.frame_topk)
            max_region_topk = max(max_region_topk, retrieved.region_topk)
            memory_updates.append(self._memory_update(frame_tokens, retrieved.context))

        self._log_mmr(
            available_frames=frame_bank.size,
            available_regions=0 if region_bank is None else region_bank.size,
            frame_topk=max_frame_topk,
            region_topk=max_region_topk,
        )
        return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)


class QwenZenViewVGGTGeometryBank(nn.Module):
    """ZenView geometry bank fusion with continuity-aware routing."""

    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = 0 if layer_idx is None else int(layer_idx)
        self.hidden_size = config.hidden_size
        fusion_layer_indices = str(getattr(config, "vggt_bank_fusion_layer_indices", "")).strip()
        if fusion_layer_indices:
            self.geometry_aware_layer_indices = {
                int(index.strip()) for index in fusion_layer_indices.split(",") if index.strip()
            }
            router_num_layers = max(self.geometry_aware_layer_indices) + 1 if self.geometry_aware_layer_indices else 1
            self.num_geometry_aware_layers = router_num_layers
        else:
            self.geometry_aware_layer_indices = None
            self.num_geometry_aware_layers = int(getattr(config, "vggt_bank_num_layers", 20))
            router_num_layers = self.num_geometry_aware_layers
        self.bank_debug = bool(getattr(config, "bank_debug", False))
        self.variant_name = str(getattr(config, "geo_inject_version", "zenview_vggt_bank"))

        if hasattr(config, "vision_config"):
            self.pooling_stride = config.vision_config.spatial_merge_size
        else:
            self.pooling_stride = 2
        if kwargs.pop("depart_smi_token", False):
            self.pooling_stride *= kwargs.pop("smi_downsample_rate", 2)

        self.router = BankRouter(
            hidden_size=config.hidden_size,
            d_geom=int(getattr(config, "vggt_bank_d_geom", 1024)),
            num_layers=router_num_layers,
            topk=int(getattr(config, "vggt_bank_topk", 2)),
            use_layer_embedding=bool(getattr(config, "vggt_bank_use_layer_embedding", True)),
            normalize_query=bool(getattr(config, "normalize_query", False)),
            normalize_bank=bool(getattr(config, "normalize_bank", False)),
            temperature=float(getattr(config, "bank_temperature", 1.0)),
            candidate_dropout_enabled=bool(getattr(config, "candidate_dropout_enabled", False)),
            g11_drop_prob=float(getattr(config, "g11_drop_prob", 0.0)),
            g17_drop_prob=float(getattr(config, "g17_drop_prob", 0.0)),
            g23_drop_prob=float(getattr(config, "g23_drop_prob", 0.0)),
        )
        self.fusion = BankFusionBlock(
            hidden_size=config.hidden_size,
            d_geom=int(getattr(config, "vggt_bank_d_geom", 1024)),
            gate_mode=str(getattr(config, "bank_gate_mode", "scalar")),
        )

    def _zero_delta(self, semantic_hidden: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(semantic_hidden)

    def _log_router_stats(
        self,
        selection_histogram: torch.Tensor,
        drop_histogram: torch.Tensor,
        bank_stats: dict,
        gate_values: list[torch.Tensor],
        raw_candidate_norm: Optional[torch.Tensor] = None,
    ):
        if not self.bank_debug:
            return
        total = float(selection_histogram.sum().item())
        if total <= 0:
            return

        ratios = selection_histogram / total
        if gate_values:
            gate_tensor = torch.cat([gate.reshape(-1) for gate in gate_values], dim=0)
            gate_mean = float(gate_tensor.mean().detach().item())
            gate_std = float(gate_tensor.std(unbiased=False).detach().item()) if gate_tensor.numel() > 1 else 0.0
        else:
            gate_mean = 0.0
            gate_std = 0.0
        if raw_candidate_norm is None:
            raw_candidate_norm = torch.zeros(4, device=selection_histogram.device, dtype=selection_histogram.dtype)

        drop_total = float(drop_histogram.sum().item())
        drop_ratios = drop_histogram / max(float(drop_total), 1.0)
        projected_candidate_norm = [
            float(bank_stats.get("g11_norm", 0.0)),
            float(bank_stats.get("g17_norm", 0.0)),
            float(bank_stats.get("g23_norm", 0.0)),
            float(bank_stats.get("cont_norm", 0.0)),
        ]
        raw_candidate_norm_list = [float(raw_candidate_norm[index]) for index in range(min(4, raw_candidate_norm.shape[0]))]
        if len(raw_candidate_norm_list) < 4:
            raw_candidate_norm_list.extend([0.0] * (4 - len(raw_candidate_norm_list)))

        print(
            f"[{self.variant_name}] layer={self.layer_idx} "
            f"g11={float(ratios[0]):.4f} g17={float(ratios[1]):.4f} "
            f"g23={float(ratios[2]):.4f} cont={float(ratios[3]):.4f} "
            f"drop_g11={float(drop_ratios[0]):.4f} drop_g17={float(drop_ratios[1]):.4f} "
            f"drop_g23={float(drop_ratios[2]):.4f} "
            f"gate_mean={gate_mean:.4f} gate_std={gate_std:.4f} "
            f"g11_norm={bank_stats.get('g11_norm', 0.0):.4f} "
            f"g17_norm={bank_stats.get('g17_norm', 0.0):.4f} "
            f"g23_norm={bank_stats.get('g23_norm', 0.0):.4f} "
            f"cont_norm={bank_stats.get('cont_norm', 0.0):.4f} "
            f"raw_g11_norm={raw_candidate_norm_list[0]:.4f} "
            f"raw_g17_norm={raw_candidate_norm_list[1]:.4f} "
            f"raw_g23_norm={raw_candidate_norm_list[2]:.4f} "
            f"raw_cont_norm={raw_candidate_norm_list[3]:.4f}"
        )

        collector = get_router_stats_collector()
        if collector is not None:
            collector.record(
                variant_name=self.variant_name,
                layer_idx=self.layer_idx,
                selection_histogram=[float(value) for value in selection_histogram.detach().cpu().tolist()],
                drop_histogram=[float(value) for value in drop_histogram.detach().cpu().tolist()],
                raw_candidate_norm=raw_candidate_norm_list,
                projected_candidate_norm=projected_candidate_norm,
                gate_mean=gate_mean,
                gate_std=gate_std,
            )

    def forward(self, semantic_hidden, geometry_bank, grid_thw=None, text_hidden_states=None, **kwargs):
        if geometry_bank is None:
            return self._zero_delta(semantic_hidden)
        if self.geometry_aware_layer_indices is not None:
            if self.layer_idx not in self.geometry_aware_layer_indices:
                return self._zero_delta(semantic_hidden)
        elif self.layer_idx >= self.num_geometry_aware_layers:
            return self._zero_delta(semantic_hidden)
        if not hasattr(geometry_bank, "frame_layout") or not hasattr(geometry_bank, "get_frame_bank"):
            raise TypeError("zenview_vggt_bank expects a GeometryBankOutput-like object.")

        semantic_frames = split_by_layout(semantic_hidden, geometry_bank.frame_layout)
        if not semantic_frames:
            fallback_layout = infer_frame_layout(semantic_hidden.shape[1], grid_thw, self.pooling_stride)
            semantic_frames = split_by_layout(semantic_hidden, fallback_layout)
        if not semantic_frames:
            return self._zero_delta(semantic_hidden)

        selection_histogram = torch.zeros(4, device=semantic_hidden.device, dtype=semantic_hidden.dtype)
        drop_histogram = torch.zeros(4, device=semantic_hidden.device, dtype=semantic_hidden.dtype)
        raw_candidate_norm = torch.zeros(4, device=semantic_hidden.device, dtype=semantic_hidden.dtype)
        raw_candidate_norm_count = 0
        gate_values = []
        delta_frames = []
        available_frames = min(len(semantic_frames), len(geometry_bank.frame_layout.token_counts))

        for frame_idx in range(available_frames):
            frame_tokens = semantic_frames[frame_idx]
            frame_bank = geometry_bank.get_frame_bank(frame_idx).to(device=frame_tokens.device)
            overlap = min(frame_tokens.shape[0], frame_bank.shape[0])
            if overlap == 0:
                delta_frames.append(torch.zeros_like(frame_tokens))
                continue

            selected_memory, router_info = self.router(frame_tokens[:overlap], frame_bank[:overlap], self.layer_idx)
            frame_delta, gate = self.fusion(frame_tokens[:overlap], selected_memory)

            if overlap < frame_tokens.shape[0]:
                padded_delta = torch.zeros_like(frame_tokens)
                padded_delta[:overlap] = frame_delta
                frame_delta = padded_delta

            selection_histogram = selection_histogram + router_info["selection_histogram"].to(selection_histogram.dtype)
            drop_histogram = drop_histogram + router_info["drop_histogram"].to(drop_histogram.dtype)
            raw_candidate_norm = raw_candidate_norm + router_info["raw_candidate_norm"].to(raw_candidate_norm.dtype)
            raw_candidate_norm_count += 1
            gate_values.append(gate)
            delta_frames.append(frame_delta)

        for frame_idx in range(available_frames, len(semantic_frames)):
            delta_frames.append(torch.zeros_like(semantic_frames[frame_idx]))

        if not delta_frames:
            return self._zero_delta(semantic_hidden)

        if raw_candidate_norm_count > 0:
            raw_candidate_norm = raw_candidate_norm / raw_candidate_norm_count
        self._log_router_stats(selection_histogram, drop_histogram, geometry_bank.stats, gate_values, raw_candidate_norm)
        return torch.cat(delta_frames, dim=0).unsqueeze(0)


class QwenZenViewContinuityBankV2(QwenZenViewVGGTGeometryBank):
    """Guide-compliant continuity bank variant without layer embeddings."""

    def __init__(self, config, layer_idx=None, **kwargs):
        setattr(config, "vggt_bank_use_layer_embedding", False)
        super().__init__(config, layer_idx=layer_idx, **kwargs)


class QwenGeoBridgeHGB(QwenZenViewVGGTGeometryBank):
    """SpatialFit Stage 2 heterogeneous competitive geometry bridge."""

    def __init__(self, config, layer_idx=None, **kwargs):
        setattr(config, "vggt_bank_use_layer_embedding", False)
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        # HGB uses its own local/continuity path and does not consume parent router/fusion.
        self.router = None
        self.fusion = None
        self.uses_parent_router_fusion = False
        d_geom = int(getattr(config, "vggt_bank_d_geom", 1024))
        self.hgb_strict_alignment = bool(getattr(config, "hgb_strict_alignment", True))
        self.hgb_allow_layout_fallback = bool(getattr(config, "hgb_allow_layout_fallback", False))
        self.hgb_alignment_audit_only = bool(getattr(config, "hgb_alignment_audit_only", False))
        self.hgb_min_overlap_ratio = float(getattr(config, "hgb_min_overlap_ratio", 1.0))
        self.hgb_layer0_g11_logit_bias = float(getattr(config, "hgb_layer0_g11_logit_bias", 2.0))
        self.last_hgb_stats: Dict[str, object] = {"layer_idx": self.layer_idx, "active": False}
        self.local_router = LocalLevelRouter(
            hidden_size=config.hidden_size,
            d_geom=d_geom,
            topk=int(getattr(config, "hgb_local_topk", 2)),
            normalize_query=bool(getattr(config, "normalize_query", True)),
            normalize_bank=bool(getattr(config, "normalize_bank", True)),
            temperature=float(getattr(config, "bank_temperature", 0.07)),
        )
        self.local_fusion = BankFusionBlock(
            hidden_size=config.hidden_size,
            d_geom=d_geom,
            gate_mode=str(getattr(config, "bank_gate_mode", "scalar")),
        )
        self.cont_fusion = BankFusionBlock(
            hidden_size=config.hidden_size,
            d_geom=d_geom,
            gate_mode=str(getattr(config, "bank_gate_mode", "scalar")),
        )
        self.bridge_gate = CompetitiveBridgeGate(
            hidden_size=config.hidden_size,
            d_geom=d_geom,
            use_saliency_prior=bool(getattr(config, "hgb_use_saliency_prior", True)),
            layer_scale_init=float(getattr(config, "hgb_layer_scale_init", 0.05)),
            gate_none_bias=float(getattr(config, "hgb_gate_none_bias", 0.0)),
            gate_local_bias=float(getattr(config, "hgb_gate_local_bias", 0.4)),
            gate_cont_bias=float(getattr(config, "hgb_gate_cont_bias", 0.6)),
            use_gate_bias_init=bool(getattr(config, "hgb_use_gate_bias_init", True)),
        )

    def _inactive_delta(self, semantic_hidden: torch.Tensor, reason: str) -> torch.Tensor:
        self.last_hgb_stats = {
            "layer_idx": self.layer_idx,
            "active": False,
            "reason": reason,
        }
        return self._zero_delta(semantic_hidden)

    def _layout_summary(self, layout) -> str:
        if layout is None:
            return "None"
        token_counts = list(getattr(layout, "token_counts", []))
        frame_shapes = list(getattr(layout, "frame_shapes", []))
        return f"token_counts={token_counts[:8]} frame_shapes={frame_shapes[:8]}"

    def _warn_alignment(self, message: str) -> None:
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _finalize_hgb_stats(self, stats: Dict[str, object], geometry_bank_stats: Optional[dict] = None) -> Dict[str, object]:
        gate_total = float(sum(stats["gate_histogram"]))
        gate_prob_count = max(int(stats["gate_prob_count"]), 1)
        local_total = float(sum(stats["local_selection_histogram"]))
        local_norm_count = max(int(stats["raw_local_norm_count"]), 1)
        cont_norm_count = max(int(stats["raw_cont_norm_count"]), 1)
        overlap_ratio_count = int(stats["overlap_ratio_count"])
        metric_count = max(int(stats["metric_count"]), 1)
        saliency_count = max(int(stats["saliency_count"]), 1)
        result = {
            "layer_idx": self.layer_idx,
            "active": True,
            "available_frames": int(stats["available_frames"]),
            "layout_fallback_count": int(stats["layout_fallback_count"]),
            "token_mismatch_count": int(stats["token_mismatch_count"]),
            "per_layer_mismatch_count": {str(self.layer_idx): int(stats["token_mismatch_count"])},
            "min_overlap_ratio": float(stats["min_overlap_ratio"] if overlap_ratio_count > 0 else 1.0),
            "mean_overlap_ratio": float(
                stats["overlap_ratio_sum"] / overlap_ratio_count if overlap_ratio_count > 0 else 1.0
            ),
            "overlap_ratio_count": overlap_ratio_count,
            "overlap_ratio_sum": float(stats["overlap_ratio_sum"]),
            "gate_histogram": [float(value) for value in stats["gate_histogram"]],
            "gate_ratio": [
                float(value / gate_total) if gate_total > 0 else 0.0 for value in stats["gate_histogram"]
            ],
            "gate_prob_mean": [float(value / gate_prob_count) for value in stats["gate_prob_sum"]],
            "local_selection_histogram": [float(value) for value in stats["local_selection_histogram"]],
            "local_selection_ratio": [
                float(value / local_total) if local_total > 0 else 0.0
                for value in stats["local_selection_histogram"]
            ],
            "raw_local_norm_mean": [float(value / local_norm_count) for value in stats["raw_local_norm_sum"]],
            "raw_cont_norm_mean": float(stats["raw_cont_norm_sum"] / cont_norm_count),
            "hidden_norm_mean": float(stats["hidden_norm_sum"] / metric_count),
            "local_delta_norm_mean": float(stats["local_delta_norm_sum"] / metric_count),
            "cont_delta_norm_mean": float(stats["cont_delta_norm_sum"] / metric_count),
            "mixed_norm_mean": float(stats["mixed_norm_sum"] / metric_count),
            "layer_scale_tanh": float(stats["layer_scale_tanh"]),
            "saliency_mean": float(stats["saliency_sum"] / saliency_count),
            "saliency_std": float(stats["saliency_std_sum"] / saliency_count),
            "local_entropy_mean": float(stats["local_entropy_sum"] / metric_count),
            "used_layer0_g11_bias": bool(stats["used_layer0_g11_bias"]),
        }
        if geometry_bank_stats is not None:
            result["projected_candidate_norm_mean"] = [
                float(geometry_bank_stats.get("g11_norm", 0.0)),
                float(geometry_bank_stats.get("g17_norm", 0.0)),
                float(geometry_bank_stats.get("g23_norm", 0.0)),
                float(geometry_bank_stats.get("cont_norm", 0.0)),
            ]
        return result

    def _log_hgb_stats(
        self,
        finalized_stats: Dict[str, object],
        raw_candidate_norm: List[float],
        projected_candidate_norm: List[float],
        local_gate_values: List[torch.Tensor],
        cont_gate_values: List[torch.Tensor],
    ) -> None:
        if self.bank_debug:
            print(
                f"[{self.variant_name}] layer={self.layer_idx} "
                f"local_g11={finalized_stats['local_selection_ratio'][0]:.4f} "
                f"local_g17={finalized_stats['local_selection_ratio'][1]:.4f} "
                f"local_g23={finalized_stats['local_selection_ratio'][2]:.4f} "
                f"none={finalized_stats['gate_ratio'][0]:.4f} "
                f"local={finalized_stats['gate_ratio'][1]:.4f} "
                f"cont={finalized_stats['gate_ratio'][2]:.4f} "
                f"prob_none={finalized_stats['gate_prob_mean'][0]:.4f} "
                f"prob_local={finalized_stats['gate_prob_mean'][1]:.4f} "
                f"prob_cont={finalized_stats['gate_prob_mean'][2]:.4f} "
                f"layer_scale_tanh={finalized_stats['layer_scale_tanh']:.4f} "
                f"local_delta_norm={finalized_stats['local_delta_norm_mean']:.4f} "
                f"cont_delta_norm={finalized_stats['cont_delta_norm_mean']:.4f} "
                f"mixed_norm={finalized_stats['mixed_norm_mean']:.4f} "
                f"hidden_norm={finalized_stats['hidden_norm_mean']:.4f} "
                f"saliency_mean={finalized_stats['saliency_mean']:.4f} "
                f"saliency_std={finalized_stats['saliency_std']:.4f} "
                f"layout_fallbacks={finalized_stats['layout_fallback_count']} "
                f"token_mismatches={finalized_stats['token_mismatch_count']} "
                f"min_overlap={finalized_stats['min_overlap_ratio']:.4f} "
                f"mean_overlap={finalized_stats['mean_overlap_ratio']:.4f}"
            )

        gate_tensors = [gate.reshape(-1) for gate in local_gate_values + cont_gate_values if gate is not None]
        if gate_tensors:
            gate_tensor = torch.cat(gate_tensors, dim=0)
            gate_mean = float(gate_tensor.mean().detach().item())
            gate_std = float(gate_tensor.std(unbiased=False).detach().item()) if gate_tensor.numel() > 1 else 0.0
        else:
            gate_mean = 0.0
            gate_std = 0.0

        collector = get_router_stats_collector()
        if collector is not None:
            collector.record(
                variant_name=self.variant_name,
                layer_idx=self.layer_idx,
                selection_histogram=[
                    float(finalized_stats["local_selection_histogram"][0]),
                    float(finalized_stats["local_selection_histogram"][1]),
                    float(finalized_stats["local_selection_histogram"][2]),
                    0.0,
                ],
                drop_histogram=[0.0, 0.0, 0.0, 0.0],
                raw_candidate_norm=raw_candidate_norm,
                projected_candidate_norm=projected_candidate_norm,
                gate_mean=gate_mean,
                gate_std=gate_std,
                extra_metrics={
                    "gate_histogram": finalized_stats["gate_histogram"],
                    "gate_prob_mean": finalized_stats["gate_prob_mean"],
                    "local_selection_histogram": finalized_stats["local_selection_histogram"],
                    "layer_scale_tanh": finalized_stats["layer_scale_tanh"],
                    "hidden_norm_mean": finalized_stats["hidden_norm_mean"],
                    "local_delta_norm_mean": finalized_stats["local_delta_norm_mean"],
                    "cont_delta_norm_mean": finalized_stats["cont_delta_norm_mean"],
                    "mixed_norm_mean": finalized_stats["mixed_norm_mean"],
                    "saliency_mean": finalized_stats["saliency_mean"],
                    "saliency_std": finalized_stats["saliency_std"],
                    "local_entropy_mean": finalized_stats["local_entropy_mean"],
                    "layout_fallback_count": finalized_stats["layout_fallback_count"],
                    "token_mismatch_count": finalized_stats["token_mismatch_count"],
                    "min_overlap_ratio": finalized_stats["min_overlap_ratio"],
                    "overlap_ratio_sum": finalized_stats["overlap_ratio_sum"],
                    "overlap_ratio_count": finalized_stats["overlap_ratio_count"],
                    "available_frames": finalized_stats["available_frames"],
                },
            )

    def forward(self, semantic_hidden, geometry_bank, grid_thw=None, text_hidden_states=None, **kwargs):
        if geometry_bank is None:
            return self._inactive_delta(semantic_hidden, reason="missing_geometry_bank")
        if self.geometry_aware_layer_indices is not None:
            if self.layer_idx not in self.geometry_aware_layer_indices:
                return self._inactive_delta(semantic_hidden, reason="layer_not_in_sparse_schedule")
        elif self.layer_idx >= self.num_geometry_aware_layers:
            return self._inactive_delta(semantic_hidden, reason="layer_outside_num_geometry_aware_layers")
        if not hasattr(geometry_bank, "frame_layout") or not hasattr(geometry_bank, "get_frame_bank"):
            raise TypeError("geobridge_hgb expects a GeometryBankOutput-like object.")

        semantic_frames = split_by_layout(semantic_hidden, geometry_bank.frame_layout)
        layout_fallback_count = 0
        if not semantic_frames:
            layout_fallback_count += 1
            fallback_layout = infer_frame_layout(semantic_hidden.shape[1], grid_thw, self.pooling_stride)
            if self.hgb_allow_layout_fallback:
                semantic_frames = split_by_layout(semantic_hidden, fallback_layout)
                self._warn_alignment(
                    f"[{self.variant_name}] layer={self.layer_idx} used layout fallback; "
                    f"semantic_hidden_shape={tuple(semantic_hidden.shape)} grid_thw={grid_thw} "
                    f"bank_layout={self._layout_summary(getattr(geometry_bank, 'frame_layout', None))}"
                )
            else:
                semantic_frames = []
        if not semantic_frames:
            message = (
                f"[{self.variant_name}] strict alignment failed at layer={self.layer_idx}: "
                f"semantic_hidden_shape={tuple(semantic_hidden.shape)} grid_thw={grid_thw} "
                f"bank_layout={self._layout_summary(getattr(geometry_bank, 'frame_layout', None))}"
            )
            if self.hgb_alignment_audit_only:
                self._warn_alignment(message)
                self.last_hgb_stats = {
                    "layer_idx": self.layer_idx,
                    "active": False,
                    "reason": "empty_semantic_frames",
                    "layout_fallback_count": layout_fallback_count,
                }
                return self._zero_delta(semantic_hidden)
            raise RuntimeError(message)

        stats = {
            "local_selection_histogram": [0.0, 0.0, 0.0],
            "gate_histogram": [0.0, 0.0, 0.0],
            "gate_prob_sum": [0.0, 0.0, 0.0],
            "raw_local_norm_sum": [0.0, 0.0, 0.0],
            "raw_local_norm_count": 0,
            "raw_cont_norm_sum": 0.0,
            "raw_cont_norm_count": 0,
            "gate_prob_count": 0,
            "hidden_norm_sum": 0.0,
            "local_delta_norm_sum": 0.0,
            "cont_delta_norm_sum": 0.0,
            "mixed_norm_sum": 0.0,
            "local_entropy_sum": 0.0,
            "layout_fallback_count": layout_fallback_count,
            "token_mismatch_count": 0,
            "min_overlap_ratio": 1.0,
            "overlap_ratio_sum": 0.0,
            "overlap_ratio_count": 0,
            "available_frames": 0,
            "metric_count": 0,
            "saliency_sum": 0.0,
            "saliency_std_sum": 0.0,
            "saliency_count": 0,
            "layer_scale_tanh": float(torch.tanh(self.bridge_gate.layer_scale.detach()).item()),
            "used_layer0_g11_bias": False,
        }
        local_gate_values = []
        cont_gate_values = []
        delta_frames = []
        available_frames = min(len(semantic_frames), len(geometry_bank.frame_layout.token_counts))
        stats["available_frames"] = available_frames

        fast_items = []
        can_use_flat_fast_path = available_frames > 0
        if can_use_flat_fast_path:
            for frame_idx in range(available_frames):
                frame_tokens = semantic_frames[frame_idx]
                frame_bank = geometry_bank.get_frame_bank(frame_idx).to(device=frame_tokens.device)
                if frame_bank.dim() != 3 or frame_bank.shape[1] < 4:
                    raise RuntimeError(
                        f"[{self.variant_name}] layer={self.layer_idx} expected frame_bank shape [tokens,4,d], "
                        f"got {tuple(frame_bank.shape)}"
                    )
                if frame_tokens.shape[0] == 0 or frame_tokens.shape[0] != frame_bank.shape[0]:
                    can_use_flat_fast_path = False
                    break
                fast_items.append((frame_idx, frame_tokens, frame_bank))

        if can_use_flat_fast_path and len(fast_items) == available_frames:
            frame_lengths = [int(frame_tokens.shape[0]) for _, frame_tokens, _ in fast_items]
            flat_tokens = torch.cat([frame_tokens for _, frame_tokens, _ in fast_items], dim=0)
            flat_local_bank = torch.cat([frame_bank[:, :3, :] for _, _, frame_bank in fast_items], dim=0)
            flat_cont_bank = torch.cat([frame_bank[:, 3, :] for _, _, frame_bank in fast_items], dim=0)

            flat_saliency_parts = []
            for frame_idx, frame_tokens, _ in fast_items:
                stats["overlap_ratio_sum"] += 1.0
                stats["overlap_ratio_count"] += 1
                saliency = geometry_bank.get_frame_saliency(frame_idx)
                if saliency is not None:
                    saliency = saliency[: frame_tokens.shape[0]].to(device=frame_tokens.device)
                    flat_saliency_parts.append(saliency)
                    stats["saliency_sum"] += float(saliency.float().mean().item())
                    saliency_std = saliency.float().std(unbiased=False).item() if saliency.numel() > 1 else 0.0
                    stats["saliency_std_sum"] += float(saliency_std)
                    stats["saliency_count"] += 1
            flat_saliency = torch.cat(flat_saliency_parts, dim=0) if flat_saliency_parts else None

            candidate_logit_bias = None
            if self.layer_idx == 0 and self.hgb_layer0_g11_logit_bias != 0.0:
                candidate_logit_bias = flat_local_bank.new_zeros((flat_local_bank.shape[1],))
                candidate_logit_bias[0] = float(self.hgb_layer0_g11_logit_bias)
                stats["used_layer0_g11_bias"] = True

            local_memory, local_info = self.local_router(
                flat_tokens,
                flat_local_bank,
                candidate_logit_bias=candidate_logit_bias,
            )
            local_delta, local_gate = self.local_fusion(flat_tokens, local_memory)
            cont_delta, cont_gate = self.cont_fusion(flat_tokens, flat_cont_bank)
            frame_delta, gate_info = self.bridge_gate(
                flat_tokens,
                local_memory,
                flat_cont_bank,
                local_delta,
                cont_delta,
                saliency_prior=flat_saliency,
            )

            for candidate_idx in range(3):
                stats["local_selection_histogram"][candidate_idx] += float(
                    local_info["selection_histogram"][candidate_idx].detach().item()
                )
                stats["raw_local_norm_sum"][candidate_idx] += float(
                    local_info["raw_candidate_norm"][candidate_idx].detach().item()
                )
            stats["raw_local_norm_count"] += 1
            stats["raw_cont_norm_sum"] += float(flat_cont_bank.float().norm(dim=-1).mean().detach().item())
            stats["raw_cont_norm_count"] += 1
            for gate_idx in range(3):
                stats["gate_histogram"][gate_idx] += float(gate_info["gate_histogram"][gate_idx].detach().item())
                stats["gate_prob_sum"][gate_idx] += float(gate_info["gate_probs"][:, gate_idx].sum().detach().item())
            stats["gate_prob_count"] += int(gate_info["gate_probs"].shape[0])
            stats["hidden_norm_sum"] += float(flat_tokens.float().norm(dim=-1).mean().detach().item())
            stats["local_delta_norm_sum"] += float(gate_info["local_delta_norm_mean"].item())
            stats["cont_delta_norm_sum"] += float(gate_info["cont_delta_norm_mean"].item())
            stats["mixed_norm_sum"] += float(gate_info["mixed_norm_mean"].item())
            stats["local_entropy_sum"] += (
                float(local_info["entropy"].mean().detach().item()) if local_info["entropy"].numel() > 0 else 0.0
            )
            stats["metric_count"] += 1
            stats["layer_scale_tanh"] = float(gate_info["layer_scale_tanh"].item())
            local_gate_values.append(local_gate)
            cont_gate_values.append(cont_gate)
            delta_frames = list(torch.split(frame_delta, frame_lengths, dim=0))
            for frame_idx in range(available_frames, len(semantic_frames)):
                delta_frames.append(torch.zeros_like(semantic_frames[frame_idx]))

            finalized_stats = self._finalize_hgb_stats(stats, geometry_bank_stats=geometry_bank.stats)
            raw_candidate_norm = [
                finalized_stats["raw_local_norm_mean"][0],
                finalized_stats["raw_local_norm_mean"][1],
                finalized_stats["raw_local_norm_mean"][2],
                finalized_stats["raw_cont_norm_mean"],
            ]
            projected_candidate_norm = list(finalized_stats.get("projected_candidate_norm_mean", [0.0, 0.0, 0.0, 0.0]))
            self.last_hgb_stats = finalized_stats
            self._log_hgb_stats(
                finalized_stats=finalized_stats,
                raw_candidate_norm=raw_candidate_norm,
                projected_candidate_norm=projected_candidate_norm,
                local_gate_values=local_gate_values,
                cont_gate_values=cont_gate_values,
            )
            return torch.cat(delta_frames, dim=0).unsqueeze(0)

        for frame_idx in range(available_frames):
            frame_tokens = semantic_frames[frame_idx]
            frame_bank = geometry_bank.get_frame_bank(frame_idx).to(device=frame_tokens.device)
            if frame_bank.dim() != 3 or frame_bank.shape[1] < 4:
                raise RuntimeError(
                    f"[{self.variant_name}] layer={self.layer_idx} expected frame_bank shape [tokens,4,d], "
                    f"got {tuple(frame_bank.shape)}"
                )
            overlap = min(frame_tokens.shape[0], frame_bank.shape[0])
            max_len = max(frame_tokens.shape[0], frame_bank.shape[0], 1)
            overlap_ratio = float(overlap) / float(max_len)
            stats["overlap_ratio_sum"] += overlap_ratio
            stats["overlap_ratio_count"] += 1
            stats["min_overlap_ratio"] = min(float(stats["min_overlap_ratio"]), overlap_ratio)
            if frame_tokens.shape[0] != frame_bank.shape[0] or overlap_ratio < self.hgb_min_overlap_ratio:
                stats["token_mismatch_count"] += 1
                mismatch_message = (
                    f"[{self.variant_name}] token alignment mismatch at layer={self.layer_idx} frame={frame_idx}: "
                    f"frame_tokens={frame_tokens.shape[0]} frame_bank={frame_bank.shape[0]} "
                    f"overlap_ratio={overlap_ratio:.4f} threshold={self.hgb_min_overlap_ratio:.4f}"
                )
                if self.hgb_strict_alignment and not self.hgb_alignment_audit_only:
                    raise RuntimeError(mismatch_message)
                self._warn_alignment(mismatch_message)
            if overlap == 0:
                delta_frames.append(torch.zeros_like(frame_tokens))
                continue

            local_bank = frame_bank[:overlap, :3, :]
            cont_bank = frame_bank[:overlap, 3, :]
            saliency = geometry_bank.get_frame_saliency(frame_idx)
            if saliency is not None:
                saliency = saliency[:overlap].to(device=frame_tokens.device)
                stats["saliency_sum"] += float(saliency.float().mean().item())
                saliency_std = saliency.float().std(unbiased=False).item() if saliency.numel() > 1 else 0.0
                stats["saliency_std_sum"] += float(saliency_std)
                stats["saliency_count"] += 1
            candidate_logit_bias = None
            if self.layer_idx == 0 and self.hgb_layer0_g11_logit_bias != 0.0:
                candidate_logit_bias = local_bank.new_zeros((local_bank.shape[1],))
                candidate_logit_bias[0] = float(self.hgb_layer0_g11_logit_bias)
                stats["used_layer0_g11_bias"] = True

            local_memory, local_info = self.local_router(
                frame_tokens[:overlap],
                local_bank,
                candidate_logit_bias=candidate_logit_bias,
            )
            local_delta, local_gate = self.local_fusion(frame_tokens[:overlap], local_memory)
            cont_delta, cont_gate = self.cont_fusion(frame_tokens[:overlap], cont_bank)
            frame_delta, gate_info = self.bridge_gate(
                frame_tokens[:overlap],
                local_memory,
                cont_bank,
                local_delta,
                cont_delta,
                saliency_prior=saliency,
            )

            if overlap < frame_tokens.shape[0]:
                padded_delta = torch.zeros_like(frame_tokens)
                padded_delta[:overlap] = frame_delta
                frame_delta = padded_delta

            for candidate_idx in range(3):
                stats["local_selection_histogram"][candidate_idx] += float(
                    local_info["selection_histogram"][candidate_idx].detach().item()
                )
                stats["raw_local_norm_sum"][candidate_idx] += float(local_info["raw_candidate_norm"][candidate_idx].detach().item())
            stats["raw_local_norm_count"] += 1
            stats["raw_cont_norm_sum"] += float(cont_bank.float().norm(dim=-1).mean().detach().item())
            stats["raw_cont_norm_count"] += 1
            for gate_idx in range(3):
                stats["gate_histogram"][gate_idx] += float(gate_info["gate_histogram"][gate_idx].detach().item())
                stats["gate_prob_sum"][gate_idx] += float(gate_info["gate_probs"][:, gate_idx].sum().detach().item())
            stats["gate_prob_count"] += int(gate_info["gate_probs"].shape[0])
            stats["hidden_norm_sum"] += float(frame_tokens[:overlap].float().norm(dim=-1).mean().detach().item())
            stats["local_delta_norm_sum"] += float(gate_info["local_delta_norm_mean"].item())
            stats["cont_delta_norm_sum"] += float(gate_info["cont_delta_norm_mean"].item())
            stats["mixed_norm_sum"] += float(gate_info["mixed_norm_mean"].item())
            stats["local_entropy_sum"] += float(local_info["entropy"].mean().detach().item()) if local_info["entropy"].numel() > 0 else 0.0
            stats["metric_count"] += 1
            stats["layer_scale_tanh"] = float(gate_info["layer_scale_tanh"].item())
            local_gate_values.append(local_gate)
            cont_gate_values.append(cont_gate)
            delta_frames.append(frame_delta)

        for frame_idx in range(available_frames, len(semantic_frames)):
            delta_frames.append(torch.zeros_like(semantic_frames[frame_idx]))

        finalized_stats = self._finalize_hgb_stats(stats, geometry_bank_stats=geometry_bank.stats)
        raw_candidate_norm = [
            finalized_stats["raw_local_norm_mean"][0],
            finalized_stats["raw_local_norm_mean"][1],
            finalized_stats["raw_local_norm_mean"][2],
            finalized_stats["raw_cont_norm_mean"],
        ]
        projected_candidate_norm = list(finalized_stats.get("projected_candidate_norm_mean", [0.0, 0.0, 0.0, 0.0]))
        self.last_hgb_stats = finalized_stats
        if delta_frames:
            self._log_hgb_stats(
                finalized_stats=finalized_stats,
                raw_candidate_norm=raw_candidate_norm,
                projected_candidate_norm=projected_candidate_norm,
                local_gate_values=local_gate_values,
                cont_gate_values=cont_gate_values,
            )

        return torch.cat(delta_frames, dim=0).unsqueeze(0)


class QwenDA3MSGFTemporalRefine(_FramewiseGeometryInteraction):
    """MSGF-Base + temporal continuity bonus + lightweight atom refinement.

    Read-stage changes vs MSGF-Base:
    1. Frame-level retrieval adds temporal_bonus: lambda_t / (1 + |i-j|)
    2. Candidate atoms are refined by a gated residual MLP before atom-level ranking

    Paper name: Zen Context Memory
    """

    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        self.stage_ranges = compute_stage_ranges(config.num_hidden_layers, "msgf", config)
        self.msgf_topr = int(getattr(config, "msgf_topr", 32))
        self.msgf_frame_topk_max = int(getattr(config, "msgf_frame_topk_max", 3))
        self.msgf_atom_topk_max = int(getattr(config, "msgf_atom_topk_max", 8))
        self.temporal_bonus_lambda = float(getattr(config, "temporal_bonus_lambda", 0.10))
        self.refiner = MemoryRefiner(
            hidden_size=config.hidden_size,
            use_gate=True,
            residual=True,
        )

    def forward(self, semantic_hidden, vggt_features, grid_thw=None, text_hidden_states=None, **kwargs):
        local_delta, layout, _, geo_frames = self._local_frame_fusion(semantic_hidden, vggt_features, grid_thw)
        fused_hidden = semantic_hidden + local_delta
        fused_frames = self._split_frames(fused_hidden, layout)

        if self.layer_idx <= self.stage_ranges.warmup_end:
            return local_delta

        # Build atoms and memory bank
        frame_atoms = [
            self._select_frame_atoms(frame_tokens, geo_tokens, self.msgf_topr)
            for frame_tokens, geo_tokens in zip(fused_frames, geo_frames)
        ]
        frame_summaries = [mean_pool_tokens(atoms) for atoms in frame_atoms if atoms.numel() > 0]

        memory_updates = []
        frame_topk = 0
        atom_topk = 0

        for frame_idx, frame_tokens in enumerate(fused_frames):
            if self.stage_ranges.write_start <= self.layer_idx <= self.stage_ranges.write_end:
                # Write stage: self-memory update with own atoms
                context = frame_atoms[frame_idx]
                frame_topk = 1
                atom_topk = max(atom_topk, int(context.shape[0]))
                memory_updates.append(self._memory_update(frame_tokens, context))
            else:
                # Read stage: temporal-aware retrieval + atom refinement
                query = self._build_frame_query(frame_tokens, text_hidden_states)

                # Frame-level coarse retrieval with temporal bonus
                frame_bank = torch.cat(frame_summaries, dim=0)
                frame_scores = _similarity(query, frame_bank)
                for j in range(len(frame_summaries)):
                    frame_scores[j] = frame_scores[j] + self.temporal_bonus_lambda / (1.0 + abs(frame_idx - j))
                _, top_frame_indices = safe_topk(frame_scores, self.msgf_frame_topk_max)
                frame_topk = max(frame_topk, int(top_frame_indices.numel()))

                # Gather candidate atoms from selected frames
                candidate_frames = [frame_atoms[idx] for idx in top_frame_indices.tolist()]
                candidate_atoms = torch.cat(candidate_frames, dim=0)

                # Lightweight semantic-geometric atom refinement
                geo_summary = mean_pool_tokens(geo_frames[frame_idx]) if geo_frames[frame_idx].numel() > 0 else query
                context_signal = query + geo_summary
                refined_atoms = self.refiner(candidate_atoms, context_signal)

                # Atom-level fine retrieval on refined atoms
                atom_scores = _similarity(query, refined_atoms)
                _, atom_indices = safe_topk(atom_scores, self.msgf_atom_topk_max)
                selected_atoms = refined_atoms[atom_indices]
                atom_topk = max(atom_topk, int(atom_indices.numel()))

                memory_updates.append(self._memory_update(frame_tokens, selected_atoms))

        self._log_memory_stats(
            tag="zenview",
            available_frames=len(frame_atoms),
            available_atoms=sum(int(a.shape[0]) for a in frame_atoms),
            frame_topk=frame_topk,
            atom_topk=atom_topk,
        )
        return local_delta + torch.cat(memory_updates, dim=0).unsqueeze(0)
