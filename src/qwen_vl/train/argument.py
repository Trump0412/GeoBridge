import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)  # Whether to use 3D geometry encoder
    geometry_encoder_type: str = field(default="vggt")  # Type of geometry encoder ("vggt", "pi3", "da3")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")  # Path to pre-trained geometry encoder model
    geometry_encoder_freeze: bool = field(default=True)
    reference_frame: str = field(default="first")  # Reference frame for geometry encoding ("first", "last"), only available for vggt
    feature_fusion_method: str = field(default="add")  # Method to fuse geometry and visual features ("add", "concat", "cross_attention", "gate")
    fusion_num_layers: int = field(default=1)  # Number of layers in the cross-attention module when feature_fusion_method is "cross_attention"
    geometry_merger_type: str = field(default="mlp")
    geo_encoder_out_layer_index: int = field(default=-1,
        metadata={"help": "Which out_layer to extract from DA3 backbone. -1=last, 0=first."})  # Type of geometry feature merger ("mlp", "avg")

    geo_cross_attn: bool = field(default=False)
    geo_inject_version: str = field(default="v2_flash")
    geo_importance_gate: bool = field(default=False)
    use_qwenvl_loss: bool = field(default=False)
    msgf_debug: bool = field(default=False)
    bank_debug: bool = field(default=False)

    vggt_bank_layers: str = field(default="11,17,23")
    vggt_bank_d_geom: int = field(default=1024)
    vggt_bank_topk: int = field(
        default=2,
        metadata={"help": "Top-k for generic ZenView geometry-bank router. SpatialFit HGB uses hgb_local_topk instead."},
    )
    vggt_bank_num_layers: int = field(default=20)
    vggt_bank_fusion_layer_indices: str = field(default="")
    vggt_bank_use_layer_embedding: bool = field(default=True)
    stage1_checkpoint_path: str = field(default="")
    freeze_projector: bool = field(default=False)
    freeze_base_geometry_fusion: bool = field(default=False)
    freeze_continuity_builder: bool = field(default=False)
    freeze_geometry_decoder: bool = field(default=False)
    normalize_query: bool = field(default=False)
    normalize_bank: bool = field(default=False)
    bank_temperature: float = field(default=1.0)
    candidate_dropout_enabled: bool = field(default=False)
    g11_drop_prob: float = field(default=0.0)
    g17_drop_prob: float = field(default=0.0)
    g23_drop_prob: float = field(default=0.0)
    continuity_drop_prob: float = field(default=0.0)
    use_continuity: bool = field(default=True)
    continuity_radius: int = field(default=1)
    continuity_use_spatial_neighbors: bool = field(default=False)
    continuity_mlp_hidden_ratio: float = field(default=2.0)
    continuity_attention_heads: int = field(default=4)
    bank_gate_mode: str = field(default="scalar")
    cache_vggt_features: bool = field(default=False)
    freeze_continuity_selector: bool = field(default=True)
    freeze_activated_corr_graph: bool = field(default=True)
    hgb_use_saliency_prior: bool = field(default=True)
    hgb_local_topk: int = field(
        default=2,
        metadata={"help": "Top-k for SpatialFit HGB local router over g11/g17/g23."},
    )
    hgb_corr_topk_neighbors: int = field(default=8)
    hgb_temporal_radius: int = field(default=2)
    hgb_layer_scale_init: float = field(default=0.05)
    hgb_gate_none_bias: float = field(default=0.0)
    hgb_gate_local_bias: float = field(default=0.4)
    hgb_gate_cont_bias: float = field(default=0.6)
    hgb_use_gate_bias_init: bool = field(default=True)
    hgb_layer0_g11_logit_bias: float = field(default=2.0)
    hgb_strict_alignment: bool = field(default=True)
    hgb_allow_layout_fallback: bool = field(default=False)
    hgb_alignment_audit_only: bool = field(default=False)
    hgb_min_overlap_ratio: float = field(default=1.0)

    msgf_topr: int = field(default=32)
    msgf_frame_topk_max: int = field(default=3)
    msgf_atom_topk_max: int = field(default=8)
    msgf_use_bidirectional: bool = field(default=True)
    msgf_warmup_start: int = field(default=-1)
    msgf_warmup_end: int = field(default=-1)
    msgf_write_start: int = field(default=-1)
    msgf_write_end: int = field(default=-1)
    msgf_read_start: int = field(default=-1)
    msgf_read_end: int = field(default=-1)

    hmsgf_frame_topk_max: int = field(default=3)
    hmsgf_region_topr: int = field(default=32)
    hmsgf_region_topk_max: int = field(default=8)
    hmsgf_warmup_start: int = field(default=-1)
    hmsgf_warmup_end: int = field(default=-1)
    hmsgf_write_start: int = field(default=-1)
    hmsgf_write_end: int = field(default=-1)
    hmsgf_read_start: int = field(default=-1)
    hmsgf_read_end: int = field(default=-1)

    rmsgf_topr: int = field(default=32)
    rmsgf_atom_topk_max: int = field(default=8)
    rmsgf_refine_gate: bool = field(default=True)
    rmsgf_refine_residual: bool = field(default=True)
    rmsgf_init_start: int = field(default=-1)
    rmsgf_init_end: int = field(default=-1)
    rmsgf_refine_start: int = field(default=-1)
    rmsgf_refine_end: int = field(default=-1)

    # ZenView (da3_msgf_temporal_refine) params
    temporal_bonus_lambda: float = field(default=0.10)

    mmr_debug: bool = field(default=False)
    mmr_use_region_memory: bool = field(default=False)
    mmr_frame_topk_max: int = field(default=3)
    mmr_region_topk_max: int = field(default=8)
    mmr_warmup_start: int = field(default=0)
    mmr_warmup_end: int = field(default=-1)
    mmr_write_start: int = field(default=-1)
    mmr_write_end: int = field(default=-1)
    mmr_read_start: int = field(default=-1)
    mmr_read_end: int = field(default=-1)
    mmr_use_view_continuity: bool = field(default=True)
    mmr_use_temporal_continuity: bool = field(default=True)
    mmr_memory_dim: int = field(default=0)
    mmr_region_atoms_per_frame: int = field(default=8)
    mmr_query_use_text: bool = field(default=True)
    mmr_query_use_visual_summary: bool = field(default=True)

    depart_smi_token: bool = field(default=False)
    smi_image_num: int = field(default=8)
    smi_downsample_rate: int = field(default=2)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)
    geometry_cache_dir: str = field(default="")
    geometry_cache_manifest: str = field(default="")
    geometry_cache_use: bool = field(default=False)
    geometry_cache_required: bool = field(default=False)


def build_hgb_effective_config(model_args: ModelArguments, data_args: Optional["DataArguments"] = None) -> Dict[str, object]:
    config = {
        "geo_inject_version": getattr(model_args, "geo_inject_version", ""),
        "vggt_bank_layers": getattr(model_args, "vggt_bank_layers", ""),
        "vggt_bank_d_geom": getattr(model_args, "vggt_bank_d_geom", 0),
        "vggt_bank_num_layers": getattr(model_args, "vggt_bank_num_layers", 0),
        "vggt_bank_fusion_layer_indices": getattr(model_args, "vggt_bank_fusion_layer_indices", ""),
        "use_continuity": getattr(model_args, "use_continuity", False),
        "continuity_radius": getattr(model_args, "continuity_radius", 0),
        "continuity_use_spatial_neighbors": getattr(model_args, "continuity_use_spatial_neighbors", False),
        "continuity_mlp_hidden_ratio": getattr(model_args, "continuity_mlp_hidden_ratio", 0.0),
        "continuity_attention_heads": getattr(model_args, "continuity_attention_heads", 0),
        "bank_gate_mode": getattr(model_args, "bank_gate_mode", ""),
        "stage1_checkpoint_path": getattr(model_args, "stage1_checkpoint_path", ""),
        "freeze_projector": getattr(model_args, "freeze_projector", False),
        "freeze_base_geometry_fusion": getattr(model_args, "freeze_base_geometry_fusion", False),
        "freeze_continuity_builder": getattr(model_args, "freeze_continuity_builder", False),
        "freeze_geometry_decoder": getattr(model_args, "freeze_geometry_decoder", False),
        "freeze_continuity_selector": getattr(model_args, "freeze_continuity_selector", False),
        "freeze_activated_corr_graph": getattr(model_args, "freeze_activated_corr_graph", False),
        "normalize_query": getattr(model_args, "normalize_query", False),
        "normalize_bank": getattr(model_args, "normalize_bank", False),
        "bank_temperature": getattr(model_args, "bank_temperature", 0.0),
        "hgb_use_saliency_prior": getattr(model_args, "hgb_use_saliency_prior", False),
        "hgb_local_topk": getattr(model_args, "hgb_local_topk", 0),
        "hgb_corr_topk_neighbors": getattr(model_args, "hgb_corr_topk_neighbors", 0),
        "hgb_temporal_radius": getattr(model_args, "hgb_temporal_radius", 0),
        "hgb_layer_scale_init": getattr(model_args, "hgb_layer_scale_init", 0.0),
        "hgb_gate_none_bias": getattr(model_args, "hgb_gate_none_bias", 0.0),
        "hgb_gate_local_bias": getattr(model_args, "hgb_gate_local_bias", 0.0),
        "hgb_gate_cont_bias": getattr(model_args, "hgb_gate_cont_bias", 0.0),
        "hgb_use_gate_bias_init": getattr(model_args, "hgb_use_gate_bias_init", False),
        "hgb_layer0_g11_logit_bias": getattr(model_args, "hgb_layer0_g11_logit_bias", 0.0),
        "hgb_strict_alignment": getattr(model_args, "hgb_strict_alignment", False),
        "hgb_allow_layout_fallback": getattr(model_args, "hgb_allow_layout_fallback", False),
        "hgb_alignment_audit_only": getattr(model_args, "hgb_alignment_audit_only", False),
        "hgb_min_overlap_ratio": getattr(model_args, "hgb_min_overlap_ratio", 1.0),
    }
    if data_args is not None:
        config.update(
            {
                "geometry_cache_use": getattr(data_args, "geometry_cache_use", False),
                "geometry_cache_required": getattr(data_args, "geometry_cache_required", False),
                "geometry_cache_dir": getattr(data_args, "geometry_cache_dir", ""),
                "geometry_cache_manifest": getattr(data_args, "geometry_cache_manifest", ""),
            }
        )
    return config


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
