"""Depth Anything 3 geometry encoder implementation."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from .base import BaseGeometryEncoder, GeometryEncoderConfig


class DA3Encoder(BaseGeometryEncoder):
    """Depth Anything 3 geometry encoder wrapper."""

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)

        kwargs = config.encoder_kwargs or {}
        self.patch_size = int(kwargs.get("patch_size", 14))
        self.ref_view_strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
        self.use_ray_pose = bool(kwargs.get("use_ray_pose", False))
        self.export_feat_layer = int(kwargs.get("export_feat_layer", -1))
        self._runtime_export_feat_layer = self.export_feat_layer
        # DA3's depth head contains LayerNorms that are not stable under bf16/fp16 autocast
        # in our training path. Keep the frozen DA3 branch in fp32 and cast features back later.
        self.force_fp32 = bool(kwargs.get("force_fp32", True))
        self.default_model_path = "depth-anything/da3-base"

        feature_dim_override = kwargs.get("feature_dim")
        self._feature_dim = int(feature_dim_override) if feature_dim_override is not None else None

        self.model_path = config.model_path or self.default_model_path
        self.da3 = self._load_da3_model(self.model_path)

        if self.freeze_encoder:
            for param in self.da3.parameters():
                param.requires_grad = False

        self._infer_feature_dim_from_model()

    def _load_da3_model(self, model_path: Optional[str]):
        target_model = model_path or self.default_model_path
        try:
            # Keep import local so non-DA3 users do not require this dependency.
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "DA3 encoder requires `depth-anything-3`. "
                "Install it first (e.g. `pip install depth-anything-3` or editable install from thirdparty)."
            ) from exc

        model = DepthAnything3.from_pretrained(target_model)
        model.eval()
        return model

    def _infer_feature_dim_from_model(self) -> None:
        if self._feature_dim is not None:
            return

        candidates = (
            ("model", "backbone", "embed_dim"),
            ("model", "backbone", "num_features"),
            ("model", "backbone", "pretrained", "embed_dim"),
            ("model", "backbone", "pretrained", "num_features"),
            ("model", "head", "in_channels"),
        )
        for path in candidates:
            value: Any = self.da3
            for attr in path:
                if not hasattr(value, attr):
                    value = None
                    break
                value = getattr(value, attr)
            if isinstance(value, int) and value > 0:
                self._feature_dim = value
                return

    def _run_da3(self, images_5d: torch.Tensor):
        target_dtype = torch.float32 if self.force_fp32 else images_5d.dtype
        images_5d = images_5d.to(dtype=target_dtype)
        self.da3 = self.da3.to(device=images_5d.device, dtype=target_dtype)
        self.da3.eval()

        with torch.no_grad():
            with torch.autocast(
                device_type=images_5d.device.type,
                enabled=False,
            ):
                return self.da3(
                    images_5d,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=[self._runtime_export_feat_layer],
                    infer_gs=False,
                    use_ray_pose=self.use_ray_pose,
                    ref_view_strategy=self.ref_view_strategy,
                )

    def _extract_aux_feature(self, output: Any, feature_key: str) -> torch.Tensor:
        direct_map = output if isinstance(output, Mapping) else None
        aux_map = None

        if direct_map is not None and "aux" in direct_map and isinstance(direct_map["aux"], Mapping):
            aux_map = direct_map["aux"]
        elif hasattr(output, "aux") and isinstance(output.aux, Mapping):
            aux_map = output.aux

        if aux_map is not None and feature_key in aux_map:
            return aux_map[feature_key]

        if direct_map is not None and feature_key in direct_map:
            return direct_map[feature_key]

        available_keys = []
        if aux_map is not None:
            available_keys.extend([f"aux.{k}" for k in aux_map.keys()])
        if direct_map is not None:
            available_keys.extend(list(direct_map.keys()))
        raise KeyError(
            f"DA3 output missing '{feature_key}'. Available keys: {sorted(set(available_keys))}"
        )

    def _resolve_export_feat_layer_with_fallback(self, error: Exception) -> bool:
        if self._runtime_export_feat_layer != -1:
            return False

        backbone = getattr(getattr(self.da3, "model", None), "backbone", None)
        if backbone is None:
            return False

        last_layer = getattr(backbone, "pretrained", backbone)
        if not hasattr(last_layer, "n_blocks"):
            return False

        # DA3 expects explicit non-negative layer ids in some versions; fallback to the final block.
        self._runtime_export_feat_layer = int(last_layer.n_blocks) - 1
        return True

    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images

    def _apply_inverse_reference_frame_transform(self, features: torch.Tensor) -> torch.Tensor:
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    def encode(self, images: torch.Tensor):
        """Encode images using DA3 auxiliary features."""
        if images.dim() != 4:
            raise ValueError(f"DA3Encoder expects input [N, C, H, W], got shape {tuple(images.shape)}")

        images = self._apply_reference_frame_transform(images)
        n_image, _, height, width = images.shape
        images_5d = images.unsqueeze(0)  # [B=1, N, C, H, W]

        try:
            output = self._run_da3(images_5d)
        except Exception as exc:
            if self._resolve_export_feat_layer_with_fallback(exc):
                output = self._run_da3(images_5d)
            else:
                raise

        feat_key = f"feat_layer_{self._runtime_export_feat_layer}"
        feat = self._extract_aux_feature(output, feat_key)

        if feat.dim() != 5:
            raise ValueError(
                f"Expected DA3 feature '{feat_key}' shape [B, N, H, W, C], got {tuple(feat.shape)}"
            )
        if feat.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 after unsqueeze, got feature shape {tuple(feat.shape)}")
        if feat.shape[1] != n_image:
            raise ValueError(
                f"Feature frame count mismatch: input N={n_image}, output N={feat.shape[1]}"
            )

        h_patch, w_patch = feat.shape[2], feat.shape[3]
        expected_h_patch = height // self.patch_size
        expected_w_patch = width // self.patch_size
        if (h_patch, w_patch) != (expected_h_patch, expected_w_patch):
            raise ValueError(
                "DA3 patch grid mismatch: "
                f"expected ({expected_h_patch}, {expected_w_patch}) from input ({height}, {width}) "
                f"with patch_size={self.patch_size}, got ({h_patch}, {w_patch})."
            )

        features = feat[0].reshape(n_image, h_patch * w_patch, feat.shape[-1]).contiguous()
        features = self._apply_inverse_reference_frame_transform(features)
        camera_token = features.mean(dim=1, keepdim=True)

        if self._feature_dim is None:
            self._feature_dim = int(features.shape[-1])

        return features, camera_token

    def get_feature_dim(self) -> int:
        """Get DA3 feature dimension."""
        if self._feature_dim is None:
            raise ValueError(
                "DA3 feature_dim is not available yet. "
                "Pass encoder_kwargs['feature_dim'] or run one encode() call to infer it."
            )
        return self._feature_dim

    def forward(self, images: torch.Tensor):
        """Forward pass for compatibility."""
        return self.encode(images)

    def load_model(self, model_path: str) -> None:
        """Load pretrained DA3 model from a local path or Hub id."""
        self.model_path = model_path or self.default_model_path
        self._runtime_export_feat_layer = self.export_feat_layer
        self.da3 = self._load_da3_model(self.model_path)
        if self.freeze_encoder:
            for param in self.da3.parameters():
                param.requires_grad = False
        self._feature_dim = self.config.encoder_kwargs.get("feature_dim", self._feature_dim)
        if self._feature_dim is not None:
            self._feature_dim = int(self._feature_dim)
        self._infer_feature_dim_from_model()
