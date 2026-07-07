"""DA3 encoder variant using blocks_to_take (concatenated local+global) features.

Instead of using the aux_output path (export_feat_layers) which captures only the
global-attention output `x` with dimension embed_dim, this encoder hooks the DINOv2
backbone to capture the main output path: `cat(local_x, x)` with dimension
2*embed_dim.  This is the same feature the DA3 depth head (DPT) consumes.

Config kwargs (via encoder_kwargs):
    out_layer_index (int): Which out_layer entry to extract.
        Default -1 (last block, e.g. layer 39 for vitg).
        0 selects the first out_layer (e.g. layer 19 for vitg).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from .da3_encoder import DA3Encoder
from .base import GeometryEncoderConfig

logger = logging.getLogger(__name__)


class DA3NewEncoder(DA3Encoder):
    """DA3 encoder that extracts blocks_to_take features (2x embed_dim).

    The DINOv2 backbone in DA3 produces two types of features:
      - blocks_to_take (output): cat(local_x, x), dim = 2*embed_dim
        → used by DA3's own DPT depth head
      - export_feat_layers (aux): just x, dim = embed_dim
        → what the original DA3Encoder uses

    This encoder uses a forward hook on the backbone to capture the richer
    blocks_to_take features before they are consumed by the depth head.
    """

    def __init__(self, config: GeometryEncoderConfig):
        # Let parent handle DA3 model loading, freezing, etc.
        super().__init__(config)

        kwargs = config.encoder_kwargs or {}
        self.out_layer_index = int(kwargs.get("out_layer_index", -1))

        # Storage for the hook-captured backbone output
        self._captured_backbone_output = None
        self._hook_handle = None

        # Override feature dim: blocks_to_take gives 2*embed_dim when cat_token=True
        self._infer_blocks_to_take_dim()
        # Install the hook
        self._install_backbone_hook()

        logger.info(
            f"DA3NewEncoder initialised: out_layer_index={self.out_layer_index}, "
            f"feature_dim={self._feature_dim}"
        )

    # ------------------------------------------------------------------
    # Dimension inference
    # ------------------------------------------------------------------
    def _infer_blocks_to_take_dim(self) -> None:
        """Set feature_dim = 2*embed_dim when cat_token is True."""
        backbone = getattr(getattr(self.da3, "model", None), "backbone", None)
        if backbone is None:
            logger.warning("Cannot locate DA3 backbone; feature_dim may be incorrect.")
            return

        pretrained = getattr(backbone, "pretrained", backbone)
        embed_dim = getattr(pretrained, "embed_dim", None)
        cat_token = getattr(backbone, "cat_token", True)

        if embed_dim is not None:
            self._feature_dim = embed_dim * 2 if cat_token else embed_dim
            logger.info(
                f"Inferred blocks_to_take dim: embed_dim={embed_dim}, "
                f"cat_token={cat_token} -> feature_dim={self._feature_dim}"
            )

    # ------------------------------------------------------------------
    # Forward hook on backbone
    # ------------------------------------------------------------------
    def _install_backbone_hook(self) -> None:
        """Register a forward hook on the DinoV2 backbone module."""
        backbone = getattr(getattr(self.da3, "model", None), "backbone", None)
        if backbone is None:
            raise RuntimeError(
                "Cannot find DA3 backbone at self.da3.model.backbone. "
                "Make sure the DA3 model is loaded correctly."
            )

        def _capture_hook(_module, _input, output):
            # DinoV2.forward() returns (feats, aux_feats)
            # feats = tuple of (patch_features, camera_token) per out_layer
            self._captured_backbone_output = output

        self._hook_handle = backbone.register_forward_hook(_capture_hook)

    def _remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ------------------------------------------------------------------
    # Feature extraction from hooked output
    # ------------------------------------------------------------------
    def _extract_backbone_features(self, n_image: int) -> torch.Tensor:
        """Extract the selected out_layer features from captured backbone output.

        The backbone output format (from DinoV2.forward / get_intermediate_layers):
            (feats, aux_feats)
        where feats = tuple of (patch_features, camera_token) per out_layer.

        For vitg with out_layers=[19, 27, 33, 39]:
            feats[0] = (layer19_patches, layer19_cam)  # [B, S, N_patch, 3072]
            feats[1] = (layer27_patches, layer27_cam)
            feats[2] = (layer33_patches, layer33_cam)
            feats[3] = (layer39_patches, layer39_cam)

        Returns:
            features: [n_image, num_patches, feat_dim]
        """
        if self._captured_backbone_output is None:
            raise RuntimeError(
                "Backbone hook did not capture output. "
                "Ensure _run_da3() was called before _extract_backbone_features()."
            )

        feats, _aux_feats = self._captured_backbone_output
        self._captured_backbone_output = None  # release memory immediately

        num_out_layers = len(feats)
        idx = self.out_layer_index
        if idx < 0:
            idx = num_out_layers + idx
        if not (0 <= idx < num_out_layers):
            raise IndexError(
                f"out_layer_index={self.out_layer_index} is invalid for "
                f"{num_out_layers} out_layers"
            )

        # feats[idx] = (patch_features, camera_token)
        # patch_features shape: [B, S, num_patches, feat_dim]
        patch_features = feats[idx][0]

        # Remove batch dim (B=1) -> [S, num_patches, feat_dim]
        if patch_features.dim() == 4:
            patch_features = patch_features.squeeze(0)

        if patch_features.shape[0] != n_image:
            raise ValueError(
                f"Frame count mismatch: expected {n_image}, "
                f"got {patch_features.shape[0]} from backbone output"
            )

        return patch_features

    # ------------------------------------------------------------------
    # Override encode() to use blocks_to_take features
    # ------------------------------------------------------------------
    def encode(self, images: torch.Tensor):
        """Encode images using blocks_to_take features (local+global, 2x embed_dim).

        Args:
            images: [N, C, H, W] input frames

        Returns:
            features: [N, num_patches, feat_dim] where feat_dim = 2*embed_dim
            camera_token: [N, 1, feat_dim] mean-pooled summary per frame
        """
        if images.dim() != 4:
            raise ValueError(
                f"DA3NewEncoder expects input [N, C, H, W], got shape {tuple(images.shape)}"
            )

        images = self._apply_reference_frame_transform(images)
        n_image, _, height, width = images.shape
        images_5d = images.unsqueeze(0)  # [B=1, N, C, H, W]

        # Run DA3 forward — the hook captures backbone output
        try:
            _ = self._run_da3(images_5d)
        except Exception as exc:
            if self._resolve_export_feat_layer_with_fallback(exc):
                _ = self._run_da3(images_5d)
            else:
                raise

        # Extract blocks_to_take features from the hooked output
        features = self._extract_backbone_features(n_image)
        # features: [n_image, num_patches, feat_dim]

        h_patch = height // self.patch_size
        w_patch = width // self.patch_size
        expected_patches = h_patch * w_patch
        if features.shape[1] != expected_patches:
            raise ValueError(
                f"Patch count mismatch: backbone gave {features.shape[1]} patches, "
                f"expected {expected_patches} from image ({height}x{width}) / "
                f"patch_size={self.patch_size}"
            )

        features = features.contiguous()
        features = self._apply_inverse_reference_frame_transform(features)
        camera_token = features.mean(dim=1, keepdim=True)

        if self._feature_dim is None:
            self._feature_dim = int(features.shape[-1])

        return features, camera_token

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_model(self, model_path: str) -> None:
        """Reload DA3 model and re-install hook."""
        self._remove_hook()
        super().load_model(model_path)
        self._infer_blocks_to_take_dim()
        self._install_backbone_hook()

    def __del__(self):
        self._remove_hook()
