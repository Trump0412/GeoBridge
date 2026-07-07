"""VGGT multi-layer extractor for the ZenView geometry bank."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..msgf_utils import FrameLayout


@dataclass
class VGGTBankFeatureOutput:
    """Aligned multi-layer VGGT tokens before geometry projection."""

    layer_tokens: Dict[str, torch.Tensor]
    frame_layout: FrameLayout
    patch_grid: Tuple[int, int]
    merged_grid: Tuple[int, int]

    @property
    def token_dim(self) -> int:
        first_key = next(iter(self.layer_tokens))
        return int(self.layer_tokens[first_key].shape[-1])


class VGGTBankExtractor(nn.Module):
    """Read aligned VGGT layer features for the geometry bank."""

    LAYER_NAME_MAP = {
        11: "g11_raw",
        17: "g17_raw",
        23: "g23_raw",
    }

    @classmethod
    def layer_name(cls, layer_id: int) -> str:
        return cls.LAYER_NAME_MAP.get(int(layer_id), f"g{int(layer_id)}_raw")

    def __init__(
        self,
        model_path: str | None = None,
        layer_ids: Iterable[int] = (11, 17, 23),
        spatial_merge_size: int = 2,
        depart_smi_token: bool = False,
        smi_image_num: int = 8,
        smi_downsample_rate: int = 2,
        cache_vggt_features: bool = False,
        freeze_encoder: bool = True,
        reference_frame: str = "first",
    ):
        super().__init__()
        from ..vggt.models.vggt import VGGT

        self.vggt: nn.Module
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.spatial_merge_size = int(spatial_merge_size)
        self.depart_smi_token = bool(depart_smi_token)
        self.smi_image_num = int(smi_image_num)
        self.smi_downsample_rate = int(smi_downsample_rate)
        self.cache_vggt_features = bool(cache_vggt_features)
        self.freeze_encoder = bool(freeze_encoder)
        self.reference_frame = reference_frame
        self.patch_size = 14
        self._feature_cache: Dict[Tuple, VGGTBankFeatureOutput] = {}
        self._projector_input_dim: int | None = None

        if model_path:
            self.load_model(model_path)
        else:
            self.vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)

        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

    def load_model(self, model_path: str) -> None:
        from ..vggt.models.vggt import VGGT

        self.vggt = VGGT.from_pretrained(
            model_path,
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

    def get_projector_input_dims(self) -> Dict[str, int]:
        if self._projector_input_dim is None:
            raise RuntimeError("VGGTBankExtractor projector input dim is unknown before the first extraction.")
        return {
            self.layer_name(layer_id): self._projector_input_dim
            for layer_id in self.layer_ids
        }

    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images

    def _apply_inverse_reference_frame_transform(self, features: torch.Tensor) -> torch.Tensor:
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    def _apply_reference_frame_transform_batch(self, images: torch.Tensor) -> torch.Tensor:
        if self.reference_frame != "first":
            return torch.flip(images, dims=(1,))
        return images

    def _apply_inverse_reference_frame_transform_batch(self, features: torch.Tensor) -> torch.Tensor:
        if self.reference_frame != "first":
            return torch.flip(features, dims=(1,))
        return features

    def _build_cache_key(self, images: torch.Tensor) -> Tuple:
        checksum = float(images.detach().float().sum().item())
        return (
            tuple(images.shape),
            str(images.dtype),
            self.reference_frame,
            checksum,
        )

    def _align_patch_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        num_frames, patch_h, patch_w, hidden_dim = tokens.shape
        if self.depart_smi_token and num_frames > self.smi_image_num:
            target_h = max(patch_h // self.smi_downsample_rate, 1)
            target_w = max(patch_w // self.smi_downsample_rate, 1)
            tokens = F.interpolate(
                tokens.permute(0, 3, 1, 2),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            patch_h, patch_w = target_h, target_w

        merge = self.spatial_merge_size
        aligned_h = max((patch_h // merge) * merge, merge)
        aligned_w = max((patch_w // merge) * merge, merge)
        tokens = tokens[:, :aligned_h, :aligned_w, :]
        tokens = tokens.reshape(
            num_frames,
            aligned_h // merge,
            merge,
            aligned_w // merge,
            merge,
            hidden_dim,
        )
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).contiguous()
        tokens = tokens.reshape(
            num_frames,
            aligned_h // merge,
            aligned_w // merge,
            hidden_dim * merge * merge,
        )
        return tokens, (patch_h, patch_w), (aligned_h // merge, aligned_w // merge)

    def extract(self, images: torch.Tensor) -> VGGTBankFeatureOutput:
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [T, C, H, W], got {tuple(images.shape)}")

        images = self._apply_reference_frame_transform(images)
        cache_key = self._build_cache_key(images) if self.cache_vggt_features and not self.training else None
        if cache_key is not None and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        if self.freeze_encoder:
            self.vggt.eval()
            grad_context = torch.no_grad()
        else:
            self.vggt.train(self.training)
            grad_context = contextlib.nullcontext()

        autocast_enabled = images.is_cuda
        autocast_dtype = torch.bfloat16 if autocast_enabled and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with grad_context:
            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled, dtype=autocast_dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])

        num_frames, _, height, width = images.shape
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size

        layer_tokens: Dict[str, torch.Tensor] = {}
        merged_grid = None
        for layer_id in self.layer_ids:
            layer_idx = int(layer_id) - 1
            if layer_idx < 0 or layer_idx >= len(aggregated_tokens_list):
                raise ValueError(f"Requested VGGT layer {layer_id}, but only {len(aggregated_tokens_list)} layers are available.")

            tokens = aggregated_tokens_list[layer_idx][0, :, patch_start_idx:]
            tokens = self._apply_inverse_reference_frame_transform(tokens)
            tokens = tokens.reshape(num_frames, patch_h, patch_w, -1)
            tokens, _, merged_grid = self._align_patch_tokens(tokens)
            tokens = tokens.reshape(num_frames, -1, tokens.shape[-1])
            layer_tokens[self.layer_name(layer_id)] = tokens

        token_count = int(layer_tokens[next(iter(layer_tokens))].shape[1])
        frame_layout = FrameLayout(
            token_counts=[token_count] * num_frames,
            frame_shapes=[merged_grid] * num_frames,
        )
        output = VGGTBankFeatureOutput(
            layer_tokens=layer_tokens,
            frame_layout=frame_layout,
            patch_grid=(patch_h, patch_w),
            merged_grid=merged_grid,
        )
        self._projector_input_dim = output.token_dim

        if cache_key is not None:
            self._feature_cache[cache_key] = output
        return output

    def extract_batch(self, image_batch: torch.Tensor) -> List[VGGTBankFeatureOutput]:
        if image_batch.ndim != 5:
            raise ValueError(f"Expected image_batch with shape [B, T, C, H, W], got {tuple(image_batch.shape)}")

        image_batch = self._apply_reference_frame_transform_batch(image_batch)
        if self.freeze_encoder:
            self.vggt.eval()
            grad_context = torch.no_grad()
        else:
            self.vggt.train(self.training)
            grad_context = contextlib.nullcontext()

        autocast_enabled = image_batch.is_cuda
        autocast_dtype = torch.bfloat16 if autocast_enabled and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with grad_context:
            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled, dtype=autocast_dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(image_batch)

        batch_size, num_frames, _, height, width = image_batch.shape
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        outputs_per_layer: Dict[str, List[torch.Tensor]] = {self.layer_name(layer_id): [] for layer_id in self.layer_ids}
        merged_grid = None

        for layer_id in self.layer_ids:
            layer_idx = int(layer_id) - 1
            if layer_idx < 0 or layer_idx >= len(aggregated_tokens_list):
                raise ValueError(f"Requested VGGT layer {layer_id}, but only {len(aggregated_tokens_list)} layers are available.")

            tokens = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            tokens = self._apply_inverse_reference_frame_transform_batch(tokens)
            tokens = tokens.reshape(batch_size, num_frames, patch_h, patch_w, -1)
            for batch_idx in range(batch_size):
                aligned_tokens, _, merged_grid = self._align_patch_tokens(tokens[batch_idx])
                outputs_per_layer[self.layer_name(layer_id)].append(
                    aligned_tokens.reshape(num_frames, -1, aligned_tokens.shape[-1])
                )

        outputs: List[VGGTBankFeatureOutput] = []
        for batch_idx in range(batch_size):
            layer_tokens = {
                name: values[batch_idx]
                for name, values in outputs_per_layer.items()
            }
            token_count = int(layer_tokens[next(iter(layer_tokens))].shape[1])
            frame_layout = FrameLayout(
                token_counts=[token_count] * num_frames,
                frame_shapes=[merged_grid] * num_frames,
            )
            output = VGGTBankFeatureOutput(
                layer_tokens=layer_tokens,
                frame_layout=frame_layout,
                patch_grid=(patch_h, patch_w),
                merged_grid=merged_grid,
            )
            outputs.append(output)

        if outputs:
            self._projector_input_dim = outputs[0].token_dim
        return outputs

    def forward(self, images: torch.Tensor) -> VGGTBankFeatureOutput:
        return self.extract(images)
