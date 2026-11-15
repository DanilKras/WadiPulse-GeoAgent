from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from loguru import logger
from timm.layers import set_fused_attn

set_fused_attn(False)
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

PRITHVI_MEAN = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
PRITHVI_STD = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)


class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_prithvi_encoder(model_id: str) -> tuple[nn.Module, int]:
    code_path = hf_hub_download(repo_id=model_id, filename="prithvi_mae.py")
    cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
    weights_path = hf_hub_download(repo_id=model_id, filename="Prithvi_EO_V2_300M.pt")

    spec = importlib.util.spec_from_file_location("prithvi_mae_hub", code_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load prithvi_mae.py from {code_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    pretrained_cfg = raw_cfg.get("pretrained_cfg", raw_cfg)

    encoder = module.PrithviViT(
        img_size=pretrained_cfg.get("img_size", 224),
        patch_size=tuple(pretrained_cfg.get("patch_size", [1, 16, 16])),
        num_frames=1,
        in_chans=pretrained_cfg.get("in_chans", 6),
        embed_dim=pretrained_cfg.get("embed_dim", 1024),
        depth=pretrained_cfg.get("depth", 24),
        num_heads=pretrained_cfg.get("num_heads", 16),
        mlp_ratio=pretrained_cfg.get("mlp_ratio", 4.0),
        coords_encoding=[],
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    encoder_weights = {
        k.removeprefix("encoder."): v
        for k, v in state_dict.items()
        if k.startswith("encoder.") and "pos_embed" not in k
    }
    missing, unexpected = encoder.load_state_dict(encoder_weights, strict=False)
    logger.info(f"Loaded {model_id} encoder (missing={len(missing)}, unexpected={len(unexpected)})")
    embed_dim = int(pretrained_cfg.get("embed_dim", 1024))
    return encoder, embed_dim


class PrithviFewShotSegmentor(nn.Module):
    def __init__(
        self,
        model_id: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder, embed_dim = _load_prithvi_encoder(model_id)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        mean = torch.tensor(PRITHVI_MEAN, dtype=torch.float32).view(1, 6, 1, 1)
        std = torch.tensor(PRITHVI_STD, dtype=torch.float32).view(1, 6, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.up1 = ConvUpsampleBlock(256, 128)
        self.up2 = ConvUpsampleBlock(128, 64)
        self.up3 = ConvUpsampleBlock(64, 32)
        self.up4 = ConvUpsampleBlock(32, 16)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = x * 10000.0
        return (x_scaled - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_norm = self._normalize(x).unsqueeze(2)

        with torch.no_grad():
            feats = self.encoder.forward_features(x_norm)
            mid_tokens = feats[len(feats) // 2][:, 1:, :]
            top_tokens = feats[-1][:, 1:, :]

        grid_h, grid_w = h // 16, w // 16
        mid_map = mid_tokens.transpose(1, 2).reshape(b, -1, grid_h, grid_w)
        top_map = top_tokens.transpose(1, 2).reshape(b, -1, grid_h, grid_w)

        fused = self.proj(torch.cat([mid_map, top_map], dim=1))
        out = self.up4(self.up3(self.up2(self.up1(fused))))
        logits = self.head(out)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits
