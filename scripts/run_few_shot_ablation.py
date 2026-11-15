from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wadipulse.config import Settings
from wadipulse.mlops.tracking import ExperimentTracker
from wadipulse.vision.gfm_prithvi import PrithviFewShotSegmentor
from wadipulse.vision.unet_baseline import UNetBaseline


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(few_shot_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    manifest = json.loads((few_shot_dir / "manifest.json").read_text(encoding="utf-8"))
    cubes: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    regions: list[str] = []
    for entry in manifest:
        npz = np.load(few_shot_dir / entry["file"])
        cubes.append(npz["cube"].astype(np.float32))
        masks.append(npz["mask"].astype(np.float32))
        regions.append(str(entry["region"]))
    return np.stack(cubes, axis=0), np.stack(masks, axis=0), regions


def stratified_split(
    cubes: np.ndarray, masks: np.ndarray, regions: list[str], test_count: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wadi_idx = [i for i, r in enumerate(regions) if r == "wadi_ad_dawasir"]
    kharj_idx = [i for i, r in enumerate(regions) if r == "al_kharj"]

    test_idx = wadi_idx[:4] + kharj_idx[: (test_count - 4)]
    test_set = set(test_idx)
    train_idx = [i for i in range(len(regions)) if i not in test_set]
    return cubes[train_idx], masks[train_idx], cubes[test_idx], masks[test_idx]


def augment_batch(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = random.randint(0, 3)
    if k > 0:
        x = torch.rot90(x, k, dims=(-2, -1))
        y = torch.rot90(y, k, dims=(-2, -1))
    if random.random() > 0.5:
        x = torch.flip(x, dims=(-1,))
        y = torch.flip(y, dims=(-1,))
    return x, y


class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6
        dice_loss = 1.0 - ((2.0 * inter + 1e-6) / union).mean()
        return 0.5 * bce_loss + 0.5 * dice_loss


@torch.no_grad()
def evaluate_model(
    model: nn.Module, test_x: torch.Tensor, test_y: torch.Tensor, device: torch.device
) -> dict[str, float]:
    model.eval()
    logits = model(test_x.to(device))
    preds = (torch.sigmoid(logits) > 0.5).float().cpu()
    targets = test_y.cpu()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3)) + 1e-6
    iou = float((inter / union).mean().item())

    f1 = float(
        ((2.0 * inter + 1e-6) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6))
        .mean()
        .item()
    )

    pred_area = preds.sum(dim=(1, 2, 3))
    true_area = targets.sum(dim=(1, 2, 3)).clamp(min=1.0)
    area_mape_pct = float(((pred_area - true_area).abs() / true_area * 100.0).mean().item())

    return {
        "test_iou": round(iou, 4),
        "test_f1": round(f1, 4),
        "area_mape_pct": round(area_mape_pct, 2),
    }


def train_and_eval(
    model_name: str,
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    shots: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[dict[str, float], nn.Module]:
    model = model.to(device)
    criterion = BCEDiceLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    sub_x = torch.from_numpy(train_x[:shots])
    sub_y = torch.from_numpy(train_y[:shots]).unsqueeze(1)
    t_test_x = torch.from_numpy(test_x)
    t_test_y = torch.from_numpy(test_y).unsqueeze(1)

    ds = TensorDataset(sub_x, sub_y)
    loader = DataLoader(ds, batch_size=min(4, shots), shuffle=True)

    for _ in range(epochs):
        model.train()
        if hasattr(model, "encoder"):
            model.encoder.eval()
        for bx, by in loader:
            bx, by = augment_batch(bx, by)
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    metrics = evaluate_model(model, t_test_x, t_test_y, device)
    return metrics, model


def main() -> None:
    seed_everything(42)
    settings = Settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        _ = torch.empty(1, device=device)
    logger.info(f"Running few-shot ablation on device={device}")

    cubes, masks, regions = load_dataset(settings.few_shot_dir)
    train_x, train_y, test_x, test_y = stratified_split(cubes, masks, regions, test_count=7)
    logger.info(f"Dataset split: train={len(train_x)} tiles, test={len(test_x)} tiles")

    tracker = ExperimentTracker(settings)
    shot_budgets = [5, 10, len(train_x)]
    summary_rows: list[dict[str, object]] = []

    shared_prithvi = PrithviFewShotSegmentor(model_id=settings.gfm_model_id, freeze_encoder=True)
    shared_encoder = shared_prithvi.encoder

    for shots in shot_budgets:
        seed_everything(42)
        unet = UNetBaseline(in_channels=6, base_ch=32)
        unet_metrics, _ = train_and_eval(
            model_name="unet_baseline",
            model=unet,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            shots=shots,
            epochs=35,
            lr=1e-3,
            device=device,
        )
        tracker.log_ablation_run(
            run_name=f"unet_{shots}shot",
            params={"model": "UNetBaseline", "shots": shots, "epochs": 35, "lr": 1e-3},
            metrics=unet_metrics,
        )
        summary_rows.append({"model": "UNet-6Band (Baseline)", "shots": shots, **unet_metrics})

        seed_everything(42)
        prithvi = PrithviFewShotSegmentor.__new__(PrithviFewShotSegmentor)
        nn.Module.__init__(prithvi)
        prithvi.encoder = shared_encoder
        prithvi.register_buffer("mean", shared_prithvi.mean.clone())
        prithvi.register_buffer("std", shared_prithvi.std.clone())
        from wadipulse.vision.gfm_prithvi import ConvUpsampleBlock

        prithvi.proj = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        prithvi.up1 = ConvUpsampleBlock(256, 128)
        prithvi.up2 = ConvUpsampleBlock(128, 64)
        prithvi.up3 = ConvUpsampleBlock(64, 32)
        prithvi.up4 = ConvUpsampleBlock(32, 16)
        prithvi.head = nn.Conv2d(16, 1, kernel_size=1)

        prithvi_metrics, trained_prithvi = train_and_eval(
            model_name="prithvi_eo_2_0",
            model=prithvi,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            shots=shots,
            epochs=35,
            lr=1e-3,
            device=device,
        )
        ckpt_path = settings.artifacts_dir / f"prithvi_decoder_{shots}shot.pt"
        decoder_state = {
            k: v.cpu()
            for k, v in trained_prithvi.state_dict().items()
            if not k.startswith("encoder.")
        }
        torch.save(decoder_state, ckpt_path)

        tracker.log_ablation_run(
            run_name=f"prithvi_eo_2_0_{shots}shot",
            params={"model": settings.gfm_model_id, "shots": shots, "epochs": 35, "lr": 1e-3},
            metrics=prithvi_metrics,
            artifact_paths=[ckpt_path],
        )
        summary_rows.append({"model": "Prithvi-EO-2.0-300M", "shots": shots, **prithvi_metrics})

    summary_path = settings.artifacts_dir / "ablation_results.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print("\n=== Few-Shot Segmentation Benchmark (Saudi Center-Pivot Fields) ===")
    print(f"{'Model':<24} | {'Shots':>5} | {'Test IoU':>8} | {'Test F1':>8} | {'Area MAPE %':>11}")
    print("-" * 68)
    for row in summary_rows:
        print(
            f"{row['model']:<24} | {row['shots']:>5} | "
            f"{row['test_iou']:>8.4f} | {row['test_f1']:>8.4f} | {row['area_mape_pct']:>10.2f}%"
        )


if __name__ == "__main__":
    main()
