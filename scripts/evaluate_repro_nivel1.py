"""
Fase 1, Nivel 1 — reproducción por INFERENCIA de exp_hipo_004_finetune_ctfix_v4.

Carga epoch=014.ckpt del proyecto viejo en el framework nuevo (strict=True), corre
forward sobre el test de 40 y reproduce la capa 1 (MAE por estructura, dose score) y
los puntos DVH de capa 2 de metrics_summary.json. Lógica portada verbatim de
evaluate_hipo.py / evaluate.py / compute_gt_dvh_hipo.py del repo viejo; NO incluye
capa 3 (clasificación/AUC), NArcos ni plots.

Modos:
  (default)         inferencia con el checkpoint.
  --pred-dir DIR    triage §9: en vez de inferir, lee dosis predichas guardadas
                    (DIR/<anonid>.npy, (Z,H,W) en % Rx, mismo padding Z que el test).

Uso (python del venv viejo, que tiene torch+lightning+omegaconf):
    C:/Pablo/ProstateDoseProject/repo/.venv/Scripts/python.exe scripts/evaluate_repro_nivel1.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config.anatomy import load_experiment_config  # noqa: E402
from src.datamodules.dose_datamodule import DoseDataModule  # noqa: E402
from src.models.lightning_module import DosePredictionModule  # noqa: E402

PRESCRIPCION_GY = 70.0
OLD_REPO = Path("C:/Pablo/ProstateDoseProject")

# Puntos DVH que entran en la tabla target (capa 2) -> columna GT
GT_COL_MAP = {
    "PTV_D95Gy":     "PTV_D95_Gy_gt256",
    "PTV_V70Gy":     "PTV_V70Gy_pct_gt256",
    "Rectum_V65Gy":  "Rectum_V65Gy_gt256",
    "Bladder_V65Gy": "Bladder_V65Gy_gt256",
}


# ─── Portado de compute_gt_dvh_hipo.py ───────────────────────────────────────

def volumen_pct_sobre_umbral(dose_gy, mask, umbral_gy):
    vals = dose_gy[mask > 0]
    if len(vals) == 0:
        return float("nan")
    return 100.0 * float(np.mean(vals >= umbral_gy))


def dosis_percentil(dose_gy, mask, pct_volumen):
    """D_x: dosis recibida por al menos x% del volumen = percentil (100-x) de la dosis."""
    vals = dose_gy[mask > 0]
    if len(vals) == 0:
        return float("nan")
    return float(np.percentile(vals, 100.0 - pct_volumen))


# ─── Portado de evaluate_hipo.py / evaluate.py ───────────────────────────────

def calcular_puntos_dvh_pred(dose_pred_gy, ptv, rectum, bladder):
    out = {
        "PTV_D95Gy": dosis_percentil(dose_pred_gy, ptv, 95),
        "PTV_V70Gy": volumen_pct_sobre_umbral(dose_pred_gy, ptv, 70.0),
    }
    for nombre, mask in (("Rectum", rectum), ("Bladder", bladder)):
        out[f"{nombre}_V65Gy"] = volumen_pct_sobre_umbral(dose_pred_gy, mask, 65.0)
    return out


def mae_en_mascara(dose_pred_pct, dose_real_pct, mask):
    return float((np.abs(dose_pred_pct - dose_real_pct) * mask).sum() / max(mask.sum(), 1))


def dose_score_openkbp(dose_real, dose_pred, body):
    """MAE global dentro del BODY (en % de prescripción)."""
    diff = np.abs(dose_pred - dose_real) * body
    return float(diff.sum() / max(body.sum(), 1))


def bootstrap_ci(valores, n_boot, rng):
    n = len(valores)
    medias = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        medias[i] = np.nanmean(valores[idx])
    lo, hi = np.nanpercentile(medias, [2.5, 97.5])
    return [float(lo), float(hi)]


# ─── Carga del modelo (receta §5: strict=True, sin fallback) ─────────────────

def cargar_modelo_strict(checkpoint_path, cfg):
    module = DosePredictionModule(cfg)
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = module.load_state_dict(ck["state_dict"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    module.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return module.to(device), device


# ─── Evaluación ──────────────────────────────────────────────────────────────

def evaluar_dataset(model, device, loader, gt_dvh, desc, pred_dir=None):
    filas = []
    for batch in tqdm(loader, desc=desc):
        anonid = batch["anonid"][0]
        if pred_dir is not None:
            dose_pred_pct = np.load(Path(pred_dir) / f"{anonid}.npy")
        else:
            batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                pred = model(model._build_input(batch_gpu))
            dose_pred_pct = pred[0].cpu().numpy()

        dose_real_pct = batch["dose"][0].numpy()
        body = batch["body_mask"][0].numpy()
        ptv = batch["ptv_mask"][0].numpy()
        rectum = batch["rectum_mask"][0].numpy()
        bladder = batch["bladder_mask"][0].numpy()

        pred_pts = calcular_puntos_dvh_pred(dose_pred_pct * PRESCRIPCION_GY / 100.0,
                                            ptv, rectum, bladder)
        gt_row = gt_dvh.loc[anonid]
        fila = {
            "AnonID": anonid,
            "mae_body":    mae_en_mascara(dose_pred_pct, dose_real_pct, body),
            "mae_ptv":     mae_en_mascara(dose_pred_pct, dose_real_pct, ptv),
            "mae_rectum":  mae_en_mascara(dose_pred_pct, dose_real_pct, rectum),
            "mae_bladder": mae_en_mascara(dose_pred_pct, dose_real_pct, bladder),
            "dose_score_openkbp": dose_score_openkbp(dose_real_pct, dose_pred_pct, body),
        }
        for nombre, gt_col in GT_COL_MAP.items():
            fila[f"{nombre}_real"] = float(gt_row[gt_col])
            fila[f"{nombre}_pred"] = pred_pts[nombre]
        filas.append(fila)
    return pd.DataFrame(filas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/exp_hipo_004_finetune_ctfix_v4.yaml")
    p.add_argument("--checkpoint",
                   default=str(OLD_REPO / "repo/checkpoints/exp_hipo_004_finetune_ctfix_v4/epoch=014.ckpt"))
    p.add_argument("--gt-dvh-csv", default=str(OLD_REPO / "repo/data/gt_dvh_hipo_256_ctfix_v4.csv"))
    p.add_argument("--output-dir", default="results/exp_hipo_004_finetune_ctfix_v4_repro_nivel1")
    p.add_argument("--pred-dir", default=None, help="triage §9: dosis predichas guardadas (.npy)")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_experiment_config(str(_REPO_ROOT / args.config))
    cfg.data.cache_train = False
    cfg.data.cache_val = False
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.experiment.seed)

    model = device = None
    if args.pred_dir is None:
        model, device = cargar_modelo_strict(args.checkpoint, cfg)
        print(f"strict=True OK — dispositivo {device}")

    dm = DoseDataModule(cfg)
    dm.setup(stage="test")
    gt_dvh = pd.read_csv(args.gt_dvh_csv).set_index("AnonID")
    anonids = [Path(q).stem for q in dm.test_ds.npz_paths]
    faltan = sorted(set(anonids) - set(gt_dvh.index))
    assert not faltan, f"{len(faltan)} test sin GT: {faltan}"
    assert len(anonids) == 40, f"test n={len(anonids)} (esperado 40)"

    df = evaluar_dataset(model, device, dm.test_dataloader(), gt_dvh, "Test hipo", args.pred_dir)

    out = _REPO_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_patient_metrics_repro_nivel1.csv", index=False)

    rng = np.random.default_rng(args.bootstrap_seed)
    capa1 = {}
    for k in ("mae_body", "mae_ptv", "mae_rectum", "mae_bladder", "dose_score_openkbp"):
        v = df[k].to_numpy()
        capa1[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                    "ci95": bootstrap_ci(v, args.n_bootstrap, rng)}
    capa2 = {}
    for nombre in GT_COL_MAP:
        d = df[f"{nombre}_pred"].to_numpy() - df[f"{nombre}_real"].to_numpy()
        capa2[nombre] = {"bias": float(d.mean()), "mae": float(np.abs(d).mean())}

    summary = {"exp": "exp_hipo_004_finetune_ctfix_v4", "checkpoint": str(args.checkpoint),
               "modo": "pred_dir" if args.pred_dir else "inferencia", "n_test": len(df),
               "n_bootstrap": args.n_bootstrap, "bootstrap_seed": args.bootstrap_seed,
               "capa1_regresion_dosis": capa1, "capa2_puntos_dvh": capa2}
    with open(out / "metrics_repro_nivel1.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Nivel 1 vs. target ===")
    target = {"mae_body": 2.552, "mae_ptv": 1.804, "mae_rectum": 6.120, "mae_bladder": 3.976,
              "dose_score_openkbp": 2.552}
    for k, t in target.items():
        print(f"{k:20s} {capa1[k]['mean']:.4f}  (target {t})  d={capa1[k]['mean'] - t:+.4f}")
    t2 = {"PTV_D95Gy": 0.350, "PTV_V70Gy": 1.460, "Rectum_V65Gy": 2.100, "Bladder_V65Gy": 1.377}
    for k, t in t2.items():
        print(f"{k:20s} {capa2[k]['mae']:.4f}  (target {t})  d={capa2[k]['mae'] - t:+.4f}")
    print(f"MAE body IC95: {capa1['mae_body']['ci95']}  (target [2.05, 3.35])")


if __name__ == "__main__":
    main()
