"""
dvh_curva_completa (CHARTER §2, métrica principal) sobre las predicciones de Fase 1 Nivel 1.

Mide mean|ΔV(D)| entre DVH real y predicha, por estructura y por banda de dosis
(baja 0-40 / media 40-80 / alta 80-110 % Rx; la media es la crítica para OARs).
exp_hipo_004 nunca la corrió: esto ESTABLECE la vara para Fase 2, no reproduce un número previo.

Las funciones de curva/bandas/renormalización son las de scripts/dvh_curva_completa.py del
repo viejo (verbatim). Cambia solo el origen de los datos: usa el mismo DoseDataModule +
carga strict=True que evaluate_repro_nivel1.py (el viejo dependía de analisis_angular.py).
La dosis predicha se clipea a 0 y se renormaliza a D95(PTV_pred)=100% antes del DVH
(mide forma, no sesgo de escala global).

Uso:
    C:/Pablo/ProstateDoseProject/repo/.venv/Scripts/python.exe scripts/dvh_curva_completa.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from src.config.anatomy import load_experiment_config  # noqa: E402
from src.datamodules.dose_datamodule import DoseDataModule  # noqa: E402
from evaluate_repro_nivel1 import cargar_modelo_strict, OLD_REPO  # noqa: E402

D_BINS = np.arange(0, 111, 1)  # 0..110 %Rx, paso 1%
IDX_BAJA = slice(0, 41)
IDX_MEDIA = slice(41, 81)
IDX_ALTA = slice(81, 111)
D95_TOL_PP = 0.5

STRUCTURES = ["PTV", "Rectum", "Bladder", "BODY"]
MASK_KEY = {"PTV": "ptv_mask", "Rectum": "rectum_mask", "Bladder": "bladder_mask", "BODY": "body_mask"}


def curva_v_d(dose_pct, mask, d_bins=D_BINS):
    """V(D) = %volumen de la estructura con dosis >= D, para cada D en d_bins."""
    vals = dose_pct[mask > 0]
    if len(vals) == 0:
        return np.full(len(d_bins), np.nan)
    vals_sorted = np.sort(vals)
    n = len(vals_sorted)
    idx = np.searchsorted(vals_sorted, d_bins, side="left")
    return 100.0 * (n - idx) / n


def d95_ptv(dose_pct, ptv_mask):
    vals = dose_pct[ptv_mask > 0]
    if len(vals) == 0:
        return float("nan")
    return float(np.percentile(vals, 5))


def analizar_paciente(anonid, arrays, dose_pred_raw):
    dose_real = arrays["dose"]
    ptv = arrays["ptv_mask"]

    dose_pred_raw = np.clip(dose_pred_raw, 0.0, None)
    d95_pred_raw = d95_ptv(dose_pred_raw, ptv)
    factor_renorm = 100.0 / d95_pred_raw if d95_pred_raw > 0 else float("nan")
    dose_pred = dose_pred_raw * factor_renorm

    d95_real_check = d95_ptv(dose_real, ptv)
    d95_pred_check = d95_ptv(dose_pred, ptv)

    filas, curvas = [], {}
    for struct in STRUCTURES:
        mask = arrays[MASK_KEY[struct]]
        v_real = curva_v_d(dose_real, mask)
        v_pred = curva_v_d(dose_pred, mask)
        delta = v_pred - v_real
        tiene = mask.sum() > 0
        filas.append({
            "AnonID": anonid, "estructura": struct,
            "mean_abs_delta_V_global": float(np.mean(np.abs(delta))),
            "mean_abs_delta_V_baja_0_40": float(np.mean(np.abs(delta[IDX_BAJA]))),
            "mean_abs_delta_V_media_40_80": float(np.mean(np.abs(delta[IDX_MEDIA]))),
            "mean_abs_delta_V_alta_80_110": float(np.mean(np.abs(delta[IDX_ALTA]))),
            "n_voxeles": int(mask.sum()),
            "factor_renorm_pred": factor_renorm,
            "D95_PTV_real_pct": d95_real_check,
            "D95_PTV_pred_post_renorm_pct": d95_pred_check,
            "D95_PTV_pred_check_ok": bool(abs(d95_pred_check - 100.0) <= D95_TOL_PP),
            "V0_real_100_ok": bool(np.isclose(v_real[0], 100.0, atol=1e-6)) if tiene else None,
            "V0_pred_100_ok": bool(np.isclose(v_pred[0], 100.0, atol=1e-6)) if tiene else None,
            "monotona_real_ok": bool(np.all(np.diff(v_real) <= 1e-9)) if tiene else None,
            "monotona_pred_ok": bool(np.all(np.diff(v_pred) <= 1e-9)) if tiene else None,
        })
        curvas[struct] = {"v_real": v_real, "v_pred": v_pred, "delta": delta}
    return filas, curvas, factor_renorm, d95_real_check, d95_pred_check


def plot_dvh_medio_por_estructura(todas_curvas, plots_dir):
    for struct in STRUCTURES:
        v_real_all = np.array([c[struct]["v_real"] for c in todas_curvas.values()])
        v_pred_all = np.array([c[struct]["v_pred"] for c in todas_curvas.values()])
        delta_all = np.array([c[struct]["delta"] for c in todas_curvas.values()])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        rm, rs = v_real_all.mean(axis=0), v_real_all.std(axis=0)
        pm, ps = v_pred_all.mean(axis=0), v_pred_all.std(axis=0)
        ax.plot(D_BINS, rm, color="black", label="Real (media)")
        ax.fill_between(D_BINS, rm - rs, rm + rs, color="black", alpha=0.15)
        ax.plot(D_BINS, pm, color="firebrick", linestyle="--", label="Predicha renorm. (media)")
        ax.fill_between(D_BINS, pm - ps, pm + ps, color="firebrick", alpha=0.15)
        for lo, hi, c in ((0, 40, "gray"), (40, 80, "steelblue"), (80, 110, "darkorange")):
            ax.axvspan(lo, hi, color=c, alpha=0.08)
        ax.set_xlabel("Dosis (% Rx)")
        ax.set_ylabel("Volumen (%)")
        ax.set_title(f"DVH medio ± std (n={len(todas_curvas)}) — {struct}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = axes[1]
        dm, ds = delta_all.mean(axis=0), delta_all.std(axis=0)
        ax2.plot(D_BINS, dm, color="darkorange")
        ax2.fill_between(D_BINS, dm - ds, dm + ds, color="darkorange", alpha=0.2)
        ax2.axhline(0, color="gray", linewidth=0.8)
        for lo, hi, c, lab in ((0, 40, "gray", "baja"), (40, 80, "steelblue", "media"),
                               (80, 110, "darkorange", "alta")):
            ax2.axvspan(lo, hi, color=c, alpha=0.08, label=f"banda {lab}")
        ax2.set_xlabel("Dosis (% Rx)")
        ax2.set_ylabel("dV = V_pred - V_real (pp)")
        ax2.set_title(f"dV(D) medio ± std — {struct}")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(plots_dir / f"dvh_medio_{struct.lower()}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/exp_hipo_004_finetune_ctfix_v4.yaml")
    p.add_argument("--checkpoint",
                   default=str(OLD_REPO / "repo/checkpoints/exp_hipo_004_finetune_ctfix_v4/epoch=014.ckpt"))
    p.add_argument("--out-dir", default="results/exp_hipo_004_finetune_ctfix_v4_dvh_curva_completa")
    args = p.parse_args()

    out_dir = _REPO_ROOT / args.out_dir
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(str(_REPO_ROOT / args.config))
    cfg.data.cache_train = False
    cfg.data.cache_val = False
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.experiment.seed)

    model, device = cargar_modelo_strict(args.checkpoint, cfg)
    dm = DoseDataModule(cfg)
    dm.setup(stage="test")

    todas_filas, todas_curvas, fallas = [], {}, []
    for batch in tqdm(dm.test_dataloader(), desc="Test"):
        anonid = batch["anonid"][0]
        bg = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            dose_pred_raw = model(model._build_input(bg))[0].cpu().numpy()
        arrays = {"dose": batch["dose"][0].numpy()}
        for k in MASK_KEY.values():
            arrays[k] = batch[k][0].numpy()

        filas, curvas, *_ = analizar_paciente(anonid, arrays, dose_pred_raw)
        todas_filas.extend(filas)
        todas_curvas[anonid] = curvas
        for f in filas:
            malos = [k for k in ("D95_PTV_pred_check_ok", "V0_real_100_ok", "V0_pred_100_ok",
                                 "monotona_real_ok", "monotona_pred_ok") if f[k] is False]
            if malos:
                fallas.append({"AnonID": anonid, "estructura": f["estructura"], "fallas": malos})

    df = pd.DataFrame(todas_filas)
    df.to_csv(out_dir / "dvh_fidelity_por_paciente.csv", index=False)
    plot_dvh_medio_por_estructura(todas_curvas, plots_dir)

    resumen = {}
    for struct in STRUCTURES:
        sub = df[df["estructura"] == struct]
        r = {}
        for col, nombre in (("mean_abs_delta_V_global", "global"),
                            ("mean_abs_delta_V_baja_0_40", "baja_0_40"),
                            ("mean_abs_delta_V_media_40_80", "media_40_80"),
                            ("mean_abs_delta_V_alta_80_110", "alta_80_110")):
            r[nombre] = {"mean": float(sub[col].mean()), "std": float(sub[col].std())}
        r["banda_dominante"] = max(("baja_0_40", "media_40_80", "alta_80_110"), key=lambda b: r[b]["mean"])
        resumen[struct] = r

    summary = {
        "exp": "exp_hipo_004_finetune_ctfix_v4", "checkpoint": str(args.checkpoint),
        "n_test": len(todas_curvas),
        "nota": "vara NUEVA (no reproduce un numero previo) — referencia para Fase 2",
        "renormalizacion": "pred clip a 0 y renormalizada a D95(PTV_pred)=100% antes del DVH",
        "bandas_dosis_pct_rx": {"baja": [0, 40], "media": [40, 80], "alta": [80, 110]},
        "sanity_checks_fallidos": fallas,
        "mean_abs_delta_V_por_estructura": resumen,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSanity checks fallidos: {len(fallas)}")
    print("=== mean|dV| por estructura (pp de volumen) ===")
    for struct in STRUCTURES:
        r = resumen[struct]
        print(f"{struct:8s} global={r['global']['mean']:.2f}  baja={r['baja_0_40']['mean']:.2f}  "
              f"media={r['media_40_80']['mean']:.2f}  alta={r['alta_80_110']['mean']:.2f}  "
              f"(dominante: {r['banda_dominante']})")


if __name__ == "__main__":
    main()
