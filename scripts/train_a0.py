"""
Entrena A0 (Fase 2, HO#2). Un solo modelo pooled sobre los 3 sitios.

    .venv/Scripts/python.exe scripts/train_a0.py --config configs/exp_a0_001_unet2d.yaml [--fast-dev-run]

No correr en background con `&` (doble lanzamiento contiende por GPU/checkpoint dir, ver Fase 1).
No evalúa test: test queda reservado para el cierre del eje A (el eval usa val).
"""
import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datamodules.a0_datamodule import A0DataModule  # noqa: E402
from src.models.a0_module import A0Module  # noqa: E402


def build_dm(cfg):
    d = cfg.data
    if d.get('datamodule', 'a0') == 'a5':   # A5: A4 + Ch7 densidad másica (sidecar density_dir)
        from src.datamodules.a5_datamodule import A5DataModule
        return A5DataModule(ROOT / d.npz_dir, ROOT / d.splits_file, ROOT / d.harmonization_dir,
                            batch_size=d.batch_size, num_workers=d.num_workers, pin_memory=d.pin_memory,
                            pad_hw=d.pad_hw, fill_sin_goal=d.fill_sin_goal, goal_mode=d.goal_mode,
                            ptv_ref=d.get('ptv_ref', 'union'), density_dir=ROOT / d.density_dir)
    if d.get('datamodule', 'a0') in ('a1', 'a4'):   # A1: + 3 canales de goals (fill_sin_goal / goal_mode / ptv_ref)
        from src.datamodules.a1_datamodule import A1DataModule
        if d.datamodule == 'a4':   # A4: A1 + Ch6 PSDM del PTV_High
            from src.datamodules.a4_datamodule import A4DataModule as A1DataModule
        return A1DataModule(ROOT / d.npz_dir, ROOT / d.splits_file, ROOT / d.harmonization_dir,
                            batch_size=d.batch_size, num_workers=d.num_workers, pin_memory=d.pin_memory,
                            pad_hw=d.pad_hw, fill_sin_goal=d.fill_sin_goal, goal_mode=d.goal_mode,
                            ptv_ref=d.get('ptv_ref', 'union'))
    if d.get('datamodule', 'a0') == 'a4solo':   # HO#9: A0 + Ch3 PSDM(PTV_High), sin goals
        from src.datamodules.a4_datamodule import A4SoloDataModule
        return A4SoloDataModule(ROOT / d.npz_dir, ROOT / d.splits_file, ROOT / d.harmonization_dir,
                                batch_size=d.batch_size, num_workers=d.num_workers,
                                pin_memory=d.pin_memory, pad_hw=d.pad_hw)
    return A0DataModule(ROOT / d.npz_dir, ROOT / d.splits_file, ROOT / d.harmonization_dir,
                        batch_size=d.batch_size, num_workers=d.num_workers,
                        pin_memory=d.pin_memory, pad_hw=d.pad_hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--fast-dev-run', action='store_true')
    ap.add_argument('--max-epochs', type=int, default=None,
                    help='corte temprano (smoke); scheduler.max_epochs NO cambia, el LR sigue la curva del horizonte completo')
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    assert cfg.scheduler.max_epochs == cfg.training.max_epochs, 'horizonte del scheduler != trainer'
    es = cfg.epoch_scale
    assert round(es.ref_max_epochs * es.n_train_ref / es.n_train) == cfg.scheduler.max_epochs, 'factor/horizonte inconsistentes'

    torch.set_float32_matmul_precision('high')
    pl.seed_everything(cfg.experiment.seed, workers=True)

    dm = build_dm(cfg)
    dm.setup()
    assert len(dm.train_ds) == es.n_train, f'N_train real {len(dm.train_ds)} != {es.n_train} del factor'
    model = A0Module(cfg)

    name = cfg.experiment.name
    ckpt_dir = ROOT / 'checkpoints' / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    record = {'exp_id': name, **dm.experiment_record(), 'epoch_scale': OmegaConf.to_container(es),
              'cfg': OmegaConf.to_container(cfg, resolve=True)}
    res_dir = ROOT / 'results' / name
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / 'record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f"[record] hash tabla={record['harmonization_hash']} npz={record['npz_dir']} "
          f"split={record['splits_file']} pad={record['pad_hw']} factor={es.factor}")

    monitor = cfg.training.monitor_metric
    callbacks = [
        ModelCheckpoint(dirpath=str(ckpt_dir), filename='{epoch:03d}', monitor=monitor,
                        mode=cfg.training.monitor_mode, save_top_k=3, save_last=True),
        EarlyStopping(monitor=monitor, mode=cfg.training.monitor_mode,
                      patience=cfg.training.early_stopping_patience, verbose=True),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs or cfg.training.max_epochs,
        precision=cfg.training.precision, gradient_clip_val=cfg.training.gradient_clip_val,
        logger=CSVLogger(str(ROOT / 'lightning_logs'), name=name), callbacks=callbacks,
        deterministic=False, fast_dev_run=args.fast_dev_run, log_every_n_steps=10)
    trainer.fit(model, datamodule=dm)
    print('best:', trainer.checkpoint_callback.best_model_path, trainer.checkpoint_callback.best_model_score)


if __name__ == '__main__':
    main()
