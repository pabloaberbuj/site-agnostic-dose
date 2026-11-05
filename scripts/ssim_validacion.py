"""
Fase 3: validación del SSIM. NO entrena, NO toca test (solo val). Re-infiere cada modelo en val, des-padea a la grilla NATIVA
(frame.undo, independiente del contenedor) y calcula el SSIM con tres regiones:
  (a) frame completo  = lo que reporta eval_a0 (referencia; se verifica contra secondary_per_patient.csv)
  (b) BODY
  (c) BODY ∩ dosis real > 5 % de Rx_High (dose_pct ya está en % de Rx_High)
Mismo SSIM que torchmetrics (gauss 11, sigma 1.5, K1=.01, K2=.03, data_range=120, cortes con body, media por corte y luego entre cortes), pero con los mapas por píxel de luminancia/contraste/estructura para promediar sobre la máscara.
Las ventanas se calculan sobre la imagen completa; la máscara solo elige qué píxeles se promedian (reflect-pad de 5 px, sin recorte: igual que torchmetrics 1.9 / eval_a0).
SSIM = l·c·s con C3 = C2/2 (exacto). Los términos se promedian por separado (media de cada mapa; media(l·c·s) != producto de medias). `_ssimpix` = misma métrica agrupando píxeles (sensibilidad).

    .venv/Scripts/python.exe scripts/ssim_validacion.py infer      # genera results/ssim_validacion/per_patient_<modelo>.csv
    .venv/Scripts/python.exe scripts/ssim_validacion.py report     # tablas con IC95 bootstrap pareado
La lógica del SSIM vive en src/eval/ssim.py (la usa también eval_a0). OJO: esta tabla se generó con la pred CRUDA (sin el
post-proceso pred*BODY); es la vara histórica de Fase 3. El harness actual aplica pred*BODY (ver results/ssim_validacion/README.md).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.eval import dvh  # noqa: E402
import eval_a0  # noqa: E402

OUT = ROOT / 'results' / 'ssim_validacion'
CK = ROOT / 'checkpoints'
# nombre -> (experimento/config, ckpt, csv secundario de eval_a0 para verificar la variante (a))
MODELOS = {
    'A0':      ('exp_a0_001_unet2d', 'epoch=097.ckpt', 'results/exp_a0_001_unet2d/secondary_per_patient.csv'),
    'A1':      ('exp_a1a_001_unet2d_fill0', 'epoch=088.ckpt', 'results/exp_a1a_001_unet2d_fill0/fill_0secondary_per_patient.csv'),
    '2D':      ('exp_a4_003_unet2d_psdm_c176x256', 'epoch=113.ckpt', 'results/exp_a4_003_unet2d_psdm_c176x256/secondary_per_patient.csv'),
    '3D':      ('exp_b1_001_unet3d_psdm', 'epoch=076.ckpt', 'results/exp_b1_001_unet3d_psdm/secondary_per_patient.csv'),
    'MedNeXt': ('exp_b2_001_mednext2d_psdm', 'epoch=084.ckpt', 'results/exp_b2_001_mednext2d_psdm/ckpt_ep084/secondary_per_patient.csv'),
}
REGIONES = ('frame', 'body', 'body_dosis5')
from src.eval.ssim import region_stats  # noqa: E402


def infer(nombre):
    from omegaconf import OmegaConf
    from src.models.a0_module import A0Module
    exp, ck, _ = MODELOS[nombre]
    cfg = OmegaConf.load(ROOT / 'configs' / f'{exp}.yaml')
    dm, ds = eval_a0.get_ds('val', cfg.data)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = A0Module.load_from_checkpoint(str(CK / exp / ck), cfg=cfg, strict=True, map_location='cpu', weights_only=False).to(dev).eval()
    rows = []
    for i, pid in enumerate(ds.ids):
        dose, chans, meta = ds.load_raw(pid)
        Z, H, W = dose.shape
        x = torch.from_numpy(np.ascontiguousarray(ds.pad_channels(chans).transpose(1, 0, 2, 3)))[None].to(dev)
        with torch.no_grad():
            pred = model(x)[0].cpu().numpy()
        pred = ds.frame.undo(pred, ds.frame_box(chans), (H, W))
        body = eval_a0.masks_from_npz(ds.npz_index[pid])['BODY']
        rows.append({'modelo': nombre, 'patient_id': pid, 'site': meta['site'], **region_stats(pred, dose, body, dev)})
        if (i + 1) % 20 == 0:
            print(f'  {nombre} {i + 1}/{len(ds.ids)}', flush=True)
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def cmd_infer():
    OUT.mkdir(parents=True, exist_ok=True)
    for n in MODELOS:
        f = OUT / f'per_patient_{n}.csv'
        if f.exists():
            print(f'[{n}] ya calculado')
            continue
        print(f'[{n}] {MODELOS[n][0]} {MODELOS[n][1]}', flush=True)
        infer(n).to_csv(f, index=False)
    print('\nVerificación variante (a) vs ssim de eval_a0 (max |dif| por paciente):')
    for n, (_, _, sec) in MODELOS.items():
        d = pd.read_csv(OUT / f'per_patient_{n}.csv').merge(pd.read_csv(ROOT / sec), on='patient_id', suffixes=('', '_ref'))
        print(f'  {n:8s} n={len(d)}  max|dif|={np.abs(d.frame_ssim - d.ssim).max():.2e}  (media propia {d.frame_ssim.mean():.4f} / eval_a0 {d.ssim.mean():.4f})')


def cmd_report():
    df = pd.concat([pd.read_csv(OUT / f'per_patient_{n}.csv') for n in MODELOS])
    sites = ['CyC', 'Pelvis', 'Prostata']
    fmt = lambda r: f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]"
    print('== SSIM medio por sitio, modelo y región (IC95 bootstrap sobre pacientes) ==')
    for reg in REGIONES:
        print(f'\n-- región: {reg}')
        for s in sites:
            line = []
            for n in MODELOS:
                b = dvh.bootstrap_ci(df[(df.modelo == n) & (df.site == s)][f'{reg}_ssim'].values)
                line.append(f"{n} {b['mean']:.3f} [{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}] (n={b['n']})")
            print(f'  {s:9s} ' + ' | '.join(line))
    print('\n== Diferencias pareadas (nuevo - ref; >0 = nuevo mejor), IC95 bootstrap pareado por paciente ==')
    for nuevo, ref in (('3D', '2D'), ('MedNeXt', '2D'), ('A1', 'A0')):
        print(f'\n-- {nuevo} - {ref}')
        m = df[df.modelo == nuevo].merge(df[df.modelo == ref], on=['patient_id', 'site'], suffixes=('_n', '_r'))
        for s in sites:
            g = m[m.site == s]
            print(f'  {s:9s} ' + ' | '.join(f"{reg} {fmt(dvh.bootstrap_ci((g[f'{reg}_ssim_n'] - g[f'{reg}_ssim_r']).values))}" for reg in REGIONES) + f'  (n={len(g)})')
    print('\n== CyC: descomposición SSIM = luminancia x contraste x estructura (media de cada mapa) ==')
    for reg in REGIONES:
        print(f'\n-- región: {reg}')
        c = df[df.site == 'CyC']
        for n in MODELOS:
            g = c[c.modelo == n]
            print(f'  {n:8s} ' + ' | '.join(f"{t} {g[f'{reg}_{k}'].mean():.3f}" for t, k in (('SSIM', 'ssim'), ('lum', 'lum'), ('con', 'con'), ('est', 'est'))))
        for nuevo, ref in (('3D', '2D'), ('MedNeXt', '2D'), ('A1', 'A0')):
            m = c[c.modelo == nuevo].merge(c[c.modelo == ref], on='patient_id', suffixes=('_n', '_r'))
            print(f'  D {nuevo}-{ref}: ' + ' | '.join(
                f"{t} {fmt(dvh.bootstrap_ci((m[f'{reg}_{k}_n'] - m[f'{reg}_{k}_r']).values))}" for t, k in (('SSIM', 'ssim'), ('lum', 'lum'), ('con', 'con'), ('est', 'est'))))
    print('\n== Diagnóstico: error abs fuera del BODY (sin supervisión) y fracción de píxeles fuera del body ==')
    print(df.groupby(['site', 'modelo'])[['mae_fuera_body', 'frac_fuera_body']].mean().round(3).to_string())
    print('\n== Píxeles promediados por región (media por paciente, modelo 2D) ==')
    print(df[df.modelo == '2D'].groupby('site')[[f'{r}_n' for r in REGIONES]].mean().round(0).to_string())


if __name__ == '__main__':
    {'infer': cmd_infer, 'report': cmd_report}[sys.argv[1]]()
