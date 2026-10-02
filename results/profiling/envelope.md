# Envelope de profiling — Fase 0.0 (RTX A2000 12GB)

> Generado por `scripts/profile_envelope.py`. Ver `docs/SPEC_profiling_0.0.md`.

- VRAM total GPU: **12.88 GB**
- Overhead de contexto CUDA (driver + cuBLAS/cuDNN, medido con mem_get_info): **0.05 GB**
- VRAM usable real en el momento de correr (incluye uso de otros procesos, si hubiera): **11.75 GB**
- Headroom exigido para "viable": **1.0 GB**

## Tabla cruda

| tag | dim | block | spatial | batch | in_ch | base_feat | ckpt | amp | cudnn_bench | peak_reserved_GB | peak_alloc_GB | step_time_ms | status | headroom_GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2A-A0 | 2D-ctx | unet | 256×256 | 64 (max) | 5 | 16 | off | 16-mixed | on | 8.52 | 7.56 | 440 | OK | 3.23 |
| 2A-A4 | 2D-ctx | unet | 256×256 | 64 (max) | 11 | 16 | off | 16-mixed | on | 8.67 | 7.72 | 450 | OK | 3.08 |
| 2B-worst-bf16-ic5-off | 3D | unet | 176×128×256 | 1 | 5 | 16 | off | 16-mixed | on | 8.56 | 7.53 | 581 | OK | 3.19 |
| 2B-worst-bf16-ic5-on | 3D | unet | 176×128×256 | 1 | 5 | 16 | on | 16-mixed | on | 5.95 | 4.71 | 800 | OK | 5.80 |
| 2B-worst-bf16-ic11-off | 3D | unet | 176×128×256 | 1 | 11 | 16 | off | 16-mixed | on | 8.77 | 7.74 | 599 | OK | 2.98 |
| 2B-worst-bf16-ic11-on | 3D | unet | 176×128×256 | 1 | 11 | 16 | on | 16-mixed | on | 6.14 | 4.85 | 819 | OK | 5.61 |
| 2B-worst-bf32-ic5-off | 3D | unet | 176×128×256 | 1 | 5 | 32 | off | 16-mixed | on | 15.34 | 14.83 | - | OOM (timeout_thrash) | -3.59 |
| 2B-worst-bf32-ic5-on | 3D | unet | 176×128×256 | 1 | 5 | 32 | on | 16-mixed | on | 12.14 | 9.39 | 2334 | OOM (no_headroom) | -0.39 |
| 2B-typical-bf16-ic5-off | 3D | unet | 160×128×208 | 1 (max) | 5 | 16 | off | 16-mixed | on | 6.38 | 5.60 | 445 | OK | 5.37 |
| 2C-worst-bf16-ic5-off | 3D | mednext-k5 | 176×128×256 | 1 | 5 | 16 | off | 16-mixed | on | 12.01 | 11.22 | 1880 | OOM (no_headroom) | -0.26 |
| 2C-worst-bf16-ic5-on | 3D | mednext-k5 | 176×128×256 | 1 | 5 | 16 | on | 16-mixed | on | 9.47 | 7.18 | 2327 | OK | 2.28 |

## Sección A — Configs operativos recomendados

**Fase 1-2 (2D-ctx):**
- A0-like (in_ch=5): batch máximo viable = **64** (reserved=8.52GB, step=440ms)
- A4-like (in_ch=11): batch máximo viable = **64** (reserved=8.67GB, step=450ms)
- Recomendado: usar el **mínimo entre A0 y A4** como batch de referencia (constante a través de las ablaciones de A-eje, para que la regla de escala por pasos de gradiente del CHARTER §5 sea comparable).

**Fase 3 vanilla-3D:**
- Volumen entero (176×128×256) sin ckpt: **SÍ** entra (base_feat=16, in_ch=5) — status=OK, reserved=8.56GB
- Volumen entero con ckpt on: **SÍ** entra — status=OK, reserved=5.95GB, step=800ms
- base_feat=32 sin ckpt: **NO** entra — status=OOM
- base_feat=32 con ckpt: **NO** entra — status=OOM

**Fase 3 MedNeXt-k5:**
- Volumen entero sin ckpt: **NO** entra — status=OOM
- Volumen entero con ckpt on: **SÍ** entra — status=OK, reserved=9.47GB, step=2327ms

## Sección B — Decisiones que el envelope resuelve

1. **¿Downsampleo de la grilla de dosis?** NO — las grillas entran enteras (o con parches) sin necesidad de downsamplear; preferible parchear antes que perder gradiente de dosis cerca del PTV.
2. **¿Parches obligatorios en 3D?** NO, en ninguno — el volumen entero (worst-case) entra en ambos con ckpt on.
3. **Forma y tamaño máximo de parche:** N/A — no hizo falta parchear.
4. **¿Gradient checkpointing necesario?** vanilla worst-case bf16: ahorra 2.61GB reserved a costo de +38% tiempo/step; mednext-k5 worst-case: NECESARIO — sin ckpt OOMea, con ckpt entra.
5. **Batch máximo viable por dim → regla de escala CHARTER §5:** 2D-ctx A0=64, A4=64; 3D típico (160×128×208)=1.
6. **Rango de base_feat legítimo para B-eje:** 16 (OK), 32 (OOM).

---

*Nota metodológica:* arquitectura MedNeXt-k5 es un stand-in de profiling (`src/models/mednext3d.py`), no la implementación final de Fase 3 (esa se define con el código público de Xiong, CHARTER §4). El orden de magnitud de memoria es representativo; los números exactos pueden variar con la implementación final.