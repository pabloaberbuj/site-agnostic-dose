"""
Fase 0.0 — Envelope de memoria/tiempo en la RTX A2000 12GB.
Ver docs/SPEC_profiling_0.0.md para la metodología completa (no reescribir sin leerla).

Mide con tensores sintéticos (torch.randn) sobre las arquitecturas reales:
GroupNorm(8), bilinear/trilinear upsample, AMP 16-mixed, AdamW, step completo
(forward + backward + optimizer.step con GradScaler). No usa NPZ ni preprocesador
(Fase 0.1 aparte) — el consumo de VRAM depende de la forma del tensor, no del
contenido, así que 0.0 puede correr en paralelo a 0.1.

Uso:
    python scripts/profile_envelope.py                  # barrido completo (~10-20 min)
    python scripts/profile_envelope.py --sections 2A     # solo una seccion (2A,2B,2C)
    python scripts/profile_envelope.py --dry-run         # shapes chicas, smoke test rapido
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.unet2d import UNet2D
from src.models.unet3d import UNet3D
from src.models.mednext3d import MedNeXt3D

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "profiling" / "envelope.md"

HEADROOM_GB = 1.0    # §3: viable = sobrevive con >=1GB reserved libre sobre lo usable
N_WARMUP = 3          # §1.4: 2-3 steps (autotune de cudnn en el primero, sobre todo 3D)
N_MEASURE = 3
MAX_BATCH_RAMP = 64   # tope de seguridad para el ramp exponencial de batch

# Hallazgo real (2026-09-25, ver docs/insumos_fase0_profiling.md): en Windows el driver
# de NVIDIA puede hacer fallback silencioso a memoria de SISTEMA cuando se pide mas VRAM
# de la que hay, en vez de tirar torch.cuda.OutOfMemoryError. El sintoma es un step que
# "funciona" pero tarda 10-100x mas (ej: base_feat=32 en el volumen worst-case: 72s/step
# vs 580ms normal, reserved=17GB > los 12.88GB totales de la GPU). Sin este circuit
# breaker el ramp de batch sigue escalando indefinidamente sin detectar el borde real.
SOFT_TIMEOUT_S = 8.0  # ningun config legitimo observado supera ~800ms/step; 8s da margen
                       # generoso para MedNeXt (mas pesado) sin dejar pasar el thrashing

# cudnn.benchmark=True hace busqueda de algoritmo la PRIMERA vez que ve una forma nueva
# (batch/canales/spatial distintos) — esa busqueda evalua varios algoritmos candidatos
# (algunos, ej. FFT, con workspaces transitorios grandes) y puede tardar varios segundos
# sin que sea senal de thrashing real. Confirmado empiricamente: con SOFT_TIMEOUT_S
# aplicado tambien al primer step de warmup, configs ya validados como viables (ej.
# 2B-worst-bf16-ic5-off, medido limpio en 8.56GB/580ms en una corrida anterior) salieron
# como falso OOM. Por eso el primer step de warmup de cada probe usa un techo mas laxo —
# red de seguridad absoluta, no criterio de deteccion — y el criterio real (SOFT_TIMEOUT_S)
# se aplica recien desde el segundo step de warmup en adelante y en toda la medicion.
FIRST_STEP_TIMEOUT_S = 30.0

# Volúmenes 3D de referencia (ver insumos_fase0_profiling.md: grillas reales 134-161 Z,
# 89-115 Y, 145-251 X — padeadas a múltiplo de 16).
WORST_3D = (176, 128, 256)     # D, H, W — worst case global
TYPICAL_3D = (160, 128, 208)   # tamaño típico
SPATIAL_2D = (256, 256)        # H, W — inplane_size de exp002


@dataclass
class ProbeResult:
    status: str  # "OK" | "OOM" | "PENDING"
    peak_alloc_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    step_time_ms: float = 0.0
    note: str = ""  # "" | "cuda_oom_exception" | "no_headroom" | "timeout_thrash"


@dataclass
class Row:
    tag: str
    dim: str
    block: str
    spatial: tuple
    batch: int
    in_ch: int
    base_feat: int
    ckpt: bool
    depth: int = 4
    ramp: Optional[str] = None   # None | "batch"
    amp: bool = True
    cudnn_bench: bool = True
    result: ProbeResult = field(default_factory=lambda: ProbeResult("PENDING"))
    max_batch_found: Optional[int] = None  # solo si ramp == "batch"


def build_model(block: str, dim: str, in_ch: int, base_feat: int, depth: int,
                 grad_checkpointing: bool) -> torch.nn.Module:
    if dim == "2D-ctx":
        return UNet2D(in_channels=in_ch, out_channels=1, base_features=base_feat, depth=depth)
    if block == "unet":
        return UNet3D(in_channels=in_ch, out_channels=1, base_features=base_feat,
                       depth=depth, grad_checkpointing=grad_checkpointing)
    if block == "mednext-k5":
        return MedNeXt3D(in_channels=in_ch, out_channels=1, base_features=base_feat,
                          depth=depth, kernel_size=5, exp_ratio=4,
                          grad_checkpointing=grad_checkpointing)
    raise ValueError(f"block desconocido: {block}")


def run_probe(dim: str, block: str, spatial: tuple, batch: int, in_ch: int,
              base_feat: int, depth: int, ckpt: bool, device: torch.device,
              usable_gb: float, cudnn_bench: bool = True) -> ProbeResult:
    """Un step completo (fwd+bwd+optimizer.step) sobre tensores sinteticos. §1, §3.

    Dos mecanismos de deteccion de "no entra", ademas de la excepcion nativa:
    1. Circuit breaker por tiempo (SOFT_TIMEOUT_S): si un step individual tarda de mas,
       se aborta ya (evita pagar el costo completo de N_WARMUP+N_MEASURE steps en un
       config que esta haciendo thrashing).
    2. Reclasificacion por headroom: si el step SI corrio pero peak_reserved supera lo
       usable menos el headroom exigido, se marca OOM igual aunque no haya habido
       excepcion — cubre el fallback silencioso a memoria de sistema de Windows.
    """
    torch.backends.cudnn.benchmark = cudnn_bench
    model = opt = scaler = None
    try:
        model = build_model(block, dim, in_ch, base_feat, depth, ckpt).to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-5)
        scaler = torch.cuda.amp.GradScaler()

        in_shape = (batch, in_ch, *spatial)
        out_shape = (batch, 1, *spatial)

        def full_step():
            x = torch.randn(*in_shape, device=device)
            y = torch.randn(*out_shape, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                pred = model(x)
                loss = F.l1_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        thrashing = False
        for i in range(N_WARMUP):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            full_step()
            torch.cuda.synchronize()
            limit = FIRST_STEP_TIMEOUT_S if i == 0 else SOFT_TIMEOUT_S
            if time.perf_counter() - t0 > limit:
                thrashing = True
                break

        times = []
        if not thrashing:
            torch.cuda.reset_peak_memory_stats(device)  # §1.3: pico del step medido, no del warmup
            for _ in range(N_MEASURE):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                full_step()
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                times.append(dt * 1000)
                if dt > SOFT_TIMEOUT_S:
                    thrashing = True
                    break

        peak_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1e9
        step_time = sum(times) / len(times) if times else float("nan")

        if thrashing:
            return ProbeResult("OOM", peak_alloc, peak_reserved, step_time, "timeout_thrash")
        if peak_reserved > usable_gb - HEADROOM_GB:
            return ProbeResult("OOM", peak_alloc, peak_reserved, step_time, "no_headroom")
        return ProbeResult("OK", peak_alloc, peak_reserved, step_time)

    except torch.cuda.OutOfMemoryError:
        try:
            peak_alloc = torch.cuda.max_memory_allocated(device) / 1e9
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1e9
        except Exception:
            peak_alloc = peak_reserved = 0.0
        return ProbeResult("OOM", peak_alloc, peak_reserved, 0.0, "cuda_oom_exception")
    finally:
        del model, opt, scaler
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def ramp_batch(dim, block, spatial, in_ch, base_feat, depth, ckpt, device, usable_gb,
               start=1, max_batch=MAX_BATCH_RAMP):
    """(b) de §2: en un tamano espacial fijo, rampa batch hasta OOM.
    Doblado exponencial + busqueda binaria entre el ultimo OK y el primer OOM — O(log)."""
    last_ok_batch, last_ok_result = None, None
    batch = start
    while batch <= max_batch:
        r = run_probe(dim, block, spatial, batch, in_ch, base_feat, depth, ckpt, device, usable_gb)
        extra = f" reserved={r.peak_reserved_gb:.2f}GB step={r.step_time_ms:.0f}ms" if r.status == "OK" else f" ({r.note})"
        print(f"    probe batch={batch:<4d} -> {r.status}{extra}")
        if r.status == "OK":
            last_ok_batch, last_ok_result = batch, r
            batch *= 2
        else:
            lo, hi = (last_ok_batch or 0), batch
            while hi - lo > 1:
                mid = (lo + hi) // 2
                rm = run_probe(dim, block, spatial, mid, in_ch, base_feat, depth, ckpt, device, usable_gb)
                print(f"    probe batch={mid:<4d} -> {rm.status} (binary search)")
                if rm.status == "OK":
                    lo, last_ok_batch, last_ok_result = mid, mid, rm
                else:
                    hi = mid
            break
    return last_ok_batch, last_ok_result


def round16(x: float) -> int:
    return max(16, int(round(x / 16.0)) * 16)


def ramp_patch(block, aspect: str, in_ch, base_feat, depth, ckpt, device, usable_gb, max_unit=256):
    """§2D: rampa el tamano de un parche 3D (batch=1) hasta OOM.
    aspect='cubic' -> D=H=W. aspect='aniso' -> respeta la relacion natural
    D:H:W ~ 176:128:256 (grilla worst-case), normalizada por 128."""
    last_ok_shape, last_ok_result = None, None
    unit = 32
    while unit <= max_unit:
        if aspect == "cubic":
            shape = (round16(unit), round16(unit), round16(unit))
        else:
            shape = (round16(unit * 1.375), round16(unit), round16(unit * 2))
        r = run_probe("3D", block, shape, 1, in_ch, base_feat, depth, ckpt, device, usable_gb)
        extra = f" reserved={r.peak_reserved_gb:.2f}GB" if r.status == "OK" else f" ({r.note})"
        print(f"    probe {aspect} patch={shape} -> {r.status}{extra}")
        if r.status == "OK":
            last_ok_shape, last_ok_result = shape, r
            unit += 16
        else:
            break
    return last_ok_shape, last_ok_result


def measure_cuda_context_overhead(device) -> tuple:
    """Overhead real de driver/contexto CUDA (cuBLAS/cuDNN handles, etc.) y memoria
    libre real post-contexto. torch.cuda.memory_reserved() NO sirve para esto (solo
    mide el pool del allocator de PyTorch); mem_get_info() sí, porque consulta al
    driver directamente (equivalente a lo que reporta nvidia-smi) y además refleja
    el uso de otros procesos en la GPU, no solo el nuestro.
    Devuelve (context_overhead_gb, usable_gb_ahora)."""
    torch.cuda.empty_cache()
    free_before, total = torch.cuda.mem_get_info(device)
    # Forzar init de contexto + handles de cuBLAS/cuDNN (un matmul minimo los toca a ambos)
    x = torch.zeros(64, 64, device=device)
    y = x @ x
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info(device)
    del x, y
    torch.cuda.empty_cache()
    context_overhead_gb = (free_before - free_after) / 1e9
    usable_gb_now = free_after / 1e9
    return context_overhead_gb, usable_gb_now


# ────────────────────────────── Matriz de barrido (§2) ──────────────────────────────

def build_matrix(dry_run: bool) -> list:
    """Devuelve la lista de Row a correr. dry_run usa shapes chicas para smoke test."""
    if dry_run:
        spatial_2d, worst_3d, typical_3d = (32, 32), (32, 32, 32), (32, 32, 32)
    else:
        spatial_2d, worst_3d, typical_3d = SPATIAL_2D, WORST_3D, TYPICAL_3D

    rows = [
        # 2A — 2D con contexto axial (ancla Fase 1-2)
        Row("2A-A0", "2D-ctx", "unet", spatial_2d, 1, 5, 16, False, ramp="batch"),
        Row("2A-A4", "2D-ctx", "unet", spatial_2d, 1, 11, 16, False, ramp="batch"),

        # 2B — 3D volumen entero, U-Net vanilla
        Row("2B-worst-bf16-ic5-off", "3D", "unet", worst_3d, 1, 5, 16, False),
        Row("2B-worst-bf16-ic5-on", "3D", "unet", worst_3d, 1, 5, 16, True),
        Row("2B-worst-bf16-ic11-off", "3D", "unet", worst_3d, 1, 11, 16, False),
        Row("2B-worst-bf16-ic11-on", "3D", "unet", worst_3d, 1, 11, 16, True),
        Row("2B-worst-bf32-ic5-off", "3D", "unet", worst_3d, 1, 5, 32, False),
        Row("2B-worst-bf32-ic5-on", "3D", "unet", worst_3d, 1, 5, 32, True),
        Row("2B-typical-bf16-ic5-off", "3D", "unet", typical_3d, 1, 5, 16, False, ramp="batch"),

        # 2C — 3D MedNeXt kernel-5 (el limite real de Fase 3)
        Row("2C-worst-bf16-ic5-off", "3D", "mednext-k5", worst_3d, 1, 5, 16, False),
        Row("2C-worst-bf16-ic5-on", "3D", "mednext-k5", worst_3d, 1, 5, 16, True),
    ]
    return rows


def section_of(tag: str) -> str:
    return tag.split("-")[0]


def run_matrix(rows: list, sections: Optional[list], device, usable_gb: float) -> tuple:
    """Corre cada Row (con ramp de batch si aplica). Devuelve (rows corridas, patch_rows)."""
    ran = []
    for row in rows:
        if sections and section_of(row.tag) not in sections:
            continue
        print(f"\n[{row.tag}] dim={row.dim} block={row.block} spatial={row.spatial} "
              f"in_ch={row.in_ch} base_feat={row.base_feat} ckpt={row.ckpt}"
              + (" (ramp batch)" if row.ramp == "batch" else f" batch={row.batch}"))
        if row.ramp == "batch":
            max_b, res = ramp_batch(row.dim, row.block, row.spatial, row.in_ch,
                                     row.base_feat, row.depth, row.ckpt, device, usable_gb)
            row.max_batch_found = max_b
            row.result = res if res is not None else ProbeResult("OOM")
            row.batch = max_b if max_b is not None else 1
        else:
            row.result = run_probe(row.dim, row.block, row.spatial, row.batch, row.in_ch,
                                    row.base_feat, row.depth, row.ckpt, device, usable_gb)
            extra = (f" reserved={row.result.peak_reserved_gb:.2f}GB step={row.result.step_time_ms:.0f}ms"
                     if row.result.status == "OK" else f" ({row.result.note})")
            print(f"    -> {row.result.status}{extra}")
        ran.append(row)

    # §2D — solo si algo de 2B/2C OOMea a volumen entero con ckpt on (el caso mas favorable)
    patch_rows = []
    by_tag = {r.tag: r for r in ran}
    vanilla_anchor = by_tag.get("2B-worst-bf16-ic5-on")
    mednext_anchor = by_tag.get("2C-worst-bf16-ic5-on")

    need_patch_vanilla = vanilla_anchor is not None and vanilla_anchor.result.status == "OOM"
    need_patch_mednext = mednext_anchor is not None and mednext_anchor.result.status == "OOM"

    if need_patch_vanilla or need_patch_mednext:
        print("\n[2D] Volumen entero OOMeo con ckpt on -> corriendo busqueda de parches.")
    for need, block, label in [(need_patch_vanilla, "unet", "unet"),
                                (need_patch_mednext, "mednext-k5", "mednext-k5")]:
        if not need:
            continue
        for aspect in ("cubic", "aniso"):
            print(f"\n[2D-{label}-{aspect}] rampa de parche, batch=1, ckpt=on")
            shape, res = ramp_patch(block, aspect, 5, 16, 4, True, device, usable_gb)
            patch_rows.append({
                "tag": f"2D-{label}-{aspect}", "block": block, "aspect": aspect,
                "shape": shape, "result": res,
            })
    return ran, patch_rows


# ────────────────────────────── Reporte (§5) ──────────────────────────────

def fmt_spatial(spatial) -> str:
    return "×".join(str(s) for s in spatial)


def write_envelope_md(rows: list, patch_rows: list, context_overhead_gb: float,
                       usable_gb: float, total_mem_gb: float, out_path: Path):
    by_tag = {r.tag: r for r in rows}

    lines = []
    lines.append("# Envelope de profiling — Fase 0.0 (RTX A2000 12GB)\n")
    lines.append(f"> Generado por `scripts/profile_envelope.py`. Ver `docs/SPEC_profiling_0.0.md`.\n")
    lines.append(f"- VRAM total GPU: **{total_mem_gb:.2f} GB**")
    lines.append(f"- Overhead de contexto CUDA (driver + cuBLAS/cuDNN, medido con mem_get_info): **{context_overhead_gb:.2f} GB**")
    lines.append(f"- VRAM usable real en el momento de correr (incluye uso de otros procesos, si hubiera): **{usable_gb:.2f} GB**")
    lines.append(f"- Headroom exigido para \"viable\": **{HEADROOM_GB:.1f} GB**\n")

    # Tabla cruda
    lines.append("## Tabla cruda\n")
    header = ("| tag | dim | block | spatial | batch | in_ch | base_feat | ckpt | amp | "
               "cudnn_bench | peak_reserved_GB | peak_alloc_GB | step_time_ms | status | headroom_GB |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        res = r.result
        headroom = usable_gb - res.peak_reserved_gb
        batch_str = f"{r.batch}" + (" (max)" if r.ramp == "batch" else "")
        status_str = res.status if not res.note else f"{res.status} ({res.note})"
        reserved_str = f"{res.peak_reserved_gb:.2f}" if res.peak_reserved_gb else "-"
        alloc_str = f"{res.peak_alloc_gb:.2f}" if res.peak_alloc_gb else "-"
        step_str = f"{res.step_time_ms:.0f}" if res.step_time_ms == res.step_time_ms else "-"  # NaN check
        headroom_str = f"{headroom:.2f}" if res.peak_reserved_gb else "-"
        lines.append(
            f"| {r.tag} | {r.dim} | {r.block} | {fmt_spatial(r.spatial)} | {batch_str} | "
            f"{r.in_ch} | {r.base_feat} | {'on' if r.ckpt else 'off'} | {'16-mixed' if r.amp else 'off'} | "
            f"{'on' if r.cudnn_bench else 'off'} | "
            f"{reserved_str} | {alloc_str} | {step_str} | {status_str} | {headroom_str} |"
        )
    if patch_rows:
        lines.append("")
        lines.append("### Parches 3D (§2D — corrido solo porque volumen entero OOMeo)\n")
        lines.append("| tag | block | aspect | shape_max_ok | peak_reserved_GB | status |")
        lines.append("|---|---|---|---|---|---|")
        for pr in patch_rows:
            if pr["shape"] is not None:
                lines.append(f"| {pr['tag']} | {pr['block']} | {pr['aspect']} | "
                              f"{fmt_spatial(pr['shape'])} | {pr['result'].peak_reserved_gb:.2f} | OK |")
            else:
                lines.append(f"| {pr['tag']} | {pr['block']} | {pr['aspect']} | - | - | OOM incluso en el mínimo |")

    # Sección A
    lines.append("\n## Sección A — Configs operativos recomendados\n")
    a0 = by_tag.get("2A-A0")
    a4 = by_tag.get("2A-A4")
    lines.append("**Fase 1-2 (2D-ctx):**")
    if a0 and a0.result.status == "OK":
        lines.append(f"- A0-like (in_ch=5): batch máximo viable = **{a0.batch}** "
                      f"(reserved={a0.result.peak_reserved_gb:.2f}GB, step={a0.result.step_time_ms:.0f}ms)")
    if a4 and a4.result.status == "OK":
        lines.append(f"- A4-like (in_ch=11): batch máximo viable = **{a4.batch}** "
                      f"(reserved={a4.result.peak_reserved_gb:.2f}GB, step={a4.result.step_time_ms:.0f}ms)")
    lines.append(f"- Recomendado: usar el **mínimo entre A0 y A4** como batch de referencia "
                 f"(constante a través de las ablaciones de A-eje, para que la regla de escala "
                 f"por pasos de gradiente del CHARTER §5 sea comparable).")

    lines.append("\n**Fase 3 vanilla-3D:**")
    v_off = by_tag.get("2B-worst-bf16-ic5-off")
    v_on = by_tag.get("2B-worst-bf16-ic5-on")
    if v_off:
        entra = "SÍ" if v_off.result.status == "OK" else "NO"
        lines.append(f"- Volumen entero ({fmt_spatial(v_off.spatial)}) sin ckpt: **{entra}** entra "
                     f"(base_feat=16, in_ch=5) — status={v_off.result.status}"
                     + (f", reserved={v_off.result.peak_reserved_gb:.2f}GB" if v_off.result.status == "OK" else ""))
    if v_on:
        entra = "SÍ" if v_on.result.status == "OK" else "NO"
        lines.append(f"- Volumen entero con ckpt on: **{entra}** entra — status={v_on.result.status}"
                     + (f", reserved={v_on.result.peak_reserved_gb:.2f}GB, step={v_on.result.step_time_ms:.0f}ms" if v_on.result.status == "OK" else ""))
    bf32_off = by_tag.get("2B-worst-bf32-ic5-off")
    bf32_on = by_tag.get("2B-worst-bf32-ic5-on")
    for tag_row, label in [(bf32_off, "base_feat=32 sin ckpt"), (bf32_on, "base_feat=32 con ckpt")]:
        if tag_row:
            entra = "SÍ" if tag_row.result.status == "OK" else "NO"
            lines.append(f"- {label}: **{entra}** entra — status={tag_row.result.status}")

    lines.append("\n**Fase 3 MedNeXt-k5:**")
    m_off = by_tag.get("2C-worst-bf16-ic5-off")
    m_on = by_tag.get("2C-worst-bf16-ic5-on")
    if m_off:
        entra = "SÍ" if m_off.result.status == "OK" else "NO"
        lines.append(f"- Volumen entero sin ckpt: **{entra}** entra — status={m_off.result.status}"
                     + (f", reserved={m_off.result.peak_reserved_gb:.2f}GB" if m_off.result.status == "OK" else ""))
    if m_on:
        entra = "SÍ" if m_on.result.status == "OK" else "NO"
        lines.append(f"- Volumen entero con ckpt on: **{entra}** entra — status={m_on.result.status}"
                     + (f", reserved={m_on.result.peak_reserved_gb:.2f}GB, step={m_on.result.step_time_ms:.0f}ms" if m_on.result.status == "OK" else ""))
    if patch_rows:
        lines.append("- Parche recomendado (ver tabla de parches arriba).")

    # Sección B
    lines.append("\n## Sección B — Decisiones que el envelope resuelve\n")

    downsample_needed = False
    if v_off and v_off.result.status == "OOM" and not patch_rows:
        downsample_needed = True
    lines.append(f"1. **¿Downsampleo de la grilla de dosis?** "
                 f"{'NO' if not downsample_needed else 'A EVALUAR'} — "
                 f"{'las grillas entran enteras (o con parches) sin necesidad de downsamplear; preferible parchear antes que perder gradiente de dosis cerca del PTV.' if not downsample_needed else 'ni volumen entero ni parches entraron — revisar antes de decidir downsampleo.'}")

    vanilla_needs_patch = need_patch_vanilla_final = (v_on is not None and v_on.result.status == "OOM")
    mednext_needs_patch = (m_on is not None and m_on.result.status == "OOM")
    if not vanilla_needs_patch and not mednext_needs_patch:
        parches_txt = "NO, en ninguno — el volumen entero (worst-case) entra en ambos con ckpt on."
    elif vanilla_needs_patch and mednext_needs_patch:
        parches_txt = "SÍ, en ambos (vanilla y MedNeXt-k5)."
    elif mednext_needs_patch:
        parches_txt = "Solo en MedNeXt-k5 — vanilla entra entero."
    else:
        parches_txt = "Solo en vanilla — revisar, es inusual que vanilla OOMee y MedNeXt no."
    lines.append(f"2. **¿Parches obligatorios en 3D?** {parches_txt}")

    if patch_rows:
        aniso_rows = [p for p in patch_rows if p["aspect"] == "aniso" and p["shape"]]
        cubic_rows = [p for p in patch_rows if p["aspect"] == "cubic" and p["shape"]]
        best_aniso = max(aniso_rows, key=lambda p: p["shape"][0] * p["shape"][1] * p["shape"][2], default=None)
        best_cubic = max(cubic_rows, key=lambda p: p["shape"][0] * p["shape"][1] * p["shape"][2], default=None)
        txt = "Comparar volumen de parche cúbico vs anisotrópico máximo encontrado: "
        if best_aniso:
            txt += f"aniso={fmt_spatial(best_aniso['shape'])} "
        if best_cubic:
            txt += f"cubic={fmt_spatial(best_cubic['shape'])}. "
        txt += "Preferir anisotrópico (más in-plane/Z generoso) salvo que el cúbico gane volumen total por buen margen."
        lines.append(f"3. **Forma y tamaño máximo de parche:** {txt}")
    else:
        lines.append("3. **Forma y tamaño máximo de parche:** N/A — no hizo falta parchear.")

    ckpt_lines = []
    for off_tag, on_tag, label in [("2B-worst-bf16-ic5-off", "2B-worst-bf16-ic5-on", "vanilla worst-case bf16"),
                                     ("2C-worst-bf16-ic5-off", "2C-worst-bf16-ic5-on", "mednext-k5 worst-case")]:
        off, on = by_tag.get(off_tag), by_tag.get(on_tag)
        if off and on and off.result.status == "OK" and on.result.status == "OK":
            delta_mem = off.result.peak_reserved_gb - on.result.peak_reserved_gb
            delta_time_pct = (on.result.step_time_ms / off.result.step_time_ms - 1) * 100 if off.result.step_time_ms else float("nan")
            ckpt_lines.append(f"{label}: ahorra {delta_mem:.2f}GB reserved a costo de +{delta_time_pct:.0f}% tiempo/step")
        elif off and off.result.status == "OOM" and on and on.result.status == "OK":
            ckpt_lines.append(f"{label}: NECESARIO — sin ckpt OOMea, con ckpt entra")
        elif off and on and off.result.status == "OOM" and on.result.status == "OOM":
            ckpt_lines.append(f"{label}: insuficiente incluso con ckpt on")
    if ckpt_lines:
        lines.append(f"4. **¿Gradient checkpointing necesario?** " + "; ".join(ckpt_lines) + ".")
    else:
        lines.append("4. **¿Gradient checkpointing necesario?** Sin datos suficientes (revisar tabla cruda).")

    typical_row = by_tag.get('2B-typical-bf16-ic5-off')
    lines.append(f"5. **Batch máximo viable por dim → regla de escala CHARTER §5:** "
                 f"2D-ctx A0={a0.batch if a0 else '?'}, A4={a4.batch if a4 else '?'}"
                 + (f"; 3D típico ({fmt_spatial(typical_row.spatial)})={typical_row.batch}"
                    if typical_row else "") + ".")

    bf_range = []
    if v_off and v_off.result.status == "OK":
        bf_range.append("16 (OK)")
    if bf32_off and bf32_off.result.status == "OK":
        bf_range.append("32 (OK)")
    elif bf32_off:
        bf_range.append("32 (OOM)")
    lines.append(f"6. **Rango de base_feat legítimo para B-eje:** {', '.join(bf_range) if bf_range else 'sin datos'}.")

    lines.append("\n---\n")
    lines.append("*Nota metodológica:* arquitectura MedNeXt-k5 es un stand-in de profiling "
                  "(`src/models/mednext3d.py`), no la implementación final de Fase 3 (esa se define "
                  "con el código público de Xiong, CHARTER §4). El orden de magnitud de memoria es "
                  "representativo; los números exactos pueden variar con la implementación final.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Envelope escrito en: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", type=str, default=None,
                         help="Coma-separado: 2A,2B,2C. Default: todas.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Shapes chicas (32^2 / 32^3) para smoke test rapido, sin valor de profiling real.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no hay GPU CUDA disponible. Este profiling no tiene sentido en CPU.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_mem_gb = props.total_memory / 1e9
    print(f"GPU: {props.name} — {total_mem_gb:.2f} GB totales")

    context_overhead, usable_gb_now = measure_cuda_context_overhead(device)
    print(f"Overhead de contexto CUDA: {context_overhead:.2f} GB — usable ahora: {usable_gb_now:.2f} GB")

    sections = [s.strip() for s in args.sections.split(",")] if args.sections else None
    rows = build_matrix(dry_run=args.dry_run)
    ran, patch_rows = run_matrix(rows, sections, device, usable_gb_now)

    write_envelope_md(ran, patch_rows, context_overhead, usable_gb_now, total_mem_gb, Path(args.out))


if __name__ == "__main__":
    main()
