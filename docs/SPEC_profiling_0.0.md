# Spec — Profiling 0.0 (envelope de la RTX A2000 12GB)

> Para ejecutar desde Claude Code. Producto: `results/profiling/envelope.md`.
> No entrena nada real: mide memoria/tiempo sobre **tensores sintéticos** con las formas
> reales. Se apoya en el `profile_step.py` de próstata (adaptado, no reescrito) y en los
> insumos de DICOM/exp002 ya relevados.
>
> Handoff de vuelta al chat: solo el delta (`envelope.md` + una línea de conclusión por
> decisión). No pegar el doc entero.

---

## 0. Por qué sintético y por qué ahora

El consumo de VRAM depende de la **forma** de los tensores, no de su contenido → se puede
medir con `torch.randn` de las dimensiones reales, sin NPZ preprocesados. Esto desacopla 0.0
del preprocesador (0.1) y permite correr el barrido en paralelo mientras se diseña la tabla de
armonización.

Insumo ya confirmado (no re-relevar): spacing de grilla de dosis uniforme en los 6 pacientes
de muestra (2.5×2.5 in-plane, 3.0 mm Z); grillas ya recortadas por Eclipse al PTV+margen
(máx. 161×115×251 vóxeles, típicas ~150×110×200). GroupNorm(8), AdamW, AMP 16-mixed
confirmados en exp002.

---

## 1. Adaptación del `profile_step.py` viejo

Reusar **la construcción de modelo desde config** (vía `DosePredictionModule`) tal cual — es
lo que lo hace comparable con exp002. Cambiar solo:

1. **Fuente de datos → sintética.** Reemplazar la carga de NPZ por un generador
   `torch.randn(batch, in_ch, *spatial)` en la forma pedida por el config (2D-context: 
   `(B, C, H, W)`; 3D: `(B, C, D, H, W)`). Target sintético del mismo modo para el loss.
   No medir I/O en esta fase (no hay pipeline real todavía).
2. **Agregar `max_memory_reserved`.** El script viejo reporta `memory_allocated` /
   `max_memory_allocated`. El OOM real es contra **reserved + fragmentación**. Agregar
   `torch.cuda.max_memory_reserved()` al reporte; es la columna que manda para "viable".
3. **`reset_peak_memory_stats()` antes de cada ventana medida**, para que el pico sea del
   step medido y no arrastre autotune previo.
4. **Warmup 2–3 steps** (el viejo hace 1). En 3D el autotune de cudnn del primer step reserva
   workspace que ensucia la medición.
5. **Envolver en try/except OOM** para que un config que revienta se registre como
   `status=OOM` y el barrido siga, en vez de matar la corrida.
6. **Wrapper de barrido:** el viejo corre 1 config por invocación. Envolverlo en un loop que
   itere la matriz de §2 y escriba una fila por config. Entre configs, liberar
   (`del model; torch.cuda.empty_cache()`).

Mantener idéntico: step completo (fwd + bwd + `optimizer.step` con GradScaler), modelo real
vía config, medición repetida ×3 tras warmup, reporte en MB→GB.

---

## 2. Matriz de barrido

Grilla usable efectiva: ~11.x GB tras driver/contexto CUDA (medir el contexto vacío una vez y
restarlo). Todo con GroupNorm(8), depth=4, AMP 16-mixed, AdamW (lo de exp002), salvo donde se
indique.

**Estrategia: buscar el borde, no el producto cartesiano.** Para cada combo
(dim, block, ckpt) hacer búsqueda del borde en dos pasadas — (a) batch=1, rampa el tamaño
espacial hasta OOM; (b) en un tamaño operativo, rampa batch hasta OOM. O(log) probes por combo.

### 2A. 2D con contexto axial (ancla Fase 1–2, exp002-like)

| dim | block | spatial | base_feat | in_ch | batch | ckpt |
|---|---|---|---|---|---|---|
| 2D-ctx | unet | 256×256 | 16 | 5 (A0-like) | 1 → OOM | off |
| 2D-ctx | unet | 256×256 | 16 | 11 (A4-like) | 1 → OOM | off |

Objetivo: confirmar que entra holgado y reportar **batch máximo viable** (alimenta la regla de
escala por pasos de gradiente §5). Canales como bookends A0(~5)/A4(~11) — baratos, solo tocan
primeras capas. No barrer `base_feat` acá (queda fijo en el de exp002=16).

### 2B. 3D volumen entero (Fase 3, U-Net vanilla)

Padear al **máximo global** múltiplo de 16: **176×128×256** (worst case). Probar también un
tamaño típico ~160×128×208.

| dim | block | spatial (D×H×W) | base_feat | in_ch | batch | ckpt |
|---|---|---|---|---|---|---|
| 3D | unet | 176×128×256 | 16 | 5 | 1 | off |
| 3D | unet | 176×128×256 | 16 | 5 | 1 | on |
| 3D | unet | 176×128×256 | 16 | 11 | 1 | on/off |
| 3D | unet | 176×128×256 | 32 | 5 | 1 | on/off |
| 3D | unet | 160×128×208 | 16 | 5 | 1 → OOM | off |

**Hipótesis a testear:** el volumen entero entra para vanilla-3D a base_feat=16 → **no harían
falta parches** en esta rama (Eclipse ya recortó el FOV). Si entra, medir hasta qué `base_feat`
/ batch aguanta. Si NO entra ni a batch=1 con ckpt, recién ahí caer a parches (§2D).

### 2C. 3D MedNeXt kernel-5 (Fase 3, target real — donde muerde)

Mismo volumen. MedNeXt-k5 es **mucho** más pesado que U-Net vanilla; es el envelope que de
verdad limita Fase 3. Perfilar un bloque tipo MedNeXt (kernel 5), no solo vanilla — si se
perfila solo vanilla, el número de Fase 3 sale optimista y miente al llegar a MedNeXt.

| dim | block | spatial | base_feat | in_ch | batch | ckpt |
|---|---|---|---|---|---|---|
| 3D | mednext-k5 | 176×128×256 | 16 | 5 | 1 | off |
| 3D | mednext-k5 | 176×128×256 | 16 | 5 | 1 | on |
| 3D | mednext-k5 | (parche si OOM) | 16 | 5 | 1 | on |

### 2D. Parches 3D (solo si 2B/2C OOMean a volumen entero)

Si hay que parchear, **no cúbico**: el volumen natural es anisotrópico (X hasta 251, Z ~160) y
la dosis es no-local. Probar al menos un parche que respete el aspecto natural (más in-plane /
Z generoso) contra uno cúbico, para no encajonar Fase 3 en cúbico por default. Registrar el
parche máximo que entra con ckpt on.

---

## 3. Metodología (cada ítem evita reportar un envelope falso)

- **Step completo, no forward-only.** El OOM pega en backward / `optimizer.step`. (El viejo ya
  lo hace bien — mantener.)
- **Optimizer materializado.** AdamW guarda estados m/v ≈ 2× los pesos; se materializan en el
  primer `step()`. Incluirlo (ya incluido vía el módulo real).
- **`max_memory_reserved` es el techo**, no `allocated`. Reportar ambos pero decidir "viable"
  con reserved.
- **Headroom ~1 GB.** No reportar el borde exacto del OOM como viable — fragmenta y OOMea
  intermitente. Viable = el config más grande que sobrevive los 3 steps con ≥1 GB reserved
  libre sobre lo usable.
- **`cudnn.benchmark`**: con padeo a tamaño fijo (recomendado, ver §4) es seguro y conviene on.
  Perfilar con el mismo valor que usará el training. Registrarlo como columna (bit del config).
- **`step_time_ms` en la tabla.** El costo del gradient checkpointing es tiempo (~+30%); hay
  que poder ponerle precio a la memoria que compra.

---

## 3bis. Nota de regímenes futuros (NO Fase 0 — no perfilar ahora)

La red opera sobre la **grilla de dosis nativa con su resolución propia** (no se resamplea a la
grilla del CT). Fase 2 = normo VMAT → 2.5×2.5×3 mm, grillas recortadas por Eclipse (~256
in-plane, ~150 en Z). Ese es el único régimen de este barrido.

Al entrar RC y SBRT (fases muy posteriores) el spacing baja a 1×1×1 (SBRT a veces 1×1×2) y hay
que **re-perfilar**. Ojo con el instinto "1mm = todo más pesado" — los dos regímenes no escalan
igual:

- **Radiocirugía (RC):** 1mm pero volúmenes chicos en x,y **y** z (lesiones pequeñas, FOV
  apretado). Grilla fina × extent chico → conteo de vóxeles similar o **menor** al de normo a
  2.5mm. Probablemente el régimen **más liviano**; entra entero con margen.
- **SBRT:** 1mm (o 1×1×2) con extents **grandes** (pulmón, columna). El conteo de vóxeles se
  dispara → es el **worst case de memoria de todo el proyecto** y el que casi seguro **fuerza
  parches** incluso en U-Net vanilla. El diseño de parches definitivo se calibra contra SBRT,
  no contra normo ni RC.

No perfilar RC/SBRT en Fase 0. Solo queda registrado para que, al comisionar esas fases, se
re-corra el barrido y se sepa de antemano que SBRT es el que ata.

---

## 4. Nota que conecta con 0.1 (datamodule)

Las grillas varían por paciente (Z 134–161, Y 89–115, X 145–251). Recomendación para el
datamodule nuevo: **padear todas a un tamaño fijo global** (p.ej. 176×128×256) en vez de por-batch
variable. Razón doble: (a) habilita `cudnn.benchmark` sin thrashing, (b) hace que el envelope
medido a ese tamaño sea exactamente el del training. Si en cambio se padea variable, el envelope
a máximo sigue siendo cota superior válida, pero se pierde benchmark. Decisión a registrar en 0.1.

---

## 5. Estructura de `results/profiling/envelope.md`

**Tabla cruda**, una fila por config corrido:

```
dim | block | spatial | batch | in_ch | base_feat | ckpt | amp | cudnn_bench |
peak_reserved_GB | peak_alloc_GB | step_time_ms | status(OK/OOM) | headroom_GB
```

**Sección A — Configs operativos recomendados**, separados por consumidor:
- Fase 1–2 (2D-ctx): batch recomendado + su justificación.
- Fase 3 vanilla-3D: volumen entero sí/no; si sí, base_feat/batch techo.
- Fase 3 MedNeXt-k5: volumen entero o parche; parche recomendado si aplica.

**Sección B — Decisiones que el envelope resuelve** (escribir explícitas, una línea c/u):
1. ¿Hace falta downsamplear la grilla de dosis? (Apuesta: NO — las grillas ya son chicas.
   Downsamplear mata el gradiente de dosis cerca del PTV = la prioridad de fidelidad del
   proyecto; preferible parchear antes que downsamplear.)
2. ¿Parches obligatorios en 3D? ¿En vanilla, en MedNeXt, o en ninguno?
3. Si hay parches: ¿qué forma (anisotrópica vs cúbica) y qué tamaño máximo?
4. ¿Gradient checkpointing necesario? ¿En qué configs, y a qué costo de tiempo?
5. Batch máximo viable por dim → alimenta la regla de escala por pasos de gradiente (§5).
6. Rango de `base_feat` legítimo para el B-eje de Fase 3.

---

## 6. Orden de ejecución sugerido

1. Medir contexto CUDA vacío (restar del usable).
2. Correr 2A (rápido, confirma el ancla trivial).
3. Correr 2B (la pregunta grande: ¿entra el volumen entero en vanilla-3D?).
4. Correr 2C (MedNeXt — el límite real).
5. 2D solo si algo de 2B/2C OOMeó.
6. Escribir `envelope.md` (tabla + secciones A/B).
7. Handoff al chat: `envelope.md` + una línea por decisión de la Sección B.
