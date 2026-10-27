# HO#4 — Fase 2: A1 — goals de OAR en 3 canales (build de A1a)

> Prompt para Claude Code. Leer con `contexto_fase_vigente.md` (Tier 2), `CHARTER.md` (§3, §4 A1, §5, §10)
> y `ARRANQUE_fase2.md`.
> **Independiente de HO#3** (A1 consume `src/eval/` sin modificarlo). Se mide contra el piso de A0
> (`checkpoints/exp_a0_001_unet2d/epoch=097.ckpt`) y la baseline, ya existentes.
> La verificación de tabla/NPZ ya se hizo; las 6 decisiones abiertas están **resueltas abajo**.
> **Alcance:** editar la tabla (ruteo §10), extender el datamodule a 6 canales y correr **A1a**.
> **A1b** queda como follow-up contingente. NO entra A2.

## Edición de tabla — ruteo §10 explícito (bumpea el hash)

El ruteo de los casi-máximo volumétricos vive **en la tabla, no en código** (decisión congelada §5/§10;
además así el hash lo captura). Agregar un campo `channels:` explícito a los constraints `type: D` con
volumen, con la regla: **volumétrico siempre; + máximos solo si el OAR no tiene `Dmax` propio.**

- `Bone_Mandible`, `Joint_TM_L/R` (`D 2%`, sin Dmax) → `channels: [volumetrico, maximos]`.
- `Brain` (`D 1cm³` + Dmax propio) → `channels: [volumetrico]`.
- Dmax / Dmean / V quedan implícitos (no ambiguos), no necesitan el campo.

Esto **cambia el hash** de la tabla (hoy `2a0c2037686c2978`). A0 conserva `2a0c2037` (no usó goals);
**A1 registra el hash nuevo** en el record de cada experimento. Reportar el hash nuevo al volver.

## Datamodule — extensión a 6 canales

A A0 (Ch0 intención/PTV, Ch1 OAR geometría consolidada, Ch2 body) se le agregan 3 canales de goals, en
**fracción de Rx_High** (1.0 = Rx_High, igual que Ch0), leídos de la tabla vía protocolo/sitio. Se usa
el valor del goal (`limit`), no `tol`. Gy → fracción dividiendo por `rx_gy.PTV_High`; `pct_rx` ya está
en escala. **Ch1 se mantiene** (OARs sin goal numérico siguen visibles). Layout: **6 canales**.

- **Ch3 — máximos:** vóxeles del OAR con techo `Dmax`, **uniforme** sobre la estructura.
- **Ch4 — medias:** vóxeles del OAR con `Dmean`, **uniforme** sobre la estructura.
- **Ch5 — volumétrico:** techo pintado solo en la **fracción preservada** (distal al PTV). Reglas:
  - Fracción preservada = distal `(1 − budget)` por ranking de distancia al PTV (EDT). Sacrificada
    (proximal `budget`) → `fill_sin_goal`.
  - **Múltiples V en un OAR (Bladder, Rectum, FemoralHead): escalera-min.** Cada vóxel toma el techo
    más estricto entre los constraints cuya fracción preservada lo incluye; proximal a todas → fill.
    Da una escalera monótona (dosis más baja distal, más alta proximal).
  - **Límites en cm³:** fracción con volumen real del OAR (`1 − cm³/vol`). Si `vol ≤ cm³` (constraint
    trivial, p.ej. `Glnd_Thyroid V60≤10cm³` en 20/97 pacientes) → **no pintar** (esos vóxeles → fill).

### `fill_sin_goal` — semántica (confirmada)

Se aplica a **todo vóxel de OAR sin goal de ese tipo en ese canal**, incluida la fracción proximal
sacrificada de un OAR con goal volumétrico. `0` ≡ fondo; `"95pct_rx"` ≡ 0.95. Como la geometría de OAR
ya está en Ch1, `fill=0` NO pierde el "acá hay OAR" → hipótesis a medir: `fill=0` puede andar bien
(debilita el argumento de Boost para 95%).

### Contradicción PTV↔OAR en el overlap — se pinta, NO se bloquea

Si un vóxel cae en PTV (Ch0 ~100%) y en fracción preservada de un OAR (Ch5 ≤techo), se pintan **ambas**.
Los canales son enunciados de deseo; la contradicción es tensión clínica real y su resolución la aporta
el target (dosis alcanzada) ahora y el canal de prioridad en A3 — NO se fusiona acá (§3). Caso borde
`overlap > budget`: el presupuesto se consume en vóxeles a distancia 0 (empate) → **desempate
determinista** (orden de índice) para reproducibilidad del hash; qué overlap queda preservado es
arbitrario y de bajo impacto.

### Toggles (construir ambos ahora)

- `fill_sin_goal ∈ {0, "95pct_rx"}` (A1a varía este).
- `goal_mode ∈ {uniform, distance_modulated}` (A1b; acá solo el hook).
- EDT: `scipy` sobre la **unión de PTVs** con espaciado nativo. *Nota:* ignora el nivel de dosis (el
  charter lo quiere ponderado por dosis en multi-PTV) → base simple para A1a; puede mis-ordenar la
  fracción preservada en **CyC**. Si el volumétrico de CyC sale raro, fallback = distancia a `PTV_High`.
  La modulación ponderada es lo que ataca A1b.

El stem de la U-Net pasa a 6 canales; todo lo de abajo congelado (§5).

## Fijo (igual que A0)

U-Net 2D, MAE puro, sampling por paciente, GroupNorm, bilinear+conv. **Epoch-scaling sin cambios**
(N train = 280 → `factor = 265/280`, horizonte ~189); `scheduler.max_epochs` al horizonte. Mismo harness
`src/eval/` (por sitio, IC95 bootstrap, banda media OARs / alta PTV). **Val** decide; **test reservado**.

## Plan de corridas

- **A1a (este handoff):** `goal_mode=uniform` fijo, variando `fill_sin_goal` = `0` vs `"95pct_rx"`.
  Elegir ganador contra el piso de A0 en val (OAR banda media, por sitio, bootstrap). Reportar A1a vs
  A0 vs baseline.
- **A1b (follow-up, NO ahora):** sobre el `fill_sin_goal` ganador, `uniform` vs `distance_modulated`.
  Empate → uniform.

Una a la vez; no saltar a A2 hasta cerrar A1.

## Items logueados (no dropear, no bloquean A1a)

- **`conditional: true`** (Eye/Parotid/Thyroid): tratado como goal normal en A1a, pero **logueado como
  diferido a A3** (probablemente codifica algo cercano a prioridad). `tol`: ignorado.
- **Próstata con roles sin goal** (`Kidney_L/R`, `PenileBulb`, `SpinalCord_PRV05`): son autocontornos de
  TC más larga de lo usual (explicado). Confirmar que pasan la resolución de estructuras (4 salidas,
  canónico/alias) — si pasan, no es problema; son los vóxeles que recibe `fill_sin_goal`.
- **`PenileBulb` con goal en Pelvis pero nunca presente:** entrada muerta de la tabla, anotar (higiene).

## Lectura esperada

A1 debe empujar la **banda media de OARs** por debajo del piso de A0 (CyC 2.21 / Pelvis 3.30 /
Próstata 2.20). El **PTV no se espera que se mueva** (goals son de OAR; palancas del PTV = A4/A6). Si no
mueve OARs vs A0: señal fuerte (goals no aportan como canal o mal codificados) → se discute antes de A2.

## Handoff de vuelta (§8)

Delta: hash nuevo de tabla + `metrics.csv` + `summary.json` + comparación A1a vs A0 vs baseline por sitio
(IC95) + cuál `fill_sin_goal` ganó + resumen corto. Con A1a cerrado, sigue A1b.
