# HO#1 — Fase 2: infra self-standing + datamodule-A0 + smoke test

> Prompt para Claude Code. Leer con `CHARTER.md` (§5 decisiones congeladas, §6 repo/datos, §7
> aprendizajes), `ARRANQUE_fase2.md` y `fase1_cerrada.md`.
> **Alcance de este handoff:** dejar el repo nuevo self-standing, montar el datamodule de A0
> (geometría + intención, SIN canales de goals) y validarlo con un smoke test. **No se entrena
> todavía.** Goals (A1), augmentations (A2), baseline y loop de train van en handoffs aparte.

## Antes de escribir código — inspeccionar y reportar

No asumir el esquema. Primero:

1. Abrir **un NPZ por sitio** (próstata, pelvis, CyC) de la versión vigente (`npz_v1_*` o la que
   corresponda) y volcar: keys, shapes, dtypes, qué roles de máscara trae, y qué metadata
   (confirmar que están `protocol_id` y `Rx_High` en Gy por paciente).
2. Abrir la **tabla de armonización** versionada y volcar su formato: cómo expone
   `sitio → protocol_id → (Rx por nivel)` y el mapeo de roles (cuáles son niveles de PTV, cuáles
   OAR, cuál body). Anotar cómo se calcula su **hash**.
3. Mirar el datamodule **legacy** de Fase 1 solo como referencia de interfaz (LightningDataModule,
   firma de batch). **NO forkearlo** (§5 repo nuevo; §7: bugs de `spacing_mm` y `cargar_ct`
   viajaron por copy-paste). El datamodule de A0 se escribe nuevo en `src/datamodules/`.

Reportar este esquema antes de seguir si algo no matchea lo que dice el charter.

## Deliverable 1 — entorno reproducible

- El repo nuevo tiene que quedar independiente del venv del proyecto viejo (Fase 1 corrió sobre
  torch 2.12 / PL 2.6.5). Pinnear un `requirements.txt` (o environment file) con versiones exactas
  y dejarlo en el repo. Decidir explícitamente si se matchean las versiones de Fase 1 o se pinnea
  fresco, y registrar la decisión.
- Chequeo mínimo: import limpio de torch + lightning + numpy, y que la GPU se ve (RTX A2000 12GB).

## Deliverable 2 — datamodule-A0 (`src/datamodules/`)

Carga del NPZ (el NPZ ya trae todo resuelto; el datamodule NO re-normaliza ni re-resuelve roles):

- **Target:** la dosis normalizada del NPZ, cargada directo (D95(PTV_High)=100%, ya calculada en
  preprocesado sobre volumen nativo). No tocar.
- **Máscaras:** por rol canónico ya resuelto (PRV y PTV según reglas del charter). Si aparece un
  rol desconocido en el NPZ → **fallar ruidoso**, no adivinar (la resolución es upstream, pero el
  datamodule no debe tragarse basura silenciosamente).

Construcción de los **3 canales de entrada de A0** al cargar:

- **Ch0 — intención/PTV:** vóxeles de cada nivel de PTV pintados con el ratio `Rx_nivel/Rx_High`
  **prescripto**, leído de la tabla vía `protocol_id` (High = 1.0; Mid/Low = su ratio; fondo = 0).
  **Nunca** derivar el ratio de la dosis (sería leakage). Multi-nivel (CyC) sale solo de la tabla.
- **Ch1 — OAR consolidado:** unión binaria de todos los roles OAR (PRVs incluidos donde son la
  geometría). Sin valores de goal, sin identidad por OAR.
- **Ch2 — body:** máscara binaria del contorno externo.

Reglas:

- **Sampling por paciente**, no por corte suelto (§5). U-Net 2D: se samplean los cortes del
  paciente; nada de mezclar cortes de pacientes distintos en un batch arbitrario.
- **Padding in-plane a 512×512** (grilla de dosis nativa; §5 / frozen-decisions). Sin resamplear
  la dosis a la grilla del CT.
- **La tabla es input versionado del modelo:** registrar su **hash** en el config/record del
  experimento (lo van a necesitar A1a/A1b y las augmentations de A2 sin regenerar NPZ).

Fuera de alcance explícito de A0: canales de goals (máx/media/vol → A1), augmentations (A2),
prioridad (A3), distance map (A4), densidad (A5), beam plate (A6).

## Deliverable 3 — smoke test (antes de cualquier train)

Cargar **1 paciente por sitio** y verificar:

1. **Shapes/dtypes:** 3 canales de entrada + target, 512×512 in-plane, sin NaN.
2. **Integridad de normalización:** D95(PTV_High) sobre la dosis+máscara cargadas ≈ 100%.
   *Ojo tolerancia:* el preprocesado calculó D95 sobre **volumen nativo**; el chequeo acá corre
   sobre la grilla de dosis, así que puede diferir un poco. Usar tolerancia (p.ej. ±1–2%), no
   igualdad exacta — y si se va de ahí, es señal de problema de vóxel, no ruido.
3. **Intención:** los ratios de Ch0 matchean la tabla para ese `protocol_id` (High=1.0; en CyC
   verificar Mid/Low con sus ratios reales).
4. **Roles:** ningún rol desconocido pasó silencioso; OARs consolidados ≠ vacío; body presente.
5. **Hash de tabla** registrado en el output del test.

Imprimir un reporte corto por sitio.

## Handoff de vuelta (§8)

Traer solo el **delta**, no el doc entero: el reporte del smoke test + las versiones pinneadas +
el esquema de NPZ/tabla que se descubrió + un resumen corto de qué quedó montado y qué falta.
