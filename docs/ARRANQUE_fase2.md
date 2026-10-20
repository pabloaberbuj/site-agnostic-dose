# Arranque — Fase 2: Primer site-agnostic (A-eje de input, SOLO VMAT)

> Para abrir el chat de Fase 2. Leer con `CHARTER.md` (Tier 1, §4 mapa de fases y §3 representación
> de input), `aprendizajes_transversales.md` (Tier 3) y `fase1_cerrada.md`.
> Fase 1 está cerrada: el framework nuevo reproduce el pipeline viejo en inferencia y en loop
> (des-riesgado el refactor). Al cerrar Fase 2, ese chat genera el MD que la congela y el arranque
> de Fase 3.

---

## Qué cambia respecto de Fase 1 (lo esencial)

Fase 1 fue un **control** sobre datos del proyecto VIEJO y la ruta legacy (5 canales PSDM horneados,
D95(PTV)=100%, 1 sitio). **Fase 2 es el primer experimento de verdad y cambia el dato:**

- **Datos nuevos de site-agnostic (Fase 0), NO los de Fase 1.** NPZ `npz_v1_*` (o la versión
  vigente), **solo pacientes aprobados**, **3 sitios** (próstata + pelvis + CyC), **VMAT-only**
  (confirmado N>100 c/u). Normalización **D95(PTV_High)=100%** (no D95(PTV)). Grilla de dosis nativa.
- **Datamodule nuevo, no el legacy.** El NPZ guarda **geometría, no canales de goals**: lleva dosis
  normalizada + máscaras por rol canónico + metadata. Los canales de goals (máximos / medias /
  volumétrico) se **construyen al cargar** desde la tabla de armonización versionada → cada
  experimento registra el **hash de la tabla** que usó. Esto es lo que permite las ablaciones A1a/A1b
  y las augmentations A2 sin regenerar NPZ.
- **Reporte por sitio siempre, con IC bootstrap.** El MAE global promedia y esconde. Baseline
  no-aprendida (DVH promedio por sitio/rol) como control permanente en paralelo.

## La vara que dejó Fase 1

`dvh_curva_completa` (métrica principal, `mean|ΔV(D)|` por banda; **banda media 40–80% crítica para
OARs**), sobre próstata hipo con la U-Net de dosis portada. **Objetivo de Fase 2: batir esto**, con
el site-agnostic sobre el NPZ nuevo. Referencia por banda media: **Rectum 7.30, Bladder 3.34** (PTV
concentra su error en la banda alta, 4.82). Detalle en `fase1_cerrada.md`.

*Ojo:* la vara se fijó sobre datos/normalización del proyecto viejo (1 sitio, D95(PTV)). En Fase 2
cambia la cohorte y la normalización, así que no es una comparación 1-a-1 — es la referencia de
orden de magnitud y de forma (qué banda duele), no un umbral a igualar al decimal.

## Plan de Fase 2 — A-eje de input, incremental (CHARTER §4)

MAE puro, U-Net 2D fija, métrica por sitio. Ablaciones de input, una por vez:

- **A0** — PTV con Rx + OARs como geometría consolidada (sin valores de goal) + body. Piso sin
  intención: mide cuánto aportan los goals.
- **A1** — + goals de OAR en 3 canales (máx / media / volumétrico, §3). Resolver adentro, una a la vez:
  - **A1a:** "sin goal de este tipo" = 0 vs. 95% Rx (default de Boost).
  - **A1b:** canales uniformes vs. modulados por distancia al PTV. Empate → uniforme (más simple).
- **A2** — + augmentations de goals (omitir un goal, suavizado, ruido, escala ×[0.8,1.2]). Ataca el
  riesgo de que los goals protocolo-estáticos funcionen como huella de sitio.
- **A3** — + prioridad (canal separado; `W=2^(1−P)`, máximo en solapamiento).
- **A4** — + PTV distance map / PSDM. Test del hallazgo de próstata en multi-sitio (en Xiong no aportó).
- **A5** — + densidad másica (NO CT crudo). Test de si "el CT no aporta" era "el CT crudo no aporta".
- **A6** — + beam plate map (templado del protocolo, no del RTPLAN).

Decisión de arranque congelada: **VMAT-only** elimina el confounder de técnica por construcción (no
hay token). IMRT entra como fase propia.

## Insumos / deuda heredada de Fase 1 a resolver ANTES o AL INICIO de Fase 2

1. **Requirements del repo nuevo (primer insumo de infra).** Fase 1 corrió sobre el venv del
   proyecto viejo (torch 2.12, PL 2.6.5). El repo nuevo tiene que quedar self-standing: pinnear un
   `requirements`/entorno reproducible antes de empezar a entrenar el site-agnostic.
2. **Datamodule site-agnostic** (goals-al-cargar desde la tabla + hash por experimento + canal de
   intención con ratios `Rx_nivel/Rx_High` prescriptos, NO leídos de la dosis). Es lo nuevo central
   de A0/A1; el legacy de Fase 1 no se reusa para esto.
3. **Baseline no-aprendida** (DVH promedio por sitio/rol) montada en paralelo como control.
4. **Rama 3D rota** (`build_model_3d` ausente en `unet3d.py`): no bloquea Fase 2 (U-Net 2D fija),
   pero la necesita Fase 3. Anotado.

## Decisiones congeladas que ahora aplican (CHARTER §5)

Sampling por paciente; GroupNorm; bilinear upsample + conv; factor de escala por pasos de gradiente
(`factor = train_A/train_B`, recalcular cada vez que cambie N de train); `scheduler.max_epochs` fijo
al horizonte completo; red sobre grilla de dosis nativa con padding a tamaño fijo; protocolos por
sitio con `protocol_id`; resolución de estructuras en 4 salidas (canónico/alias/ignore/falla
ruidosa), PRV no se ignoran; pacientes fuera de convención como conjunto reservado (zero-shot), no
al train; QA gate offline sobre Gy absoluto.

## Flujo (§8)

Un chat = una fase. Handoff chat↔Code por archivo (no se pega el doc entero): al volver de Code se
trae solo el **delta** (metrics.csv + summary.json + resumen corto). Tier 2
(`contexto_fase_vigente.md`) se **reescribe** a Fase 2 al arrancar (no acumula desde Fase 1).

## Preguntas vivas que Fase 2 puede tocar (CHARTER §10)

- A1a (valor "sin goal") y A1b (uniforme vs. modulado) se resuelven dentro de A1.
- Métricas casi-máximo volumétricas (D1cm³, D0.5cm³, D2%): a qué canal van — se fija en la tabla de
  armonización, afecta Bone_Mandible / Joint_TM / Brain.
- Regla de solapamiento por canal de goals (mínimo en máximos natural; en medias raro).
- Viabilidad del Pareto-conditioning (Fase 5+): depende de la auditoría multi-plan de Fase 0.
