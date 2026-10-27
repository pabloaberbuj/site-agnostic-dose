# HO#2b — Fase 2: baseline no-aprendida (DVH promedio por sitio/rol)

> Prompt para Claude Code. Leer con `CHARTER.md` (§2 métricas y control, §7). Corre **en paralelo**
> a HO#2 (run de A0) — no depende del datamodule de train.
> **Alcance:** montar el control permanente de Fase 2. Es la vara científica real: responde si la
> red aprende algo más que "la forma típica del protocolo" (§2). NO es ML clásico (descartado, §2/§7).

## Inputs

Datos `data/npz_v2b_body/`, split `data/splits/split_v2.json`, hash de tabla `2a0c2037686c2978`.
Script nuevo en `scripts/baseline_dvh_promedio` (§6).

## Qué es

Para cada `(sitio, rol de estructura)`, una **DVH promedio poblacional** construida **solo del
split train**. Funciona con cualquier cantidad de OARs (una curva por rol, no un modelo por
paciente) — por eso reemplaza al control clásico, que no generaliza al target de DVH completa con N
variable de OARs.

## Fit (solo train)

- Juntar, por `(sitio, rol)`, las DVH de todos los pacientes de **train** para ese rol, sobre la
  **dosis normalizada** (D95(PTV_High)=100%), eje de dosis en **% de Rx_High**, DVH sobre **volumen
  nativo**.
- Promediar → una curva por `(sitio, rol)`. **Nunca** usar val/test en el fit (sería leakage).
- Guardar las curvas por `(sitio, rol)` para trazabilidad.

## Predict + score

- La "predicción" de la baseline para una estructura de un paciente = la DVH promedio de
  `(su sitio, su rol)`. No hay dosis 3D → la baseline se scorea **solo en espacio DVH** (sin
  MAE body, sin SSIM; esas son de la red).
- Scorear con el **mismo harness `dvh_curva_completa`** que HO#2 (módulo compartido `src/eval/`):
  `mean|ΔV(D)|` por banda, por estructura, **por sitio, con IC95 bootstrap** (resampleo de pacientes
  dentro del sitio). Si el módulo todavía no existe (este handoff llegó primero), montarlo según el
  contrato de HO#2: eje % Rx_High, banda media 40–80% resaltada para OARs, banda alta para PTV.
- Evaluar sobre **val** (alineado con el driver del sweep de A0; test reservado).

## Record

Datos `npz_v2b_body`, `split_v2.json`, hash de tabla `2a0c2037686c2978`.

## Handoff de vuelta (§8)

Solo el delta: curvas promedio por `(sitio, rol)` + `metrics.csv`/`summary.json` por sitio con IC95
+ resumen corto. Queda como control permanente que se compara contra cada paso del eje A.
