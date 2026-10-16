# Paso 4 (0.2) — Contaminación sobre la dosis (no sobre nombres de estructura)

> Script: `scripts/audit_dose_extent.py`. Indicador: extensión craneo-caudal (Z) del
> volumen que recibe ≥X% de Rx_High, medida sobre la dosis cruda (no sobre
> estructuras). Calibrado con los 2 casos conocidos de Pelvis
> (`PT_49f9ceef1c10627d`, `PT_258346f1e40fbf8f`). Detalle en
> `paso4_contaminacion_dosis.csv`.

## Distribución por sitio (mm, craneo-caudal)

| sitio | umbral | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| CyC | ≥10% Rx | 207 | 243 | 250 | 276 | 291 |
| CyC | ≥50% Rx | 186 | 219 | 231 | 255 | 267 |
| Pelvis | ≥10% Rx | 252 | 294 | 426 | 445 | 450 |
| Pelvis | ≥50% Rx | 222 | 270 | 393 | 409 | 420 |
| Prostata | ≥10% Rx | 108 | 123 | 126 | 131 | 237 |
| Prostata | ≥50% Rx | 87 | 99 | 104 | 107 | 219 |

## Validación del indicador

**Pelvis:** salto brusco entre p90 (270mm @50%) y p95 (393mm @50%) — el indicador
separa limpio dos poblaciones, no es un continuo. Los 2 casos conocidos
(`PT_258346f1e40fbf8f`=408mm, `PT_49f9ceef1c10627d`=420mm @50%) caen junto al máximo,
como se esperaba. **De los 14 pacientes de Pelvis por encima de 300mm (@50%), 13 YA
eran `out_of_convention`** (los mismos de Paso 2, confirmados también por este método
independiente — buena señal de que el criterio de Paso 2 es sólido).

**Prostata:** un solo outlier extremo (237mm vs p99=131mm @10%) —
`PT_fe24a2bdb79bd38a`, el mismo paciente ya marcado `out_of_convention` hoy en Paso 3
(`PTV_Low` distinto, probable lecho prostático). Confirmación cruzada independiente.

**CyC:** distribución suave, sin outliers — sin indicio de contaminación por este método.

## Hallazgo nuevo (no resuelto): 1 paciente de Pelvis, borderline

`PT_0ee66af2dd4c0880`: extensión de dosis 306mm (@50% Rx), 330mm (@10% Rx) — por
encima de 300mm, en el rango de los 13 ya conocidos, pero **nunca fue evaluado en
Paso 2** porque el chequeo de tamaño de campo (Y del RTPLAN >300mm) de esa ronda solo
se aplicaba a pacientes con `Kidney` contorneado — este paciente tiene una lista de
estructuras completamente estándar (`Bladder`, `Bowel`, `Rectum`, `PTV_High`, etc.,
sin Kidney/SpinalCanal/Ovary/Liver), así que nunca entró a ese filtro.

Verificado independientemente: su campo de tratamiento (RTPLAN, jaw Y) mide **308mm**
— por encima del mismo umbral de 300mm usado en Paso 2. Dos señales independientes
(campo RTPLAN + extensión real de dosis) coinciden en que es el mismo patrón que los
13 ya marcados `out_of_convention`, solo que el criterio de Paso 2 no lo alcanzaba por
estar scopeado a "pacientes con Kidney".

**Resuelto por el usuario (2026-10-16):** revisado en Eclipse, el plan está OK — NO se
marca `out_of_convention`. El caso límite (308mm de campo, 306mm de extensión de
dosis) no corresponde a un campo extendido real pese a estar justo por encima del
umbral de 300mm; queda como evidencia de que el umbral es un indicador, no una regla
mecánica, y que casos borderline ameritan revisión directa en Eclipse en vez de
aplicar el corte ciegamente.

## Paso 4a: cerrado
