# Paso 2 — decisiones tomadas en modo autónomo (revisar al volver)

> El usuario pidió avanzar sin esperar confirmación en todo lo que no requiera un
> "break" explícito. Este archivo registra cada decisión de armonización tomada sin
> confirmación en vivo, con la evidencia que la respalda, para revisión posterior.
> Nada de esto se commitea a git todavía.

## Aplicadas (evidencia directa, bajo riesgo)

1. **`ignore_patterns`: `z_*`/`zring*`/`zopt*` → `z*`.** Datos reales (CyC) traen
   estructuras de optimización con el mismo convenio interno pero SIN guion bajo:
   `zArtifact`, `zBody`, `zCal_BrainStem`, `zCal_Low`, `zCal_Mid`, `zCtrol`,
   `zDosis5940`, `zHigh`, `zPTV_High57`, `zShoulders`. Las 9 instancias vistas son
   helpers de optimización, ninguna es un OAR real. `z*` es seguro porque el chequeo
   de ignore corre DESPUÉS del canónico/alias (ver `name_resolver.py`), así que nunca
   puede tapar un rol real aunque empezara con z (ninguno lo hace hoy).

2. **`exclude_names` (nuevo, lista explícita, no patrón) agregados a
   `harmonization_schema.yaml`:**
   - `Brainstem`, `SpinalCord`, `OpticChiasm`, `OpticNrv_L`, `OpticNrv_R`,
     `BrachialPlex_L`, `BrachialPlex_R`: **confirmado en datos reales** que coexisten
     con su PRV ya canónico (`Brainstem_PRV05`, etc.) en el mismo paciente, con
     volumen del PRV siempre mayor (el margen esperado) — es el órgano sin margen,
     no una estructura alternativa. Aplica directo CHARTER §5 ("el PRV es la
     geometría" en seriales).
   - `Cochlea_PRV03_L`, `Cochlea_PRV03_R`, `Bowel_PRV`: mismo razonamiento al revés
     — acá el canónico de la plantilla es el órgano llano (no serial en este
     protocolo), así que el PRV extra que aparece en algunos pacientes es el que
     sobra.
   - `PTVp`, `PTVn`, `PTV_Prostate`, `PTV_SeminalVes`: ya documentados como "NO
     aliasar" en `patient_exceptions.yaml` (comentarios existentes del usuario);
     formalizado acá como `exclude` en vez de quedar como `unknown`.

3. **Alias `cavity_oral` → `Oral_Cavity`** (`MANUAL_OAR_ALIASES`): mismo nombre,
   orden de palabras invertido. Sin ambigüedad.

4. **Resta sobre PTV: agregado el separador `!`** (`Bladder!PTV`, `PTV!PRV`,
   `Rectum!PTV`) a la regla ya existente de `-PTV` (`Bone_Mand-PTV_Hi`). Mismo
   patrón semántico, el separador real variaba.

5. **BODY ya no cuenta como "estructura desconocida"** en el pipeline (se identifica
   aparte, antes del chequeo de nombres) — esto era un bug del código, no una
   decisión de armonización.

## Segunda ronda (tras correr el preprocesador completo sobre los 503)

6. **GTV_\* → `exclude`**, mismo criterio que CTV (objetivo de optimización, no
   intención). Cubre `GTV`, `GTVn`, `GTVn2/3`, `GTV_N`, `gtvn`, `GTVp`, `GTVp_High`,
   `GTV_n_boost`, `GTV_CHHiP`, `GTVresidual`.

7. **Alias de formato/abreviatura/castellano** (mismo órgano y mismo margen que un rol
   ya canónico, solo cambia orden de palabras o padding del número de margen):
   `Cabeza_femoral_L/R`→`FemoralHead_L/R` (castellano); `BrchPlex_L_PRV05`/`R_PRV05`→
   `BrachPlex_PRV5_L/R` (abrev. + orden invertido); `OpticNrv_L_PRV03`/`R_PRV03`→
   `OpticNrv_PRV03_L/R` (orden invertido); `OpticChiasmPRV03`→`OpticChiasm_PRV3`
   (mismo margen 3mm, sin guion bajo).

8. **`exclude_names` ampliado:**
   - `Cochlea_L_PRV03`/`Cochlea_R_PRV03` (orden invertido de los ya excluidos
     `Cochlea_PRV03_L/R` — mismo motivo).
   - `OpticChiasm_PRV2`, `OpticNrv_PRV02_L/R`: margen **2mm**, DISTINTO del 3mm ya
     canónico para ese órgano — no se funde (conflaría dos geometrías clínicas
     distintas), se reporta aparte.
   - `PTV_Uterus`, `PTV_LN_Pelvics`, `PTV_LN_Inguinofe`, `PTV_LN_Iliac`,
     `PTV_Pelvics`: mismo patrón que `PTVp`/`PTVn` ya excluidos — subcomponentes
     primario/nodal de `PTV_High` en Pelvis, aparecen en **decenas** de pacientes
     (no son casos raros). NO incluí `PTV_Retro` acá — ese probablemente ES el target
     real de los casos de campo extendido (ver pendiente #2 abajo), no un
     subcomponente a excluir.

9. **`ignore_patterns` ampliado:** `*marcador*` (castellano de marker), `*control*`
   (`Control Region`, `NS_Control`), `dose*`/`dosis*` (isodosis convertida a ROI,
   ej. `Dose 107[%]`, `Dosis 108[%]`), `match*` (`Match points`, unión de isocentros).

## Tercera ronda (tras terminar la corrida completa de Prostata)

10. **Alias muy frecuentes en Prostata** (sinónimos estándar, no ambigüedad):
    `Bowel_Bag`→`Bowel_Small` (técnica "bowel bag", mismo órgano que ya modelamos),
    `Colon_Sigmoid`/`Colon-Sigmoid`→`Sigmoid` (~45 pacientes — muy probable que sea
    el mismo Sigmoid del protocolo, nombre más verboso de esta clínica; no 100%
    verificado con volúmenes), `penilbulb`→`PenileBulb` (sin guion bajo + minúscula).
11. **`PTV_Pelvis` (singular) agregado a `exclude_names`** junto a `PTV_Pelvics`,
    mismo patrón.

## Cuarta ronda (tras terminar Pelvis)

12. **`ignore_patterns`: `opt_*` → `opt*`** (mismo razonamiento que `z*`: visto
    `OPTPTV_Low` concatenado sin guion bajo).
13. **Alias `bowelbag` (sin guion bajo) → `Bowel_Small`**, y typo `PTV_Uterurs` →
    `exclude_names` (mismo motivo que `PTV_Uterus`, un solo caracter de diferencia).
14. **`RectumPTV`/`BladderPTV` (resta OAR+PTV concatenada, sin separador) →
    `exclude_names`**, mismo patrón semántico que `-PTV`/`!PTV` pero sin signo.

**Hallazgo que sube de prioridad:** en Pelvis, el cluster "campo extendido"
(`Kidney_L/R`, `Liver`, `SpinalCanal`/`SpinalCanal_PRV` y variantes, `Heart`,
`Lung_L/R`) aparece en **una fracción grande del sitio** (decenas de los 161, no un
puñado) — ver pendiente de abajo. No es ruido de tipeo: es una subpoblación clínica
que el template de `sites_pelvis.yaml` no cubre. Prioridad alta para tu revisión.

## Quinta ronda (tras terminar CyC)

15. **Bug de robustez corregido (no es decisión de armonización):** `rasterize_structure`
    fallaba con `GeometryError` cuando un contorno (típicamente `BODY`) cae fuera del
    rango Z de la grilla de dosis — normal, el RTSTRUCT suele cubrir más CT del que
    Eclipse calculó dosis. Ahora esos contornos se descartan en vez de fallar (no hay
    dosis ahí, no pueden aportar a ningún cálculo sobre la grilla nativa igual).

16. **Regla nueva en `name_resolver.py`: duplicado numerado de un rol conocido.**
    `Cochlea_L1`, `SpinalCord2`, `Trachea1`, `A_Carotid_L1`, `Musc_Constrict1` →
    mismo rol que `Cochlea_L`/`SpinalCord`/etc. sin el dígito final (recontoreado).
    **Explícitamente NO se aplica a nombres que empiezan con PTV/GTV/CTV** — ahí el
    número es semántico (`PTV_Mid01`/`PTV_Mid02` son niveles DISTINTOS, no un
    duplicado; ver pendiente de abajo). Cubre la familia sin enumerar cada caso.

17. **Alias de formato:** `PituitaryGland`→`Pituitary`, `TMJoint_L/R`→`Joint_TM_L/R`
    (orden invertido), `Lens Left/Right`→`Lens_L/R` (inglés con espacio).

18. **`exclude_names` ampliado:** `BrachialPlexus_L/R` (palabra completa, mismo
    organo que `BrachialPlex_L/R` ya excluido), `Chiasm` (forma corta de
    `OpticChiasm` ya excluido), `SpinalCord_PRV1` (margen no estándar), `PTV_total`/
    `PTV_Low_total` (combinación de niveles, no un nivel propio).

19. **`ignore_patterns` ampliado:** `ns_*` (prefijo interno de varias estructuras de
    trabajo: `NS_Control`, `NS_NormalTissue`, `NS_LN`, `NS_Ring`, `NS_UTERUS`), `poi*`
    (Point Of Interest de Eclipse), `*_eval*` (sufijo, complementa `ptv_eval*` que ya
    cubría el prefijo), `*marcapaso*` (dispositivo), `*caliente*` (región "hot" de
    revisión), `*cal*` (helper de cálculo: `Cal`, `cal_low/high`, `Low_Cal`, `Mid_Cal`,
    `High_Cal` — ningún rol canónico contiene "cal"), `none` (artefacto de export,
    nombre literal "None").

## Sexta ronda (tras terminar Pelvis la segunda vez)

20. **Gap de diseño corregido:** `exclude_names` solo hacía match exacto, pero
    estructuras como `PTVn`/`PTVp` también aparecen con sufijo de crop (`PTVn-0.4`,
    igual que un PTV real) sin pasar por `normalize_ptv_name` (no son `ptv_<nivel>`).
    Agregué el mismo chequeo de crop (`_CROP` de `normalize_structure_name.py`) para
    `exclude_names`.
21. **`PTVn_`/`PTVp_` (guion bajo final) agregados a `exclude_names`**, mismo motivo
    que `PTVn`/`PTVp`.
22. **Alias `Bladder_P`/`Bowel_P`/`Rectum_P`→`Bladder`/`Bowel_Small`/`Rectum`**: un
    paciente (ya `reviewed_ok`) tiene todo su set con sufijo `_P`, mismo órgano.
23. **`ignore_patterns`: `ctrl*`** (forma corta de "control", mismo criterio que
    `*control*`).

## Séptima ronda (tras terminar CyC la segunda vez)

24. **Bug de robustez corregido:** cuando un paciente tiene MÁS de una estructura
    tipo BODY (ej. `Body` + `BODY1` recontoreado), el pipeline solo excluía la
    elegida del chequeo de nombres y la otra cola quedaba "desconocida". Ahora se
    excluyen todas las que matchean el patrón BODY/EXTERNAL.

**Hallazgo (no es bug, es el resolver funcionando bien):** 2 pacientes CyC tienen
dos contornos con volumen>0 que resuelven al mismo rol (`Cochlea_L`+`Cochlea_L1`,
`Musc_Constrict`+`Musc_Constrict1` — probablemente recontoreos, no error de regla).
El pipeline correctamente falla en vez de elegir uno al azar; son 2 casos, quedan
para que los mires en Eclipse.

**Confirmado con el número real de pacientes:** en CyC, `Lobe_Temporal_L/R` +
`Lobe_Tem_PRV3_L/R` (y variantes: `Lobe_Tem_PRV3_Lb/Rb`, `PRV_LobeTempL/R_03`,
`PRV_LobeTempL/R03`, `TemporalLobe_L/R`) aparecen en **~55 de los 81 pacientes que
todavía fallan** — es, con diferencia, la decisión de más impacto pendiente en todo
Paso 2. `PTV_Mid01`/`PTV_Mid02` (~20 pacientes) y `Trachea`/`Traqueostomia`/
`Pretraqueal` (~8 pacientes) le siguen en impacto.

## Octava ronda (decisiones del usuario, 2026-10-14)

25. **`Lobe_Temporal_PRV3_L/R` promovido a rol canónico en CyC.** Constraint
    confirmado por el usuario: Dmax≤70Gy, prioridad 3, sobre el PRV. Nuevo
    mecanismo `MANUAL_ROLES` (`build_harmonization.py`) + `MANUAL_OARS`
    (`convert_constraints.py`) para que estas adiciones que no vienen de ninguna
    plantilla sobrevivan a la regeneración. Alias para las 4 variantes de nombre
    vistas (`Lobe_Tem_PRV3_L/R`, `PRV_LobeTempL/R_03`, `PRV_LobeTempL/R03`) +
    exclusión del órgano llano (`Lobe_Temporal_L/R`, `TemporalLobe_L/R`). CyC pasó de
    115/196 (58.7%) a 157/196 (80.1%) OK.
26. **49 pacientes de Pelvis (campo extendido) marcados `out_of_convention`**
    (motivo `campo_extendido_a_confirmar`) por decisión del usuario ("al ser pocos
    los dejaría"). Lista completa de `patient_id` entregada al usuario junto con la
    ubicación de `mapping_ids.csv` para que cruce contra Eclipse/ARIA. Pelvis pasó de
    105/161 (65.2%) a 154/161 (95.7%) OK.
27. **Gap de diseño corregido: `out_of_convention` no alcanzaba para que el paciente
    procesara.** La excepción solo relajaba el requisito de nivel de PTV, pero las
    estructuras desconocidas (`Kidney_L`, `Liver`, etc.) seguían hecho fallar todo el
    paciente — contradecía "nunca se descarta" del CHARTER (el conjunto reservado no
    se podía generar). Ahora **cualquier** paciente con entrada en
    `patient_exceptions.yaml` (`out_of_convention` o `reviewed_ok`) tolera estructuras
    desconocidas: se ignoran y quedan logueadas en `metadata['warnings']`, no frenan
    el pipeline. Sin excepción registrada, sigue fallando ruidoso como siempre.
28. **Alias `brainstem_prv05`→`Brainstem_PRV05`** (mayúscula distinta en "Stem":
    `BrainStem_PRV05`).

## Novena ronda (decisiones del usuario sobre Pelvis, revisadas caso por caso)

29. **`SpinalCord`/`SpinalCanal` (con o sin PRV explícito) → rol canónico
    `SpinalCord_PRV05` también en Pelvis** (ya existía para CyC). Dmax≤50Gy, prioridad 1
    — valor explícito del usuario, NO reusa el de CyC (45Gy). 8 variantes de nombre
    aliased (`SpinalCanal`, `SpinalCanal_PRV`, `SpinalCanalPRV05`, `SpinalCanal_PRV5`,
    `SpinalCord_PRV`, `SpinalCord_PRVx`, `SpinalCord_PRV5`, `spinal canal`).
30. **Bug encontrado al aplicar lo anterior:** varios pacientes tienen `SpinalCanal` Y
    `SpinalCanal_PRV` como contornos SEPARADOS con volumen>0 — al aliasar ambos al mismo
    rol, el pipeline (correctamente) marcaba ambigüedad. Agregué un desempate genérico
    en `pipeline.py`: si exactamente uno de los candidatos ambiguos tiene "prv" en el
    nombre crudo, gana (geometría más explícita) — mismo criterio que Brainstem/
    SpinalCord en general, ahora aplicado a nivel de resolución, no solo de alias.
31. **`Ovary_L`/`Ovary_R` → `exclude_names`** (decisión directa del usuario).
32. **`Liver`, `Heart`, `Lung_L`, `Lung_R`, `Lungs` → `exclude_names`** ("desestimar la
    estructura", sin chequeo de tamaño de campo — a diferencia de Kidney, no son
    indicio fuerte de campo extendido por sí solos).
33. **Kidney: roles canónicos `Kidneys`/`Kidney_L`/`Kidney_R` en Pelvis**, Dmean≤15Gy
    prioridad 3 cada uno. **Criterio de distinción campo extendido vs. incidental**
    (del usuario): tamaño Y del campo de tratamiento (jaw Y/ASYMY del RTPLAN) >300mm →
    campo realmente extendido, paciente a `out_of_convention`; ≤300mm → riñón
    incidental, paciente se mantiene con el constraint aplicado. Implementado
    `max_y_field_size_mm()` en `dicom_io.py` (lee `BeamLimitingDevicePositionSequence`
    de todos los beams/control points del RTPLAN). Resultado sobre los 36 pacientes de
    Pelvis con Kidney: split limpio y bimodal — 13 por encima de 300mm (321-400mm), 23
    por debajo (216-276mm), **sin casos ambiguos cerca del umbral** — buena señal de
    que es una distinción real, no arbitraria.
34. **`out_of_convention` de Pelvis recalculado de 49 a 13** (los 2 ya conocidos +
    11 nuevos por Y>300mm). Los otros 36 de los 49 originales ya no necesitan
    exception: sus estructuras (SpinalCord/Ovary/Liver/Heart/Lung/Kidney-normal) ahora
    resuelven solas vía la tabla. Pelvis: 105/161 (65.2%) → **147/161 (91.3%)**.
35. **`PTV_LN_Pelvic`/`PTV_LN_Inguin` (singular, sin "s"/"ofe") agregados a
    `exclude_names`**, mismas variantes de las ya excluidas `PTV_LN_Pelvics`/
    `PTV_LN_Inguinofe` — ahora el chequeo de sufijo de crop también las cubre
    (`PTV_LN_Pelvic-04`, `PTV_LN_Inguin-04`).

## Décima ronda (decisiones del usuario sobre CyC, revisadas caso por caso)

36. **`Traqueostomia`/`Traqueostoma`, `Pretraqueal`, `Retina_L/R`, `Cornea_L/R`,
    `Eye_R_PRV2mm`, `Parotida_L-0.2`/`Parotida_R-0.2`, `CuelloDer`/`CuelloIzq`,
    `Nivel6`, `SCVIZQ` → `exclude_names`** (estructura desestimada, el paciente se
    mantiene — decisión directa del usuario).
37. **`PTV_Intermediate` → alias de nivel `PTV_Mid`** en `normalize_structure_name.py`
    (mismo mecanismo que `ptv`/`ptv_high_p`/`ptv_ptta`). **Nota:** ese archivo tiene una
    copia espejo en el repo del extractor (C#) — hay que agregar el alias ahí también
    si se quiere que `DicomExtract` lo resuelva igual (no lo pude tocar yo).
38. **PRV con prefijo en vez de sufijo, mismo margen → alias al rol ya canónico.**
    Confirmado en los 2 pacientes con esa convención propia
    (`PT_559aefbb26262e5c`, `PT_8a9d69ea4ef84f91`): `PRV_Brainstem05`/`BrainstemPRV`→
    `Brainstem_PRV05`, `PRV_SpinalCord05`/`SpinalCordPRV`→`SpinalCord_PRV05`,
    `PRV_BrachialPL05`/`PRV_BrachialPR05`/`BrachialP_L_PRV`/`BrachialP_R_PRV`→
    `BrachPlex_PRV5_L/R`, `PRV_OpticC03`→`OpticChiasm_PRV3`, `PRV_OpticNL03`/
    `PRV_OpticNR03`→`OpticNrv_PRV03_L/R`. `PRV_CochleaL03`/`PRV_CochleaR03`/
    `CochleaLPRV`/`CochleaRPRV` → `exclude_names` (mismo criterio que los demás PRV de
    Cochlea: el canónico es el órgano llano, no el PRV). **Resultado: ambos pacientes
    resuelven completos** — no hizo falta "quitar paciente" como se había planteado al
    principio, la regla de alias alcanzó. Enumerado explícito (no regex genérica): las
    abreviaturas de órgano varían demasiado para una regla única confiable con solo 2
    ejemplos.
39. **`Trachea`/`EyeBack_L`/`EyeFront_L` → `exclude_names` (estructura) + paciente a
    `out_of_convention`** (a diferencia de Traqueostomia/Pretraqueal, acá el usuario
    pidió sacar también al paciente — indicio de extensión mediastínica/orbitaria real,
    no solo una mención incidental). 9 pacientes: `PT_1c53a9b84029afec`,
    `PT_1f99cbd2a709b242`, `PT_29f05c0351fa8932`, `PT_3560368155841ffd`,
    `PT_5b6a74017a45908c`, `PT_909a424ff2e5e6d8` (`Trachea1`), `PT_ac70c40cbcd1ef75`,
    `PT_bc3d6255ad42ab92`, `PT_c6a05735c3c0c43e`. Los primeros dos de esa lista estaban
    en `reviewed_ok` por otro motivo (cavum) — se movieron a `out_of_convention`, el
    motivo nuevo pesa más.
40. **`PTV_C2`, `PTV66`, `PTV_Mid_3mm` → paciente a `out_of_convention`**
    (`PT_caefecceac8746a3`, `PT_271fb2489b4a962c`, `PT_ee9e7d85f565a70b`):
    convención propia de un paciente puntual, no generalizable.

## Resueltas por el usuario (2026-10-14)

- ~~Subpoblación en Pelvis con anatomía no-pélvica~~ → desglosada caso por caso
  (decisiones #29-35): mayoría resuelta vía tabla, 13/49 confirmados campo extendido
  real por tamaño de campo (Y>300mm) → `out_of_convention`.
- ~~`Lobe_Temporal_L/R` + PRV en CyC~~ → rol canónico `Lobe_Temporal_PRV3_L/R`,
  Dmax≤70Gy prioridad 3 (decisión #25).
- ~~Nombres sin resolver en CyC (Trachea, Retina/Cornea, PRV con convención propia,
  PTV_C2/66/Mid_3mm, etc.)~~ → resueltos caso por caso (decisiones #36-40).

## Undécima ronda (decisiones del usuario sobre CyC/Pelvis/Próstata, 2026-10-14)

41. **Bug encontrado y corregido (aviso del usuario):** en la octava/novena ronda,
    `Ovary_L`/`Ovary_R` habían quedado mezclados por error en el mismo `exclude_names`
    que `Liver`/`Heart`/`Lung_*`. Son decisiones DISTINTAS: Ovary = "excluirlos" (sacar
    el paciente), Liver/Heart/Lung = "desestimar la estructura" (paciente se mantiene).
    Corregido: Ovary sale de `exclude_names`, queda `unknown` adrede. 7 pacientes de
    Pelvis con Ovary/variantes detectados (`Ovary_L`, `Ovary_R`, `Ovary_R1`) → todos a
    `out_of_convention` (motivo `ovario_detectado`); 2 de ellos (`PT_01bb8559edc26ae5`,
    `PT_129a4d7fa81eedbd`) eran justo los pacientes "CHHiP" que quedaban sin explicar.
    Pelvis recalculado: 147/161 → **151/161 (93.8%)**.
42. **`PTV_Mid01`/`PTV_Mid02` en CyC — resuelto con la regla del usuario** ("si tienen
    volumen>0 eliminar paciente; si volumen=0 desestimar estructura"). Se calculó el
    volumen rasterizado real (no solo presencia del nombre) para los 22 pacientes CyC
    con estos nombres: 15 con volumen>0 real → `out_of_convention` (motivo
    `mas_niveles_ptv_que_protocolo_mid01_02`; 11 nuevos + 4 que ya estaban mal
    etiquetados como "Mid ausente" en una categoría previa — RECATEGORIZADOS); 7 con
    volumen=0 → estructura a `exclude_names` (2 nuevos a `out_of_convention` con el
    motivo previo `protocolo_distinto_sin_PTV_Mid`, 5 ya estaban). Un paciente
    (`PT_514bde984025ac6d`) estaba en `reviewed_ok` por otro motivo — se movió, el
    motivo nuevo pesa más.
43. **`SpinalCord_PRV1`/`SpinalCord_PRVx` → "eliminar paciente" (margen no estándar).**
    Los 2 únicos pacientes reales con estos nombres (`PT_caefecceac8746a3` en CyC,
    `PT_7da7b2507f0ac933` en Pelvis) YA estaban `out_of_convention` por otro motivo, así
    que no hubo que agregar pacientes nuevos. Se corrigió la tabla igual: se sacó el
    alias `spinalcord_prvx` → `SpinalCord_PRV05` (no es el mismo margen). **Bug
    encontrado al intentar dejar `SpinalCord_PRV1` como `unknown`:** la regla de
    duplicado numerado le saca el dígito final ("1"), y "SpinalCord_PRV" (sin dígito)
    YA es alias válido → el nombre terminaba resolviendo igual, generando ambigüedad
    real con `SpinalCord_PRV05` y haciendo FALLAR por completo al paciente (ni NPZ
    reservado). Corregido devolviendo `SpinalCord_PRV1` a `exclude_names` (mismo
    resultado final para el único paciente real: reservado, no en train).
44. **Próstata, decisiones del usuario:** `Colostomia`, `Protesis`, `PTV_ProstateBed`,
    `"PTV VS OK"` → `out_of_convention` (4 pacientes). `RectumCorregido` → alias a
    `Rectum`; si coexiste con `Rectum` llano, nuevo desempate genérico en pipeline.py
    (igual que "prv": si exactamente un candidato dice "corregido", gana). `BowelLarge`/
    `BowelLargePRV01`: solo el PRV (margen de SBRT) saca al paciente
    (`PT_c1ba3262f2e7e342`, ex-`reviewed_ok`); `BowelLarge` sin PRV queda abierto (no
    fue parte de la instrucción de hoy). `PTV_High100%`/`PTV_High95%`/`PTV_High69`/
    `PTV_High70`/`PTV_High_mod`, `RectoenPTV`/`PRVRECTOENPTV`/`PTVSINRECTO`/`Bladder1`
    → `exclude_names` (desestimar estructura, paciente se mantiene).
45. **`Contraste`/`PTV_Low!High` resultaron ser de Pelvis, no CyC** (verificado en
    datos reales — la instrucción del usuario no especificaba sitio). `Contraste` →
    `ignore_patterns` (`*contraste*`, mismo tratamiento que `Cal`/`Caliente`);
    `PTV_Low!High` → `exclude_names`. El único paciente con `PTV_Low!High` ya estaba
    `out_of_convention` por campo extendido.
46. **Nuevo artefacto reusable:** `results/auditorias/paso2_estructuras_por_paciente.csv`
    (16918 filas: site, patient_id, raw_name — los 503 pacientes). Evita releer DICOM
    cada vez que surge una decisión de armonización nueva; solo lectura de RTSTRUCT
    (sin dosis), generado una vez. El volumen rasterizado (más caro, necesita grilla de
    dosis) se sigue calculando aparte, bajo demanda, solo cuando una decisión puntual
    depende de él (ej. ítem 42 arriba).

**Resultados tras esta ronda:** CyC 175/196→**187/196 (95.4%)**; Pelvis 147/161→
**151/161 (93.8%)**; Prostata 134/146→**145/146 (99.3%)**. Total 456/503→
**483/503 (96.0%)**.

## Resueltas por el usuario (2026-10-14)

- ~~Subpoblación en Pelvis con anatomía no-pélvica~~ → desglosada caso por caso
  (decisiones #29-35, #41): mayoría resuelta vía tabla, 13/49 confirmados campo
  extendido real por tamaño de campo (Y>300mm), 7/49 por Ovary → `out_of_convention`.
- ~~`Lobe_Temporal_L/R` + PRV en CyC~~ → rol canónico `Lobe_Temporal_PRV3_L/R`,
  Dmax≤70Gy prioridad 3 (decisión #25).
- ~~Nombres sin resolver en CyC (Trachea, Retina/Cornea, PRV con convención propia,
  PTV_C2/66/Mid_3mm, Cal/Caliente, Contraste, RectumPTV, PTV_Low!High,
  SpinalCord_PRV1/PRVx, PTV_Mid01/02, etc.)~~ → resueltos caso por caso
  (decisiones #36-40, #42-45).
- ~~Nombres sin resolver en Próstata (RectumCorregido, Colostomia, Protesis,
  PTV_ProstateBed, BowelLargePRV01, PTV_High100%/95%/69/70/_mod, RectoenPTV/
  PRVRECTOENPTV/PTVSINRECTO/Bladder1, "PTV VS OK")~~ → resueltos (decisión #44).

## Pendiente de revisión humana (NO resuelto autónomamente)

Cola corta, todos casos de 1-3 pacientes, sin patrón sistemático (no vale la pena una
regla genérica). Gracias al inventario (`paso2_estructuras_por_paciente.csv`) ya no
hace falta rescanear DICOM para decidir estos — solo el volumen rasterizado si la
decisión depende de él.

- **CyC (9 fallos):** `Hippocampus_L/R`; `Lobe_Tem_PRV3_Lb/Rb` (variante con sufijo "b",
  ¿otra revisión de contorno, mismo rol que `Lobe_Temporal_PRV3_L/R`?); `SpinalCord_Aux`;
  `Elbow`; `amigdala`/`supra` (castellano, fragmentos de nombre); `PTV_Boost`/
  `PTV_boost-04`; 3 ambigüedades REALES de recontoreo (dos contornos con volumen>0 para
  el mismo rol, no se puede elegir sin ver el plan en Eclipse) — `Cochlea_L`+`Cochlea_L1`,
  `Musc_Constrict`+`Musc_Constrict1`, `OpticChiasmPRV03`+`OpticChiasm_PRV3`.
- **Pelvis (10 fallos):** `Vulva` (3 pacientes — nueva, ¿anatomía vulvar en el campo?);
  `Stomach` (2 pacientes — ver nota abajo, mismo nombre que en Próstata); `Sigma`
  (probable alias de `Sigmoid`, no confirmado con volumen); `PelvicBones`; `PRV_Rectum`/
  `PTV_Rectum` (2 pacientes distintos); `PTV_High1`/`PTV_Uterus1` (sufijo numérico sobre
  PTV — semántico o recontoreo, no está claro); `SacralPlexus` (OAR nuevo); `PTV_LN`/
  `PTV_Lesion`; `eventracion`; `Bones`.
- **Prostata (1 fallo):** `Stomach` (`PT_ebb40c323922547a`) — mismo nombre que en
  Pelvis, candidato a una sola decisión que cubra ambos sitios (¿excluir estructura,
  o indica campo extendido como Kidney en Pelvis?).
- **`BowelLarge` sin PRV** (Prostata, sin decisión — distinto de `BowelLargePRV01` que
  sí se resolvió hoy): ¿mismo órgano que `Bowel_Small`/`Sigmoid`, o un rol nuevo?
## Duodécima ronda (resolución de los 20 pendientes, usuario, 2026-10-15)

Resuelto vía `paso2_pendientes_20_pacientes_resuelto.csv` (respuesta del usuario sobre
`paso2_pendientes_20_pacientes.csv`), caso por caso:

47. **CyC — "desestimar estructura" (paciente se mantiene):** `Hippocampus_L/R`,
    `SpinalCord_Aux` (ya resuelve via `SpinalCord_PRV05`), `Elbow`, `amigdala`/`supra`,
    `PTV_Boost`/`PTV_boost-04`, `Lobe_Tem_PRV3_Lb`/`Rb` (ya resuelve via `Lobe_Tem_PRV3_L/R`).
48. **CyC — 3 ambigüedades reales de recontoreo, resueltas por el usuario** (qué
    contorno retener, cuál desestimar): `Cochlea_L1`→excluido (retiene `Cochlea_L`);
    `Musc_Constrict1`→excluido (retiene `Musc_Constrict`); `OpticChiasmPRV03`→excluido
    (retiene `OpticChiasm_PRV3` — requirió sacar el alias `opticchiasmprv03` de
    `MANUAL_OAR_ALIASES`, porque el chequeo de alias corre antes que `exclude_names` y
    lo tapaba). Los 3 nombres excluidos son únicos en todo el dataset (verificado con
    `paso2_estructuras_por_paciente.csv`), sin riesgo de afectar otro paciente.
49. **Pelvis — "desestimar estructura":** `Sigma`, `Duodenum`, `Stomach`, `Vulva`
    (Stomach/Vulva cubren también los otros 2 pacientes con el mismo nombre),
    `SacralPlexus`, `Bones`, `PTV_Uterus1` (mismo criterio que `PTV_Uterus`), `PTV_LN`,
    `PTV_Lesion` (PTV_High ya es el target real de ese paciente).
50. **Pelvis — "quitar paciente"** (igual criterio que Ovary: el nombre NO se agrega a
    `exclude_names`, queda `unknown` a propósito para que un paciente futuro no
    revisado con el mismo nombre siga fallando ruidoso): `PT_7a3de494ae747cdb`
    (`PelvicBones`), `PT_95abfd12e84c4f1d` (`PRV_Rectum`), `PT_a067e0c9e98eb65c`
    (`PTV_Rectum`), `PT_e3cd1ac3cecd73e3` (`eventracion`) → los 4 a `out_of_convention`.
51. **`PT_c83d88c040c3a900` (Pelvis) — caso especial, nuevo mecanismo.** Único caso del
    dataset con DOS contornos reales con volumen>0 para `PTV_High` (`PTV_High` 1583cc,
    `PTV_High1` 1711cc) — confirmado por volumen antes de decidir (no se podía adivinar
    cuál era cual). Decisión del usuario: el PTV real es `PTV_High1`. No se puede
    resolver con alias/exclude_names global (`PTV_High` es el nombre canónico normal en
    300+ pacientes). Se agregó: (a) alias global `ptv_high1`→nivel `high` en
    `normalize_structure_name.py` (único paciente con ese nombre, sin riesgo); (b) nuevo
    diccionario `_PATIENT_RAW_NAME_OVERRIDES` en `pipeline.py` — la ÚNICA forma de
    "excepción por nombre, scopeada a un paciente puntual" del pipeline — que descarta
    el `PTV_High` llano SOLO para este paciente. `PTV_Uterus`/`PTV_Uterus1`/
    `PTV_LN_Pelvics`/`Vulva` de este mismo paciente resuelven via las reglas de arriba.
52. **Prostata — "desestimar estructura":** `Stomach` (mismo nombre que en Pelvis,
    cubierto por el ítem 49).

**Resultado final: CyC 195/196 (99.5%), Pelvis 161/161 (100%), Prostata 146/146
(100%). Total 502/503 (99.8%).**

**1 caso nuevo surgido al resolver los 20** (no estaba en la lista original: quedaba
tapado por `Hippocampus_L/R`, que fallaba primero): `PT_0e31af2ee9f039e5` tenía
`Oral_Cavity` + `Cavity_Oral` con volumen>0 cada uno — misma familia de ambigüedad real
de recontoreo que los ítems 48.

53. **`PT_0e31af2ee9f039e5` resuelto (usuario, 2026-10-15): "son idénticas, retener
    Oral_Cavity".** Mismo patrón que el ítem 51 (`PTV_High1`): NO se puede sacar el
    alias `cavity_oral`→`Oral_Cavity` globalmente — 5 de los 6 pacientes con
    `Cavity_Oral` en todo el dataset (verificado con
    `paso2_estructuras_por_paciente.csv`) dependen de ese alias como ÚNICA fuente de
    Oral_Cavity (no tienen el nombre llano). Se agregó una segunda entrada a
    `_PATIENT_RAW_NAME_OVERRIDES` (pipeline.py) que descarta `Cavity_Oral` SOLO para
    `PT_0e31af2ee9f039e5`. Como el cambio es solo código (no toca ningún YAML de
    armonización), el `harmonization_hash` no cambió — solo hizo falta reprocesar CyC,
    Pelvis/Prostata siguieron válidos tal cual.

**Resultado final: 503/503 (100%) en los 3 sitios.**

- **Dos pacientes CyC con convención de PRV propia** (`PT_559aefbb26262e5c`,
  `PT_8a9d69ea4ef84f91`, ya resueltos vía alias) conservan estructuras en castellano sin
  decisión: `Lips`, `TemporalLobe_L/R` (ya cubierto por la regla de Lobe_Temporal),
  `Parotida_L-0.2`/`Parotida_R-0.2` ya excluidas — sin impacto en el resultado, solo
  quedan como nota para revisar en Eclipse si se quiere exprimir el último detalle.
