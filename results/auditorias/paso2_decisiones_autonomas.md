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

## Pendiente de revisión humana (NO resuelto autónomamente)

- **Subpoblación en Pelvis con anatomía no-pélvica.** Al correr el preprocesador
  completo aparecieron pacientes "PelvisGin" con `Kidney_L/R`, `Liver`, `Stomach`,
  `Colon`, `Ureter_L/R`, `GreatVessels`, y nombres de PTV tipo `PTV1_CHHiP` /
  `GTV_CHHiP` (protocolo de hipofraccionamiento prostático inglés, no ginecológico).
  Esto es MUY distinto del protocolo pélvico ginecológico de `sites_pelvis.yaml`.
  No decidí nada acá — puede ser (a) campo extendido/retroperitoneal real (mismo
  fenómeno que los 2 `out_of_convention` ya conocidos, solo que con más nombres
  nuevos), o (b) pacientes de otro sitio mezclados en el export por error. Necesito
  tu criterio clínico antes de tocar `harmonization_schema.yaml` por esto — ver
  reporte completo cuando termine la corrida. Nota de escala: en la corrida sobre
  los 161, falló una fracción grande de Pelvis por `PTV_LN_Pelvics`/`PTV_Uterus`/
  `Kidney_*`/`Liver`/etc. — no son 2-3 casos sueltos, es una porción considerable del
  sitio. Separé lo que es claramente "subcomponente de PTV" (excluido arriba, #8) de
  lo que es anatomía de OAR nueva (`Kidney_L/R`, `Liver`, `Stomach`, `SpinalCanal`,
  `SpinalCanal_PRV`, `GreatVessels`, etc.) — esto último NO lo excluí ni lo promoví a
  canónico: si estos pacientes reciben dosis renal/hepática real de forma rutinaria,
  excluir esos OAR perdería información dosimétrica relevante justo para el
  subgrupo que más la necesita.

- **`Lobe_Temporal_L/R` + `Lobe_Tem_PRV3_L/R` en CyC, muy frecuente** (decenas de
  pacientes, patrón idéntico al de Brainstem/SpinalCord: órgano llano + su PRV
  coexistiendo). A diferencia de esos casos, acá **ninguno de los dos está en el
  vocabulario canónico actual** — el template `zPabloCyC.txt` no incluía este
  constraint. No agregué el rol porque decidir "el lóbulo temporal entra como OAR
  modelado, con PRV3 como geometría" es una decisión de protocolo (como las de
  CHARTER §5), no una corrección de nombre. Si confirmás que es un OAR real del
  protocolo CyC, lo agrego como rol canónico igual que los demás seriales-con-PRV.

- **Nombres sin resolver, menor frecuencia, sin criterio claro todavía** (no son
  errores de tipeo obvios, prefiero preguntarte en vez de adivinar): `Cal_Low`/
  `Cal_Mid`/`cal`/`High_Cal`/`Low_Cal`/`Mid_Cal` y `Caliente`/`Caliente 2` (¿helpers
  de cálculo/hotspot, ignorables?); `Contraste`; `PTV_Mid01`/`PTV_Mid02` (¿varios
  Mid por paciente?); `PTV_Boost`/`PTV_boost-04`; `PTV_C2`; `PTV_Low_total`;
  `PTV_Mid_3mm`; `Trachea`/`Pretraqueal`/`Traqueostomia` (CyC, ¿otro OAR nuevo como
  Lobe_Temporal?); `RectumPTV` (resta sin separador); `PTV_Low!High` (resta entre
  niveles, no sobre PTV genérico); `SpinalCord_PRV1`/`SpinalCord_PRVx`.

  **De la corrida de Prostata:** `BowelLarge`/`BowelSmall`/`BowelLargePRV01` (¿otro
  OAR nuevo, "intestino grueso" distinto de Sigmoid/Bowel_Small? mismo patrón que
  Lobe_Temporal — aparece también en un Pelvis CHHiP-like como `Bowel_Large`);
  `PTV_ProstateBed` (¿caso de lecho prostático post-quirúrgico, target clínicamente
  distinto de intacto?); `RectumCorregido`, `Colostomia`, `Protesis` (casos únicos);
  `PTV_High100%`/`PTV_High95%`/`PTV_High69`/`PTV_High70`/`PTV_High_mod` (subdivisiones
  de PTV por nivel de dosis o variantes puntuales, no claras); `RectoenPTV`/
  `PRVRECTOENPTV`/`PTVSINRECTO`/`Bladder1` (un paciente con convención propia, resta
  rectum-PTV en castellano sin separador reconocible); `PTV VS OK`/`pros`/`ves` (un
  paciente con nombres muy no estándar, posible caso para revisar en Eclipse
  directamente en vez de intentar resolver por nombre).

- **`PTV_Mid01`/`PTV_Mid02` (+ sus formas `-04`) en CyC — frecuente (8+ pacientes).**
  Esto es exactamente el caso que CHARTER/el prompt pide frenar explícitamente:
  "cualquier paciente con más niveles de PTV que el protocolo de su sitio". A
  propósito NO lo colapsé con la regla de duplicado numerado (#16) — acá el número
  sí es semántico (¿Mid01 y Mid02 son dos volúmenes nodales separados del mismo
  nivel de dosis? ¿se deberían unir?). Necesito tu criterio: ¿se unen en una sola
  máscara `PTV_Mid`, se modelan como out_of_convention, o es otra cosa?

- **`Trachea`/`Pretraqueal`/`Traqueostomia` en CyC** — recurrente (5+ pacientes,
  algunos con traqueostomía). Mismo tipo de decisión que `Lobe_Temporal` (#pendiente
  arriba): ¿son OAR reales del protocolo CyC que faltan en la tabla, o quedan fuera?

- **Otras estructuras nuevas vistas una sola vez (CyC):** `Retina_L/R` + `Cornea_L/R`
  (un paciente, subestructuras orbitarias finas); `Eye_R_PRV2mm` (PRV de Eye, no
  modelado hoy); `Parotida_L-0.2`/`Parotida_R-0.2` (castellano + margen 2mm, ¿mismo
  Parotid_L/R o geometría distinta?); `TemporalLobe_L/R`, `EyeBack_L`, `EyeFront_L`,
  `Lips` (mismo paciente que resuelve distinto — `TemporalLobe` es el mismo concepto
  que `Lobe_Temporal` con el orden de palabras invertido, va de la mano con esa
  decisión); `amigdala`/`supra` (fragmentos en castellano, probablemente estructuras
  nodales/amigdalinas no identificadas); `PTV_C2`, `PTV66`, `PTV_Mid_3mm`,
  `PTV_Mid-04-ext` (variantes puntuales sin patrón claro). `PTV_Highext` (mismo patrón
  "ext"). `Parotida_L-0.2`/`Parotida_R-0.2` (castellano + margen 2mm). `Eye_R_PRV2mm`.
  `Elbow` (¿codo? ¿por qué en CyC?). Dos pacientes (`PT_559aefbb26262e5c`,
  `PT_8a9d69ea4ef84f91`) usan una convención de nombres de PRV completamente distinta
  (`PRV_Brainstem05`, `PRV_CochleaL03`, etc., prefijo en vez de sufijo) y uno de ellos
  además tiene estructuras en castellano sin equivalente claro (`CuelloDer`, `CuelloIzq`,
  `Nivel6`, `SCVIZQ`, `PTV_Intermediate`) — parecen casos aislados de un planificador/
  época distinta, mejor revisarlos en Eclipse que intentar generalizar la regla para 2
  pacientes.
