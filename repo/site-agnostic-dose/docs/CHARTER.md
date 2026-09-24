# Charter — Site-Agnostic Dose Prediction (KBP)

> **Documento Tier 1 (estable).** Vive fijo en el Claude Project. Cambia solo ante
> decisiones estructurales. Si algo de acá hay que cambiar, que chat o Claude Code lo
> señalen explícitamente y se actualice este archivo — no se corrige "de facto" en otros lados.
>
> Última actualización: Fase 0 — convención de normalización multi-sitio cerrada (D95(PTV_High)=100%).

---

## 1. Objetivo macro

Entrenar una **red site-agnostic única** que prediga una **distribución de dosis 3D certera
en PTVs y OARs** a partir de la anatomía + información de prescripción/objetivos, para dos usos:

1. **Visualización médica** (mostrar dosis alcanzable pre-planificación).
2. **RapidPlan casero** (target de referencia para setear objetivos de optimización).

**Prioridad de fidelidad:** siempre PTV y OARs. La dosis en BODY / zonas intermedias-bajas
fuera de OARs se evalúa y monitorea, pero no es el norte.

**Fuera de alcance de este proyecto:** conversión a plan entregable (PDRT / RTPLAN). Queda para
un trabajo posterior; el proyecto termina en "dosis 3D certera". El stack PDRT del proyecto de
próstata se hereda como contexto en pausa, no como objetivo activo (ver §9).

### Alcance final vs. primera etapa

- **Meta final:** un site-agnostic único para todas las patologías, normo + hipo + SBRT + RC,
  VMAT + IMRT + 3DCRT.
- **Primera etapa (este proyecto):** tres sitios anatómicamente muy distintos —
  **próstata, pelvis ginecológica, cabeza y cuello (CyC)** — arrancando **solo VMAT**
  (ver §4, decisión de arranque). IMRT, mama y el resto entran por fases posteriores.

---

## 2. Métricas de éxito

- **Principal:** fidelidad de **DVH en PTV y OARs** (curva completa, no solo 3 puntos de
  constraint). Heredar la métrica `dvh_curva_completa` del proyecto de próstata (mean|ΔV(D)|
  por banda de dosis; la banda media 40–80% Rx es la crítica para OARs).
- **Secundarias / monitoreo:** MAE body, dose_score OpenKBP, y una métrica de **coherencia
  espacial** tipo SSIM/MS-SSIM (relevante para la visualización médica, no solo la DVH).
- **Reporte siempre por sitio** (y por técnica cuando entre IMRT). El MAE global promedia y
  esconde — un canal puede mejorar un sitio y empeorar otro. IC95 por bootstrap por sitio.
- **Control metodológico — no aplica como en próstata, y por diseño.** En próstata el clásico
  ganaba porque el target era clasificar unos pocos constraints **conocidos** con features
  geométricos **conocidos** (feature engineering acotado). Acá el target es la **curva DVH
  completa** para un **número indeterminado de OARs por sitio** — no hay un feature set fijo que
  generalice entre próstata/pelvis/CyC sin terminar armando un modelo clásico por sitio (lo que
  ya no sería un control site-agnostic, serían N controles site-specific). No se monta ML clásico
  como control permanente en este proyecto (ver aprendizaje transversal §7).
  **Control real que sí aplica:** una baseline **no-aprendida** — DVH promedio
  poblacional/de protocolo por `(sitio, rol de estructura)` — que funciona igual con cualquier
  cantidad de OARs (una curva por rol, no un modelo por paciente) y responde la pregunta que
  importa: ¿la red aprende algo más que "la forma típica del protocolo"? Análogo a la baseline
  trivial de nnU-Net (predecir la media).

### N razonable (supuesto de trabajo)

- Piso operativo: **100 pacientes/sitio**. Apuntar a **200 donde sea recuperable**.
- Pool combinado ~300+ es del orden de nnDoseNet single-site (250) y muy por encima del
  fine-tuning de H&N de Mashayekhi (43) → razonable para un modelo combinado.
- Split estratificado por sitio, 65/15/20 aprox. El test por sitio (~20) es delgado pero
  funcional (hipo fue 31); reportar con IC bootstrap.

---

## 3. Representación de input — filosofía

Tres cosas conceptualmente distintas que NO hay que confundir (error de origen: Mashayekhi y
Xiong codifican el canal OAR de formas distintas):

1. **Geometría** — dónde está cada estructura (máscaras / distancia).
2. **Intención** — qué DVH se quiere (goals de DVH). *Mashayekhi:* DVH deseada mapeada por
   ranking de distancia al PTV.
3. **Prioridad** — cuánto importa cada objetivo / quién gana los conflictos. *Xiong:* peso
   `W = 2^(1−P)` en el canal OAR, máximo en vóxeles de solapamiento.

**Decisión:** arrancar con el canal de **goals de DVH** (intención). La **prioridad** entra
como canal separado *después* (ablación), no fundida de entrada — así se puede medir su aporte
aislado. Fusión (`goal × W`) queda como variante posterior. Justificación: en planes sencillos
la prioridad es irrelevante; su efecto solo aparece en casos en tensión → es una ablación, no
un requisito.

**Prioridad — estructura de datos:** la prioridad está atada a la **métrica**, no al OAR
(usualmente 1 OAR = 1 métrica, pero no siempre). Es **por protocolo/sitio** (estática dentro
del sitio) → en esta etapa es un *descriptor de sitio*, no una perilla de control. Los casos
que la mueven por paciente (reirradiación) se excluyen ahora. La tabla de armonización se
modela como `sitio → {(OAR, métrica): (goal, prioridad)}` desde el inicio, con el caso
1-a-muchos contemplado.

**Sub-problema abierto (Fase 2):** los goals son constraints discretos (pocos puntos por OAR),
no curvas DVH completas como las que tenía Mashayekhi. Mapear "V65<15%" a un canal espacial
requiere construir una pseudo-DVH monótona desde los puntos, o elegir otra codificación.

---

## 4. Mapa de fases

> Detalle de cada fase lo cierra su propio chat (ver §8, flujo). Acá va el grueso.

- **Fase 0 — Infraestructura y datos.**
  - 0.0 Profiling de la PC (envelope de parche × batch × features × 2D/3D en RTX A2000 12GB,
    con/sin gradient checkpointing, a resolución de grilla de dosis).
  - 0.1 Preprocesador nuevo y limpio (NO fork del de próstata) + tabla de armonización
    (nomenclatura TG-263, estructura `sitio→(OAR,métrica)→(goal,prioridad)`).
  - 0.2 Auditorías: **contaminación sobre la dosis** (lección nodal), escala de vóxel, y
    **conteo de versiones de plan por paciente en ARIA** (decide viabilidad del Pareto futuro).
    Baseline no-aprendida (DVH promedio por sitio/rol de estructura) montada en paralelo.

- **Fase 1 — Reproducción del baseline.** Portar exp002 (U-Net 2D + PSDM, MAE) al framework
  nuevo sobre próstata VMAT y confirmar que reproduce los números. Des-riesga el refactor.

- **Fase 2 — Primer site-agnostic (A-eje: input). SOLO VMAT.** Próstata + Pelvis + CyC.
  Ablaciones de input incrementales, MAE puro, U-Net fija, métrica por sitio:
  - A0: PTV con Rx + OAR (goals de DVH) + body. Piso.
  - A1: + prioridad (canal separado; `W=2^(1−P)`, máximo en solapamiento).
  - A2: + PTV distance map / PSDM. **Test del hallazgo de próstata en régimen multi-sitio**
    (en Xiong el distance map NO aportó — puede volverse redundante).
  - A3: + densidad másica (NO el CT crudo). Test de si "el CT no aporta" era en realidad
    "el CT crudo no aporta".
  - A4: + beam plate map (templado, sintetizado del protocolo — no del RTPLAN).

- **Fase 3 — Arquitectura (B-eje).** Con la mejor representación de A fija:
  U-Net 2D → U-Net 3D con parches → MedNeXt (kernel 3→5, UpKern; código público de Xiong).
  DropBlock / CCReLU como ablaciones de regularización opcionales, prioridad baja.

- **Fase 4 — Loss.** MAE es baseline. Probar DVH loss por tipo de estructura (serie/paralelo,
  estilo Xiong/Jhanwar) y SSIM para coherencia espacial. (Nota: en próstata MAE puro ganó y
  las losses DVH no aportaron — empezar por MAE y testear el resto contra esa vara.)

- **Fase de incorporación de IMRT (propia).** Al mezclar IMRT + VMAT del mismo sitio, la
  **desambiguación de técnica** pasa a ser pregunta de primera clase, medible contra el
  baseline VMAT-only limpio. Decidir ahí: token categórico de técnica (estilo class-label de
  ADDiff-Dose) vs. desambiguación vía beam plate (Xiong no usa token — el beam plate de un IMRT
  de 7 campos y un VMAT en arco son espacialmente muy distintos). **Mama entra en esta línea o
  junto al beam plate**, no antes (ver decisión de arranque).

- **Fase 5+ (exploratoria) — conditioning por prioridad/preferencia (Pareto).** Solo si la
  auditoría de Fase 0 dio cobertura de manifold suficiente. Acá se abre el desacople
  intención/prioridad y el interés de variar goals/prioridades para inferir planes que prioricen
  PTV vs OAR (generador tipo Pareto).

### Decisión de arranque: VMAT-only (registrada)

Se arranca **solo VMAT** en Fase 2 en vez de mezclar IMRT+VMAT con token desde A0. Razón: el
token de técnica no es una ablación (no hay "¿ayuda?" — sin él el mapeo input→dosis es
uno-a-muchos y el modelo promedia). VMAT-only **elimina el confounder por construcción** (misma
filosofía que sacar los 28 pacientes nodales en vez de modelarlos), evita el riesgo de que un
token escalar sea subusado y deje promediado residual contaminando todas las ablaciones de A.
Confirmado que VMAT cubre los 3 sitios con N>100 cada uno. IMRT entra como fase propia, donde
la desambiguación es medible contra el baseline limpio. Esto disuelve la vieja pregunta de
ordenar A4/A5.

**Rama alternativa documentada (por si algún sitio quedara flaco sin IMRT):** arrancar mixto
con desambiguación adentro desde A0. No es el caso actual (los 3 sitios tienen N>100 en VMAT).

---

## 5. Decisiones de diseño congeladas (NO cambiar sin consultar)

Heredadas del proyecto de próstata y confirmadas:

- Sampling por paciente (no por corte suelto).
- GroupNorm (no BatchNorm) con batch chico.
- Bilinear upsample + conv 3×3 (no transposed conv) — revisable en B-eje.
- Factor de escala de épocas por **pasos de gradiente**, no épocas:
  `factor = train_A / train_B` (con batch=1). Aplicar cada vez que cambie el N de train.
- `scheduler.max_epochs` fijo al horizonte completo, independiente de `trainer.max_epochs`,
  para que cualquier corte temprano (sweeps) sea comparable en LR.
- Dosis normalizada (D95(PTV)=100% en próstata; convención multi-sitio D95(PTV_High)=100%, cerrada — ver "Nuevas" abajo).

Nuevas de este proyecto:

- **Repo y proyecto nuevos**, no fork del de próstata (evita heredar bugs por copy-paste).
- **Reporte por sitio (y por técnica) siempre**, con IC bootstrap.
- **Baseline no-aprendida (DVH promedio por sitio/rol de estructura) en paralelo** como control
  permanente — no ML clásico (ver §2 y §7: no aplica al target de este proyecto).
- **VMAT-only** en Fase 2 (ver §4).
- **Normalización de dosis multi-sitio: D95(PTV_High)=100%.** Re-normalizada desde la grilla
  de dosis cruda (ignorando la normalización de Eclipse), con D95 calculado sobre **volumen
  nativo**, no la grilla downsampleada (instancia de la higiene de vóxel del §7). PTV_High = PTV
  de mayor Rx; nomenclatura TG-263: `PTV` o `PTV_High` en single, `PTV_High`+`PTV_Low` en dos
  niveles, `+PTV_Mid` en tres (típico CyC), con una capa de alias/excepción para nombres fuera
  de estándar. El **canal de intención** codifica cada PTV como `Rx_nivel/Rx_High` con ratios
  **prescriptos del protocolo**, NO leídos de la dosis (sería leakage: se estaría alimentando el
  target al input) — la tabla de armonización es la fuente de verdad, el RTPLAN el chequeo.
  `Rx_High` (Gy) se guarda como metadata por paciente para des-normalizar a Gy en la
  visualización médica (§1). La banda "% Rx" de la métrica DVH (§2) se define como % de Rx_High.
- Beam plate para inferencia se **sintetiza del template protocolar** (isocentro ≈ centroide
  PTV, ángulos/arcos del protocolo), no se toma del RTPLAN (no existe pre-plan). Al comisionar
  esta fase, medir la distancia beam-map real-vs-templado por sitio para decidir si se entrena
  con templado (consistencia train/inference) o con real.

---

## 6. Esquema de repo y datos

**Principio:** git lleva solo código + configs + docs. Los datos pesados viven fuera del repo,
versionados por etapa de preprocesado (nunca pisar — cada corrida escribe a una carpeta con
nombre de versión, y el config del experimento apunta explícitamente a cuál usa).

Estructura en disco: dos carpetas **hermanas** bajo un mismo padre (no anidadas — evita que un
gitignore mal escrito arrastre datos pesados al repo, o que `git status` escanee árboles enormes):

Dos carpetas de nombre `data`, en niveles distintos, con contenido distinto — no confundir una
con otra: `repo\data\` (chica, va a git) vive DENTRO del repo; `SiteAgnostic\data\` (pesada, no
va a git) es HERMANA del repo. Árbol completo:

```
C:\Pablo\SiteAgnostic\
│
├── repo\                              # ============ TODO ESTO VA A GIT ============
│   └── site-agnostic-dose\
│       ├── configs\            # un yaml por experimento; apunta a la versión de datos que usa
│       ├── src\
│       │   ├── preprocess\     # preprocesador nuevo, limpio (sin reimplementar por-sitio)
│       │   ├── datamodules\
│       │   ├── models\
│       │   ├── losses\
│       │   └── bridge\         # dosis→grid físico (heredar unet_to_target del piloto RD)
│       ├── scripts\            # train, evaluate, profile, auditorías, baseline_dvh_promedio
│       ├── commissioning\      # PARADO — herencia PDRT, no activo en este proyecto
│       ├── docs\
│       │   ├── CHARTER.md              # este archivo (Tier 1)
│       │   ├── contexto_fase_vigente.md    # Tier 2, doc vivo de la fase actual
│       │   └── aprendizajes_transversales.md   # Tier 3, append-only curado
│       └── data\                # chica: SOLO splits/ y harmonization/, va a git
│           ├── splits\          # json de splits — definen la reproducibilidad de cada experim.
│           └── harmonization\   # tabla sitio→(OAR,métrica)→(goal,prioridad)
│
└── data\                              # ======= TODO ESTO NO VA A GIT (pesado) =======
    ├── dicom_raw\<sitio>\<paciente>\      # DICOM crudos, exportados desde Eclipse
    ├── dicom_extracted\<sitio>\           # CSV de métricas + estructuras (C#/pydicom)
    ├── npz_v<N>_<etiqueta>\<sitio>\       # NPZ preprocesados, VERSIONADOS por pipeline
    ├── checkpoints\<exp_id>\
    ├── predictions\<exp_id>\              # dosis predichas para evaluación/piloto
    └── results\<exp_id>\                  # metrics.csv, summary.json, análisis, plots
```

**Van a git aunque parezcan datos:** splits y tabla de armonización (fuente de verdad de
nombres/goals/prioridades — código disfrazado de datos), ambos en `repo\...\data\`. Todo lo
pesado vive en la carpeta hermana `SiteAgnostic\data\` y queda afuera de git.

---

## 7. Aprendizajes transversales heredados (del proyecto de próstata)

Los que aplican directo a este proyecto (el detalle vive en `aprendizajes_transversales.md`):

- **El baseline de ML clásico le ganó a la U-Net en próstata, pero para clasificar constraints
  conocidos con features conocidos** — no transfiere directo acá (target = DVH completa, OARs
  variables por sitio). Control reemplazado por una baseline no-aprendida (§2). El ML clásico
  sigue siendo la herramienta correcta para los proyectos derivados de §9 (mama, DIBH), que sí
  son clasificación/regresión con features geométricos acotados y conocidos.
- **La contaminación vive en la dosis, no en la geometría de las máscaras.** Los chequeos de
  volumen/overlap/MU no la detectan. Precedente: 28 pacientes nodales generaron el "techo de
  arcos" espurio que sobrevivió 3 corridas. La auditoría de Fase 0 debe mirar la **dosis**.
- **Bugs de datos que reaparecen por copy-paste:** escala de vóxel (`spacing_mm` nativo aplicado
  a grilla downsampleada, factor 1.1×–4.2× variable por paciente) y carga de CT (`cargar_ct`
  cargaba la serie RD). Argumento para preprocesador nuevo y limpio.
- **MAE puro ganó en próstata; las losses basadas en DVH y momentos no aportaron.** Empezar por
  MAE, testear el resto contra esa vara.
- **3D no aportó en próstata, pero era próstata** (gradiente axial suave). Se espera que sí
  aporte en CyC/pelvis (estructura cráneo-caudal). No es contradicción — es contexto.
- **Higiene de documentación:** el estado actual vive en UN lugar (Tier 2, reescrito a la verdad
  de hoy); el *por qué cambió* va al append-only. No apilar correcciones inline (el doc viejo
  llegó a 2556 líneas con hallazgos revisados 3 veces).

---

## 8. Flujo de trabajo — Proyecto / chat / Claude Code

Tres tiers separados por ciclo de vida:

- **Tier 1 — Charter (este doc).** En el Claude Project, fijo. Objetivos, fases, decisiones
  congeladas, esquema de repo, este flujo. Cambia rara vez.
- **Tier 2 — Contexto de fase vigente (`contexto_fase_vigente.md`).** Uno por fase. Lo que
  Claude Code lee y escribe durante la fase. Enfocado SOLO en la fase actual; **nunca acumula
  a través de fases**. Refleja siempre "qué es verdad ahora", no la traza.
- **Tier 3 — Fases cerradas + aprendizajes (`aprendizajes_transversales.md` y archivos de fase
  congelados).** Append-only, curado. Memoria institucional en una línea con puntero por ítem.

**Handoff chat ↔ Code:** el contexto de fase vive como archivo (repo / doc del Project). NO se
pega entero en el chat. Al volver de Code al chat, se trae solo el **delta** (metrics.csv +
summary.json + resumen corto), no el doc. (Este patrón funcionó en próstata y evita el consumo
innecesario de tokens.)

**Un chat = una fase.** Arranca leyendo el charter + el MD de arranque que dejó el chat anterior.
Al cerrar la fase, el chat genera dos MD: (1) uno que congela la fase (a Tier 3 del Project) y
(2) uno de arranque para el chat siguiente. Los aprendizajes transversales se appendean, curados.

**Reabrir fases:** las fases no siempre son secuenciales limpias (un bug de armonización de
Fase 0 puede aparecer en Fase 2). Los archivos de Tier 3 son legibles y reabribles, no sellados.

---

## 9. Proyectos derivados / en pausa

- **PDRT → preplan entregable.** Todo el stack de dosis→RTPLAN vía PyDoseRT (motor comisionado
  6X, mimicking voxel-wise, writer pydicom). En pausa. Contexto, no objetivo activo de este
  proyecto.
- **ML clásico para elegir tangenciales de mama.** Modelo geométrico de preprocesamiento, aguas
  arriba de la red de dosis. Métricas más geométricas/menos limpias que el overlap de próstata.
- **ML clásico dosis-corazón / DIBH mama izquierda.** Predecir dosis cardíaca en tomógrafo para
  decidir si conviene inspiración profunda, como decisión clínica pre-plan. Análogo directo al
  ML clásico de constraints de próstata. Trabajo paralelo, autocontenido.

---

## 10. Preguntas vivas / decisiones abiertas

- Codificación de goals de DVH discretos → canal espacial (Fase 2).
- Token de técnica vs. desambiguación vía beam plate (fase de IMRT).
- Viabilidad del Pareto-conditioning (depende de la auditoría multi-plan de Fase 0).
- Estrategia de parches: ROI-crop agresivo (nnDoseNet) + parches grandes para preservar
  contexto global (la dosis es no-local) — evitar parches chicos sobre volumen full-body.
