# Arranque — Fase 4: loss

> Para abrir el chat de Fase 4. Leer con `CHARTER.md` (§4 Fase 4, §5, §11), `fase3_cerrada.md` y
> `aprendizajes_transversales.md`. Un chat = una fase. Reescribir `contexto_fase_vigente.md` a Fase 4 al
> arrancar. Handoff chat↔Code por delta. El chat redacta las indicaciones y Pablo se las pasa a Code.

## Qué es la Fase 4

Con input (A4) y arquitectura (U-Net 2D) fijos, se varía la **loss**. MAE (`masked_mae_body`) es la
vara. Candidatas del charter: DVH loss por tipo de estructura (serie/paralelo, estilo Xiong/Jhanwar) y
una loss de coherencia espacial. En próstata, MAE puro ganó y las losses DVH no aportaron: cada
candidata se testea contra MAE, de a una.

## Configuración fija

- Input A4 (7 canales), `npz_v2b_body`, `split_v2.json`, hash `7ab16f73fa632a3e`.
- U-Net 2D base16/depth4, `a0_module`, contenedor (176,256) crop-to-BODY.
- Dosis predicha × máscara BODY en inferencia/eval.
- AdamW 1e-4/1e-5, cosine 189, patience 28, 16-mixed, seed 42, batch=1 por paciente.

## Vara a batir: `exp_a4_003` ep113 (val)

**Primer paso de Fase 4:** re-evaluar `exp_a4_003` con el harness vigente (pred × BODY, `ssim_body`)
y tomar ese resultado como vara oficial. Solo evaluación, sin entrenar. Los números de abajo son con
la predicción sin máscara; difieren en centésimas, pero vara y corridas nuevas tienen que pasar por
el mismo pipeline. `ssim_frame_obsoleto` no se usa para decidir.

- Criterio pooled OAR banda media: **2.487** (CyC 2.18 / Pelvis 3.34 / Próstata 1.97).
- OAR-media y PTV-alta por sitio, con IC: en `results/exp_a4_003.../summary.json`.
- MAE-body: 2.324 / 2.530 / 0.917.
- SSIM BODY: 0.923 / 0.960 / 0.978 · BODY∩dosis>5%: 0.889 / 0.951 / 0.949.
- d2 (xy/dosis media/vfrac): CyC 0.97/1.00/0.88 · Pelvis 0.93/0.99/0.89 · Próstata 0.89/0.93/0.87.

## Antes de entrenar

1. **Revisión clínica de la cola** (nivel 1, de Pablo en Eclipse): `PT_20011a605a19589c`,
   `PT_d41fa07c4f323437`, `PT_b55d907a5f2cc495` (CyC; el primero además tiene vóxeles de PTV_High
   fuera del BODY); `PT_1236511005be4d68`, `PT_c687aea3fc02ef4f`
   (Pelvis). Si son planes atípicos → `out_of_convention` y conjunto reservado (§5). Como son de val,
   cambiaría la vara: re-evaluar `exp_a4_003` sin ellos.
2. **Criterio de selección de checkpoint.** El early stopping por val/mae (body) está desalineado con
   la prioridad DVH. En 2D/3D coincidió con el óptimo DVH; en MedNeXt no. Una loss nueva puede
   separarlos. Decidir al arrancar: mantener val/mae como monitor y reportar siempre el criterio
   DVH sobre el top-3, o monitorear directo por el criterio DVH. Mismo criterio para todos los
   modelos.

## Advertencias que hereda

- **Loss SSIM:** solo enmascarada por BODY. El SSIM de frame completo premia el nivel de dosis en el
  aire (Fase 3). Dentro del cuerpo, las diferencias entre modelos pasan sobre todo por la luminancia
  (media local), que se superpone con lo que ya optimiza el MAE → prioridad baja.
- **Expectativas acotadas en OARs:** hay firma de piso (los errores OAR de 2D y 3D están
  correlacionados por paciente). PTV-alta es donde los modelos discrepan: ahí es más probable que una
  loss mueva algo.
- **Piso de ruido:** las deltas de loss probablemente sean chicas. Si una comparación cae en zona gris,
  medir seed 43 del baseline 2D (barato) antes de decidir. Efectos opuestos en sitios vecinos = señal
  de varianza entre corridas.
- **Regla de corte pre-fijada por experimento**, como en Fase 3: gana si mejora el pooled, ningún sitio
  empeora con IC que excluya 0 (OAR-media o PTV-alta), y al menos un sitio mejora con IC que excluya
  0. Una mejora aislada en PTV-alta de un solo sitio no alcanza sin piso de seed.

## Lectura y rigor

Por sitio, IC95 bootstrap, set completo (DVH + MAE-body + SSIM BODY + d2). Validar qué mide una
métrica antes de decidir con ella (lección SSIM). No reabrir eje A ni B, salvo por los triggers de
`cabos_sueltos.md`. Uno relevante acá: si se adopta una loss no-MAE, re-testear el 3D con esa loss
(una corrida).

## Al cerrar Fase 4

Generar el MD que la congela (Tier 3) + arranque de la fase siguiente. Appendear aprendizajes curados.
