# Appends de cierre de Fase 3

> Dos bloques para que Code los agregue al final de cada archivo, sin tocar el resto.

---

## Bloque 1 → `docs/aprendizajes_transversales.md`

### Fase 3 (eje B, arquitectura) → detalle en `fase3_cerrada.md`

- **Una métrica también puede estar mal: validar qué mide antes de decidir con ella.** El SSIM de
  frame completo medía el aire no supervisado (71% de los píxeles en CyC) y sostenía una decisión
  del eje A (goals). Toda métrica se restringe a la región supervisada (BODY).
- **La red predice cualquier cosa donde no hay supervisión.** El 2D ponía ~10% Rx de error en el aire.
  Post-proceso estándar: dosis × BODY.
- **En VMAT coplanar, un 2D con PSDM se comporta como 2.5D.** El 3D no mejoró el DVH en próstata ni en
  multi-sitio; su ganancia aparece en coherencia y MAE voxel a voxel en sitios largos (CyC). La
  literatura rankea por métricas voxel (dose score de OpenKBP), donde el 3D sí gana: no es una
  contradicción, se mide otra cosa.
- **Firma de piso:** si arquitecturas distintas fallan en los mismos pacientes (errores correlacionados,
  diferencia entre modelos << dispersión entre pacientes), el cuello de botella no es la arquitectura.
- **Efectos opuestos en sitios vecinos con una sola seed → sospechar varianza entre corridas.** El IC
  bootstrap por paciente no la incluye.
- **Perfilar memoria con la forma real que entrega el datamodule.** El envelope medía 176×128 y el
  datamodule padeaba a 256²: el supuesto "entra sin ckpt" era falso.
- **El padding de aire es memoria tirada:** crop al BODY + contenedor fijo ajustado antes de pensar en
  parches.

---

## Bloque 2 → `docs/cabos_sueltos.md`

Además: marcar como **RESUELTA** la entrada "SSIM enmascarado por BODY (des-diluir el aire)", con
puntero a `fase3_cerrada.md` (se resolvió en Fase 3 y el SSIM de frame completo quedó retirado).
Actualizar la entrada de densidad: la condición de reapertura "re-testear en 3D" pasa a ser "si el 3D
pasa a arquitectura de operación".

### Piso de seed en PTV-alta (Fase 3)
- **Qué es:** el 3D da Pelvis +1.75 / Próstata −1.63 en PTV-alta, con IC que excluyen 0, pero con una
  sola seed. Medir seed 43 del 2D para conocer la varianza entre corridas.
- **Por qué se difirió:** sea real o ruido, la decisión es la misma (el 3D no gana en general).
- **Qué lo reabriría:** que una comparación de Fase 4 caiga en zona gris, o querer atribuir esos
  efectos.

### 3D como trade-off de coherencia en CyC (Fase 3)
- **Qué es:** el 3D mejora el SSIM dentro del BODY (+0.03/+0.04) y el MAE-body (1.72 vs 2.32) en
  CyC, con DVH empatado.
- **Por qué se difirió:** la prioridad es el DVH, y el 3D es caro y está al límite de memoria.
- **Qué lo reabriría:** que la visualización médica (§1, uso 1) pase a ser el objetivo principal, o que
  una loss de Fase 4 cambie el ranking en DVH.

### Variantes 3D no probadas (Fase 3)
- **Qué es:** pooling anisótropo en Z, más capacidad, MedNeXt-3D, UpKern k5.
- **Por qué se difirió:** firma de piso en OARs; ni el 3D vanilla ni MedNeXt-2D ganaron en DVH.
- **Qué lo reabriría:** evidencia de error atribuible a la arquitectura (por ejemplo, el piso de seed
  muestra que la arquitectura mueve PTV-alta).

### Loss × arquitectura (Fase 3)
- **Qué es:** el veredicto 2D ≥ 3D vale con MAE.
- **Qué lo reabriría:** que la Fase 4 adopte una loss no-MAE → una corrida 3D con esa loss.

### PSDM como 2.5D (Fase 3)
- **Qué es:** 3D vs 2D sin PSDM (input A1), para atribuir por qué el 3D no aporta.
- **Por qué se difirió:** explica, pero no cambia la decisión operativa.
- **Qué lo reabriría:** querer publicar "el 3D no aporta" y atribuirlo.

### A4 sin goals (Fase 3)
- **Qué es:** el efecto de los goals se midió sin PSDM (A0 vs A1); el costo de ~0.9 pp en PTV-alta de
  CyC no se conoce con PSDM.
- **Por qué se difirió:** los goals se mantienen por diseño (canal de intención); la corrida no cambiaría
  esa decisión.
- **Qué lo reabriría:** que PTV-alta de CyC pase a ser el cuello de botella.

### Piso de etiqueta definitivo (Fase 3)
- **Qué es:** medir la variabilidad entre planificadores con pacientes que tengan más de una versión de
  plan (auditoría de Fase 0).
- **Qué lo reabriría:** querer cuantificar el techo alcanzable (Fase 4, publicación).

### Técnicas no coplanares (Fase 3)
- **Qué es:** "el 3D no aporta" es un hallazgo de VMAT coplanar. Con arcos no coplanares (SBRT, RC)
  deja de valer la separabilidad por corte.
- **Qué lo reabriría:** la incorporación de esas técnicas → re-evaluar el 3D desde cero.
