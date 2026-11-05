# MedNeXt v1 (vendorizado)

Código de arquitectura copiado de https://github.com/MIC-DKFZ/MedNeXt (rama `main`, descargado 2026-11-01),
`nnunet_mednext/network_architecture/mednextv1/{MedNextV1.py, blocks.py, create_mednext_v1.py}`.
Licencia: Apache-2.0 (ver `LICENSE`, copia literal del repo original). Copyright de los autores originales (MIC-DKFZ / DKFZ).

Referencia: Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation", MICCAI 2023.

Modificaciones respecto del original (marcadas con `VENDOR-MOD`): ÚNICAMENTE los dos `import` que apuntaban a
`nnunet_mednext.*` ahora apuntan a `src.models.mednext_v1.*`. No se instaló el paquete (arrastra nnU-Net v1).
No se vendorizó nada más (sin trainers, sin nnU-Net).
