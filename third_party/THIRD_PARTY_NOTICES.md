# Third-party notices

The preliminary bottle baseline uses these installed dependencies and assets:

- PyTorch 2.6.0, BSD-3-Clause (`torch-2.6.0.dist-info/LICENSE`).
- torchvision 0.21.0, BSD-3-Clause (`torchvision-0.21.0.dist-info/LICENSE`).
- torchvision `Wide_ResNet50_2_Weights.IMAGENET1K_V2`, downloaded from the official PyTorch
  model URL. SHA-256:
  `9ba9bcbebc349d733a72eb7608143fd754e4689ac8d8ce2916ca0bdff6443950`.
- scikit-learn 1.6.1, BSD-3-Clause, used for the preliminary linear probe.
- Requests 2.32.3, Apache-2.0, used by the pinned community-mirror downloader.

The pretrained weight is not committed. Its redistribution and ImageNet-derived data-use terms
still require human review before product packaging. The PatchCore-lite implementation in this
repository is a new minimal baseline implementation; it is not copied official PatchCore code and
must not be described as the official PatchCore reproduction.

The formal PatchCore baseline uses an unmodified nested checkout of Amazon Science
`patchcore-inspection` at commit `fcaa92f124fb1ad74a7acf56726decd4b27cbcad` under Apache-2.0.

The EfficientAD-S/RCBR candidate uses an unmodified nested checkout of the official
Open Edge Platform `anomalib` repository, tag `v2.3.0`, commit
`091ca6aca92c8d0e416394f79e52f5a3cea3db73`, under Apache-2.0. Its upstream EfficientAD
teacher archive and Imagenette download are not committed. Expected SHA-256 values are:

- EfficientAD pretrained teacher archive:
  `c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a`;
- Imagenette archive:
  `6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd`.

These assets are obtained by the pinned upstream download helper. Redistribution and upstream
dataset/weight terms still require human review before submission packaging.
