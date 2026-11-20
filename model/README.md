# Trained model for the notebook

The notebook uses the native phi^4 model at L = 8, kappa = 0.27 (kernel `nhmc_resid2` with odd nets and the Z2
wrapper, T = 16 stages, 5 leapfrog substeps per stage, trained with the round-trip buffer term; `config.json`).

This directory holds the weights (`kernel_final.pt`, 640793 bytes, MD5 `9f68f751d81248fb22bf5c02271eb303`), the
configuration, the training and evaluation logs and the forward-path evaluation with 262144 paths
(`eval_N262144.json`). The same file is in the Zenodo record, DOI
[10.5281/zenodo.24163330](https://doi.org/10.5281/zenodo.24163330), archive `B1_models_phi4_native.tar.gz`, path
`B1_models_phi4_native/models/phi4_buffer/L8_k0.27_buf_s0/kernel_final.pt`.

All other phi^4 models of the paper (one per lattice size and coupling, the T-scan models, the kappa-conditional L = 8
model), the CMCD / DBS models and the DW / LJ13 kernels are in archives B1-B4 of the same record.
