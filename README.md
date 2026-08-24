# EvoInspect-130

This repository is in phase G0. It currently contains a deterministic engineering fixture for
checking data isolation and evidence flow; it does **not** yet contain a reproduced research
baseline or any competition-valid metric.

## Verified vertical slice

The implemented path is:

```text
data validate -> data split -> adapt product -> infer image -> evaluate -> report generate
```

Run it without downloading data or using a GPU:

```bash
EVOINSPECT_RUN_ID=fixture-local ./scripts/run_fixture_smoke.sh
```

The default interpreter is `/home/CuiMinghao/apps/miniforge3/bin/python`; override it with
`EVOINSPECT_PYTHON`. Outputs go to `reports/experiments/<run-id>/`. Fixture metrics are marked
`engineering_test_only` and are forbidden in research, deployment or competition claims.

Run the dependency-free unit suite:

```bash
PYTHONPATH=src /home/CuiMinghao/apps/miniforge3/bin/python -m unittest discover -s tests -v
```

The isolated development environment is specified in `environment.yml`; it has not yet been
created. GPU work remains blocked until `nvidia-smi` can enumerate every card and unrelated
compute process, after which a job may use only a confirmed idle GPU through
`CUDA_VISIBLE_DEVICES`.

After a dataset archive is obtained through its official form, inspect it before extraction:

```bash
PYTHONPATH=src /home/CuiMinghao/apps/miniforge3/bin/python -m evoinspect.cli data inspect-archive \
  --archive /external/read-only/path/dataset.zip \
  --expected-sha256 <trusted-or-recorded-sha256> \
  --dataset-id MVTec_AD \
  --license-id CC-BY-NC-SA-4.0 \
  --output evidence/dataset_receipts/mvtec_ad.json
```

This command never extracts the archive. It rejects hash mismatches, path traversal and link
members, but human license signoff is still required.
