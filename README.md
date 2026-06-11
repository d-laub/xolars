# xolars

**X**array + p**olars**: an [xarray](https://xarray.dev) `Dataset` paired with one
[Polars](https://pola.rs) frame per dimension, kept aligned to the Dataset's
coordinate order — including under `isel`/`sel` selection and zarr + parquet
round-trips.

## Install

```bash
uv add xolars      # or: pip install xolars
```

## Usage

```python
import numpy as np
import polars as pl
import xarray as xr
from xolars import Xolars

ds = xr.Dataset(
    {"expr": (["gene_id", "sample_id"], np.arange(12.0).reshape(3, 4))},
    coords={"gene_id": ["G1", "G2", "G3"], "sample_id": ["S1", "S2", "S3", "S4"]},
)
genes = pl.DataFrame({"gene_id": ["G1", "G2", "G3"], "chrom": ["c1", "c2", "c3"]})

xol = Xolars(ds=ds, df={"gene_id": genes})

# Selection filters the Dataset AND every per-dimension frame, together:
sub = xol.sel(gene_id=["G3", "G1"])
assert list(sub.df["gene_id"]["gene_id"]) == list(sub.ds["gene_id"].values)

# Persist to zarr (Dataset) + parquet (per-dim frames), then reopen lazily:
xol.write("mydata.xolars", mode="w")
reloaded = Xolars.open("mydata.xolars")        # frames are pl.LazyFrame
eager = reloaded.collect()                      # -> pl.DataFrame
```

`Xolars` is a frozen, generic container: `Xolars[pl.LazyFrame]` after `open`,
`Xolars[pl.DataFrame]` after `collect`. Construction validates that each frame's
dim column exactly matches the Dataset coordinate (same set, same multiplicity)
and reorders rows to the Dataset's order.

## Development

```bash
uv sync                          # create .venv with runtime + dev deps
uv run pytest -q                 # tests
uv run prek run --all-files      # ruff + pyrefly + hygiene hooks
uv run prek install              # enable git hooks locally
```

## Releasing

Releases run via the manual `Release` GitHub Actions workflow
(`workflow_dispatch`), which uses [commitizen](https://commitizen-tools.github.io/commitizen/)
to bump the version from conventional commits, tag, create a GitHub release, and
publish to PyPI via OIDC trusted publishing. Before the workflow can succeed,
the following one-time setup is required on GitHub (out of scope for the initial
extraction):

1. Push this repository to `https://github.com/d-laub/xolars`.
2. Add a `GH_ACTIONS` repository secret (a PAT able to push to `main`) so the
   bump commit + tag can be pushed past branch protection.
3. Configure a `pypi` environment and enable
   [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) for the
   `xolars` project pointing at the `publish` job.
