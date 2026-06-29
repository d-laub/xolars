from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr

from xolars import Xolars


def _ds():
    """3 genes × 4 samples expression dataset."""
    return xr.Dataset(
        {"expr": (["gene_id", "sample_id"], np.arange(12.0).reshape(3, 4))},
        coords={
            "gene_id": ["ENSG001", "ENSG002", "ENSG003"],
            "sample_id": ["S1", "S2", "S3", "S4"],
        },
    )


def _gene_df(genes=None):
    genes = genes or ["ENSG001", "ENSG002", "ENSG003"]
    return pl.DataFrame(
        {
            "gene_id": genes,
            "chrom": [f"chr{i + 1}" for i in range(len(genes))],
        }
    )


def _sample_df(samples=None):
    samples = samples or ["S1", "S2", "S3", "S4"]
    return pl.DataFrame(
        {
            "sample_id": samples,
            "age": list(range(40, 40 + len(samples))),
        }
    )


def test_construction_valid():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    assert list(xol.df["gene_id"]["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]
    assert list(xol.df["sample_id"]["sample_id"]) == ["S1", "S2", "S3", "S4"]


def test_construction_reorders_rows():
    # DataFrame rows in reverse order — should be reordered to match ds
    gene_df_reversed = _gene_df(["ENSG003", "ENSG002", "ENSG001"])
    xol = Xolars(ds=_ds(), df={"gene_id": gene_df_reversed})
    assert list(xol.df["gene_id"]["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]


def test_construction_error_missing_dim():
    with pytest.raises(ValueError, match="not a dimension"):
        Xolars(ds=_ds(), df={"bad_dim": _gene_df()})


def test_construction_error_missing_column():
    bad_df = pl.DataFrame({"wrong_col": ["ENSG001", "ENSG002", "ENSG003"]})
    with pytest.raises(ValueError, match="missing column"):
        Xolars(ds=_ds(), df={"gene_id": bad_df})


def test_construction_error_wrong_values():
    bad_df = pl.DataFrame(
        {
            "gene_id": ["ENSG001", "ENSG002", "ENSG999"],
            "chrom": ["chr1", "chr2", "chr3"],
        }
    )
    with pytest.raises(ValueError, match="don't match"):
        Xolars(ds=_ds(), df={"gene_id": bad_df})


def test_construction_error_duplicate_values():
    # Same set as ds but wrong multiset (ENSG001 duplicated, ENSG003 missing)
    bad_df = pl.DataFrame(
        {
            "gene_id": ["ENSG001", "ENSG001", "ENSG002"],
            "chrom": ["chr1", "chr1", "chr2"],
        }
    )
    with pytest.raises(ValueError, match="don't match"):
        Xolars(ds=_ds(), df={"gene_id": bad_df})


def test_construction_lazy_frame():
    lazy_gene_df = _gene_df().lazy()
    xol = Xolars(ds=_ds(), df={"gene_id": lazy_gene_df})
    assert isinstance(xol.df["gene_id"], pl.LazyFrame)
    result = xol.df["gene_id"].collect()
    assert list(result["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]


# ── isel ──────────────────────────────────────────────────────────────────────


def test_isel_gene_subset():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    sub = xol.isel(gene_id=[0, 2])
    assert list(sub.ds["gene_id"].values) == ["ENSG001", "ENSG003"]
    assert list(sub.df["gene_id"]["gene_id"]) == ["ENSG001", "ENSG003"]
    # sample_id dim untouched
    assert list(sub.df["sample_id"]["sample_id"]) == ["S1", "S2", "S3", "S4"]


def test_isel_sample_subset():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    sub = xol.isel(sample_id=[1, 3])
    assert list(sub.ds["sample_id"].values) == ["S2", "S4"]
    assert list(sub.df["sample_id"]["sample_id"]) == ["S2", "S4"]
    # gene dim untouched
    assert list(sub.df["gene_id"]["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]


def test_isel_preserves_data():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    sub = xol.isel(gene_id=[2])
    assert sub.ds["expr"].values.tolist() == [[8.0, 9.0, 10.0, 11.0]]


def test_isel_scalar_drops_dim():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    sub = xol.isel(gene_id=1)
    assert "gene_id" not in sub.df
    assert "sample_id" in sub.df


def test_isel_preserves_order():
    # Reversed index order — df must match ds coord order, not selection order
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    sub = xol.isel(gene_id=[2, 0])
    assert list(sub.ds["gene_id"].values) == list(sub.df["gene_id"]["gene_id"])


# ── sel ───────────────────────────────────────────────────────────────────────


def test_sel_gene_by_label():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    sub = xol.sel(gene_id=["ENSG001", "ENSG003"])
    assert list(sub.ds["gene_id"].values) == ["ENSG001", "ENSG003"]
    assert list(sub.df["gene_id"]["gene_id"]) == ["ENSG001", "ENSG003"]


def test_sel_sample_by_label():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    sub = xol.sel(sample_id=["S2", "S4"])
    assert list(sub.ds["sample_id"].values) == ["S2", "S4"]
    assert list(sub.df["sample_id"]["sample_id"]) == ["S2", "S4"]


def test_sel_lazy_frame():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df().lazy()})
    sub = xol.sel(gene_id=["ENSG002"])
    assert isinstance(sub.df["gene_id"], pl.LazyFrame)
    result = sub.df["gene_id"].collect()
    assert list(result["gene_id"]) == ["ENSG002"]


def test_sel_scalar_drops_dim():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    sub = xol.sel(gene_id="ENSG002")
    assert "gene_id" not in sub.df
    assert "sample_id" in sub.df


def test_sel_preserves_ds_coord_order():
    # Labels given in reverse — df order must match resulting ds coord order
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    sub = xol.sel(gene_id=["ENSG003", "ENSG001"])
    assert list(sub.ds["gene_id"].values) == list(sub.df["gene_id"]["gene_id"])


# ── write / open ──────────────────────────────────────────────────────────────


def test_write_creates_expected_files(tmp_path):
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    path = tmp_path / "mydata"
    xol.write(path, mode="w")
    assert (path / "dataset.zarr").exists()
    assert (path / "gene_id.parquet").exists()
    assert (path / "sample_id.parquet").exists()


def test_open_returns_lazy(tmp_path):
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    path = tmp_path / "mydata"
    xol.write(path, mode="w")
    loaded = Xolars.open(path)
    assert isinstance(loaded.df["gene_id"], pl.LazyFrame)
    assert isinstance(loaded.df["sample_id"], pl.LazyFrame)


def test_round_trip_eager(tmp_path):
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    path = tmp_path / "mydata"
    xol.write(path, mode="w")
    loaded = Xolars.open(path).collect()
    assert list(loaded.df["gene_id"]["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]
    assert list(loaded.df["sample_id"]["sample_id"]) == ["S1", "S2", "S3", "S4"]
    np.testing.assert_array_equal(loaded.ds["expr"].values, _ds()["expr"].values)


def test_open_accepts_str_path(tmp_path):
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df(), "sample_id": _sample_df()})
    path = tmp_path / "mydata"
    xol.write(path, mode="w")
    loaded = Xolars.open(str(path))
    assert isinstance(loaded.df["gene_id"], pl.LazyFrame)
    assert isinstance(loaded.df["sample_id"], pl.LazyFrame)


def test_round_trip_lazy_write(tmp_path):
    xol = Xolars(
        ds=_ds(),
        df={"gene_id": _gene_df().lazy(), "sample_id": _sample_df().lazy()},
    )
    path = tmp_path / "lazydata"
    xol.write(path, mode="w")
    loaded = Xolars.open(path).collect()
    assert list(loaded.df["gene_id"]["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]


# ── assign ────────────────────────────────────────────────────────────────────


def test_assign_1d_into_existing_frame():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    out = xol.assign(gc=("gene_id", [0.4, 0.5, 0.6]))
    frame = out.df["gene_id"]
    assert list(frame["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]
    assert list(frame["gc"]) == [0.4, 0.5, 0.6]
    # original is untouched (frozen / functional)
    assert "gc" not in xol.df["gene_id"].columns


def test_assign_1d_auto_creates_frame():
    xol = Xolars(ds=_ds(), df={"sample_id": _sample_df()})  # no gene_id frame
    out = xol.assign(gc=("gene_id", [0.4, 0.5, 0.6]))
    assert "gene_id" in out.df
    frame = out.df["gene_id"]
    assert list(frame["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]
    assert list(frame["gc"]) == [0.4, 0.5, 0.6]


def test_assign_overwrites_existing_column():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})  # has a "chrom" column
    out = xol.assign(chrom=("gene_id", ["chrX", "chrY", "chrM"]))
    assert list(out.df["gene_id"]["chrom"]) == ["chrX", "chrY", "chrM"]


def test_assign_preserves_lazy_kind():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df().lazy()})
    out = xol.assign(gc=("gene_id", [0.4, 0.5, 0.6]))
    assert isinstance(out.df["gene_id"], pl.LazyFrame)
    assert list(out.df["gene_id"].collect()["gc"]) == [0.4, 0.5, 0.6]
