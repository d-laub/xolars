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


def test_assign_2d_goes_to_dataset():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    counts = np.arange(12.0, 24.0).reshape(3, 4)
    out = xol.assign(counts=(("gene_id", "sample_id"), counts))
    assert "counts" in out.ds.data_vars
    np.testing.assert_array_equal(out.ds["counts"].values, counts)
    # not added to any frame
    assert "counts" not in out.df["gene_id"].columns


def test_assign_scalar_goes_to_dataset():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    out = xol.assign(version=3)
    assert int(out.ds["version"].values) == 3
    assert out.ds["version"].ndim == 0


def test_assign_dataarray_aligns_by_label():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    # coords deliberately shuffled relative to ds order
    da = xr.DataArray(
        [0.6, 0.4, 0.5],
        dims="gene_id",
        coords={"gene_id": ["ENSG003", "ENSG001", "ENSG002"]},
        name="gc",
    )
    out = xol.assign(gc=da)
    frame = out.df["gene_id"]
    # values realigned to ds coord order ENSG001, ENSG002, ENSG003
    assert list(frame["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]
    assert list(frame["gc"]) == [0.4, 0.5, 0.6]


def test_assign_mapping_and_kwargs_combine():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    out = xol.assign({"a": ("gene_id", [1, 2, 3])}, b=("gene_id", [4, 5, 6]))
    assert list(out.df["gene_id"]["a"]) == [1, 2, 3]
    assert list(out.df["gene_id"]["b"]) == [4, 5, 6]


def test_assign_2d_overwrites_existing_variable():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})  # ds has "expr" (3x4)
    new_expr = np.zeros((3, 4))
    out = xol.assign(expr=(("gene_id", "sample_id"), new_expr))
    np.testing.assert_array_equal(out.ds["expr"].values, new_expr)


def test_assign_rejects_dataset_value():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    extra = xr.Dataset({"a": ("gene_id", [1, 2, 3]), "b": ("gene_id", [4, 5, 6])},
                       coords={"gene_id": ["ENSG001", "ENSG002", "ENSG003"]})
    with pytest.raises(ValueError, match="Use merge"):
        xol.assign(thing=extra)


def test_assign_rejects_bare_array_no_dims():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    with pytest.raises(ValueError, match="no dimension information"):
        xol.assign(gc=np.array([0.4, 0.5, 0.6]))


def test_assign_rejects_unknown_dim():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    with pytest.raises(ValueError, match="not a dimension"):
        xol.assign(x=("nope", [1, 2, 3]))


# ── merge ─────────────────────────────────────────────────────────────────────


def test_merge_dataset_splits_1d_and_2d():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    incoming = xr.Dataset(
        {
            "gc": ("gene_id", [0.4, 0.5, 0.6]),                       # 1-D -> polars
            "counts": (("gene_id", "sample_id"), np.zeros((3, 4))),   # 2-D -> ds
        },
        coords={
            "gene_id": ["ENSG001", "ENSG002", "ENSG003"],
            "sample_id": ["S1", "S2", "S3", "S4"],
        },
    )
    out = xol.merge(incoming)
    assert list(out.df["gene_id"]["gc"]) == [0.4, 0.5, 0.6]
    assert "counts" in out.ds.data_vars
    assert "gc" not in out.ds.data_vars  # 1-D was peeled out, not left in ds


def test_merge_dataarray_1d_goes_to_polars():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    da = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims="gene_id",
        coords={"gene_id": ["ENSG001", "ENSG002", "ENSG003"]},
        name="score",
    )
    out = xol.merge(da)
    assert list(out.df["gene_id"]["score"]) == [1.0, 2.0, 3.0]


def test_merge_unnamed_dataarray_raises():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    da = xr.DataArray([1.0, 2.0, 3.0], dims="gene_id",
                      coords={"gene_id": ["ENSG001", "ENSG002", "ENSG003"]})
    with pytest.raises(ValueError, match="unnamed DataArray"):
        xol.merge(da)


def test_merge_bare_frame_infers_dim():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    extra = pl.DataFrame({"gene_id": ["ENSG001", "ENSG002", "ENSG003"],
                          "gc": [0.4, 0.5, 0.6]})
    out = xol.merge(extra)
    assert list(out.df["gene_id"]["gc"]) == [0.4, 0.5, 0.6]


def test_merge_bare_frame_ambiguous_raises():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    # both gene_id and sample_id are ds dims -> ambiguous
    bad = pl.DataFrame({"gene_id": ["ENSG001"], "sample_id": ["S1"], "v": [1]})
    with pytest.raises(ValueError, match="infer target dimension"):
        xol.merge(bad)


def test_merge_bare_frame_no_match_raises():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    bad = pl.DataFrame({"not_a_dim": [1, 2, 3], "v": [4, 5, 6]})
    with pytest.raises(ValueError, match="infer target dimension"):
        xol.merge(bad)


def test_merge_frames_keyword_single():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    extra = pl.DataFrame({"gene_id": ["ENSG001", "ENSG002", "ENSG003"],
                          "gc": [0.4, 0.5, 0.6]})
    out = xol.merge(frames={"gene_id": extra})
    assert list(out.df["gene_id"]["gc"]) == [0.4, 0.5, 0.6]


def test_merge_frames_keyword_sequence():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    f1 = pl.DataFrame({"gene_id": ["ENSG001", "ENSG002", "ENSG003"], "a": [1, 2, 3]})
    f2 = pl.DataFrame({"gene_id": ["ENSG001", "ENSG002", "ENSG003"], "b": [4, 5, 6]})
    out = xol.merge(frames={"gene_id": [f1, f2]})
    assert list(out.df["gene_id"]["a"]) == [1, 2, 3]
    assert list(out.df["gene_id"]["b"]) == [4, 5, 6]


def test_merge_left_align_subset_nulls():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    partial = pl.DataFrame({"gene_id": ["ENSG001", "ENSG003"], "gc": [0.4, 0.6]})
    out = xol.merge(partial)
    frame = out.df["gene_id"]
    assert list(frame["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]
    assert frame["gc"].to_list() == [0.4, None, 0.6]


def test_merge_superset_drops_extra():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    over = pl.DataFrame({"gene_id": ["ENSG001", "ENSG002", "ENSG003", "ENSG999"],
                         "gc": [0.4, 0.5, 0.6, 0.9]})
    out = xol.merge(over)
    frame = out.df["gene_id"]
    assert list(frame["gene_id"]) == ["ENSG001", "ENSG002", "ENSG003"]
    assert list(frame["gc"]) == [0.4, 0.5, 0.6]


def test_merge_overwrites_existing_column():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})  # has "chrom"
    over = pl.DataFrame({"gene_id": ["ENSG001", "ENSG002", "ENSG003"],
                         "chrom": ["chrX", "chrY", "chrM"]})
    out = xol.merge(over)
    assert list(out.df["gene_id"]["chrom"]) == ["chrX", "chrY", "chrM"]


def test_merge_preserves_lazy_kind():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df().lazy()})
    extra = pl.DataFrame({"gene_id": ["ENSG001", "ENSG002", "ENSG003"],
                          "gc": [0.4, 0.5, 0.6]})
    out = xol.merge(extra)
    assert isinstance(out.df["gene_id"], pl.LazyFrame)
    assert list(out.df["gene_id"].collect()["gc"]) == [0.4, 0.5, 0.6]
