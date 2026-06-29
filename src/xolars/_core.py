from __future__ import annotations

from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any, Generic, Iterable, TypeVar, cast

import numpy as np
import polars as pl
import xarray as xr
from attrs import define, evolve
from typing_extensions import Self
from xarray.core.types import ErrorOptionsWithWarn, ZarrWriteModes

F = TypeVar("F", bound=pl.DataFrame | pl.LazyFrame)


@define(frozen=True)
class Xolars(Generic[F]):
    ds: xr.Dataset
    df: Mapping[Hashable, F]

    def __attrs_post_init__(self):
        new_df = {}
        for dim, frame in self.df.items():
            dim_str = str(dim)
            if dim_str not in self.ds.sizes:
                raise ValueError(
                    f"'{dim}' is not a dimension in ds. Available: {list(self.ds.dims)}"
                )
            col_names = (
                frame.collect_schema().names()
                if isinstance(frame, pl.LazyFrame)
                else frame.columns
            )
            if dim_str not in col_names:
                raise ValueError(
                    f"DataFrame for dim '{dim}' is missing column '{dim_str}'"
                )
            # Collect only the dim column for validation (cheap — just IDs)
            if isinstance(frame, pl.LazyFrame):
                id_col: pl.Series = frame.select(dim_str).collect()[dim_str]
            else:
                id_col = cast(pl.DataFrame, frame)[dim_str]
            ds_dim_values = self.ds[dim_str]
            df_dim_values: pl.Series = id_col
            common = np.intersect1d(ds_dim_values.to_numpy(), df_dim_values.to_numpy())
            if len(common) != len(ds_dim_values) or len(common) != len(df_dim_values):
                raise ValueError(
                    f"DataFrame for dim '{dim}' values don't match ds['{dim_str}'] coordinates. "
                    f"Expected {len(ds_dim_values)} number of shared values, got {len(common)}"
                )
            new_df[dim] = _reorder(frame, ds_dim_values)
        object.__setattr__(self, "df", new_df)

    @classmethod
    def open(cls, path: str | Path) -> Xolars[pl.LazyFrame]:
        path = Path(path)
        ds = xr.open_zarr(path / "dataset.zarr")
        df: dict[Hashable, pl.LazyFrame] = {}
        for parquet_path in sorted(path.glob("*.parquet")):
            df[parquet_path.stem] = pl.scan_parquet(parquet_path)
        return Xolars(ds=ds, df=df)

    def write(self, path: Path, mode: ZarrWriteModes):
        path.mkdir(parents=True, exist_ok=True)
        self.ds.to_zarr(path / "dataset.zarr", mode=mode)
        for dim, frame in self.df.items():
            out = path / f"{dim}.parquet"
            if isinstance(frame, pl.LazyFrame):
                frame.sink_parquet(out)
            else:
                cast(pl.DataFrame, frame).write_parquet(out)

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        missing_dims: ErrorOptionsWithWarn = "raise",
        **indexers_kwargs: Any,
    ) -> Self:
        merged = dict(indexers or {})
        merged.update(indexers_kwargs)
        new_ds = self.ds.isel(merged, drop=drop, missing_dims=missing_dims)
        return evolve(self, ds=new_ds, df=_filter_df(self.df, new_ds, merged))

    def sel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: int | float | Iterable[int | float] | None = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> Self:
        merged = dict(indexers or {})
        merged.update(indexers_kwargs)
        new_ds = self.ds.sel(merged, method=method, tolerance=tolerance, drop=drop)
        return evolve(self, ds=new_ds, df=_filter_df(self.df, new_ds, merged))

    def collect(self) -> Xolars[pl.DataFrame]:
        df: dict[Hashable, pl.DataFrame] = {
            k: v.collect() if isinstance(v, pl.LazyFrame) else cast(pl.DataFrame, v)
            for k, v in self.df.items()
        }
        return Xolars(self.ds, df)

    def assign(
        self,
        variables: Mapping[Hashable, Any] | None = None,
        /,
        **variables_kwargs: Any,
    ) -> Self:
        merged = dict(variables or {})
        merged.update(variables_kwargs)
        new_ds = self.ds
        new_df: dict[Hashable, pl.DataFrame | pl.LazyFrame] = dict(self.df)
        kind = _frame_kind(self.df)
        for name, value in merged.items():
            route = _prepare_assign(str(name), value, new_ds)
            if route[0] == "polars":
                _, dim, incoming = route
                frame = _ensure_frame(new_df, new_ds, dim, kind)
                new_df[dim] = _attach_columns(frame, _coerce_kind(incoming, kind), dim)
            else:
                new_ds = new_ds.assign({name: route[1]})
        return evolve(self, ds=new_ds, df=new_df)

    def merge(
        self,
        *objects: xr.Dataset | xr.DataArray | pl.DataFrame | pl.LazyFrame,
        frames: Mapping[str, Any] | None = None,
    ) -> Self:
        new_ds = self.ds
        new_df: dict[Hashable, pl.DataFrame | pl.LazyFrame] = dict(self.df)
        kind = _frame_kind(self.df)

        def route_frame(dim: str, frame: pl.DataFrame | pl.LazyFrame) -> None:
            # new_df is mutated in-place (dict item assignment), not rebound,
            # so no `nonlocal` is needed here.
            _require_dim(new_ds, dim, dim)
            base = _ensure_frame(new_df, new_ds, dim, kind)
            new_df[dim] = _attach_columns(base, _coerce_kind(frame, kind), dim)

        def route_oned(da: xr.DataArray) -> None:
            dim = str(da.dims[0])
            name = str(da.name)
            key_vals = (
                da[dim].to_numpy() if dim in da.coords else new_ds[dim].to_numpy()
            )
            route_frame(dim, pl.DataFrame({dim: key_vals, name: da.to_numpy()}))

        for obj in objects:
            if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
                route_frame(_infer_dim(obj, new_ds), obj)
            elif isinstance(obj, (xr.Dataset, xr.DataArray)):
                keep, oned = _peel_1d(obj)
                # Merge ≥2-D vars FIRST so that a 2-D var introducing a NEW
                # ds dimension (intended — no guard) is present in new_ds
                # before peeled 1-D vars or frames={newdim: ...} attach to it.
                if len(keep.data_vars) > 0:
                    overlap = [v for v in keep.data_vars if v in new_ds.data_vars]
                    base = new_ds.drop_vars(overlap) if overlap else new_ds
                    new_ds = xr.merge([base, keep], join="left", compat="override")
                for da in oned:
                    route_oned(da)
            else:
                raise TypeError(
                    f"merge() got an unsupported object of type "
                    f"{type(obj).__name__}; expected xarray.Dataset/DataArray or "
                    f"polars DataFrame/LazyFrame."
                )

        for dim, val in (frames or {}).items():
            seq = val if isinstance(val, (list, tuple)) else [val]
            for frame in seq:
                route_frame(str(dim), frame)

        return evolve(self, ds=new_ds, df=new_df)


def _reorder(frame: F, coords: xr.DataArray) -> F:
    """Reorder frame rows to match the given coordinate order."""
    dim_name = str(coords.name)
    order_df = pl.DataFrame({dim_name: coords.to_numpy()}).with_row_index("__i__")
    if isinstance(frame, pl.LazyFrame):
        joined = frame.join(order_df.lazy(), on=dim_name, how="right")
    else:
        joined = cast(pl.DataFrame, frame).join(order_df, on=dim_name, how="right")
    return cast(F, joined.sort("__i__").drop("__i__"))


def _filter_df(
    df: Mapping[Hashable, F],
    new_ds: xr.Dataset,
    merged: dict[str, Any],
) -> dict[Hashable, F]:
    """Filter df entries whose dim appears in merged indexers to match new_ds coords."""
    new_df: dict[Hashable, F] = {}
    for dim, frame in df.items():
        dim_str = str(dim)
        if dim_str in merged:
            if dim_str not in new_ds.dims:
                continue  # scalar index dropped this dimension
            keep = new_ds[dim_str].to_numpy()
            new_df[dim] = cast(F, frame.filter(pl.col(dim_str).is_in(keep)))
        else:
            new_df[dim] = frame
    return new_df


def _frame_kind(df: Mapping[Hashable, Any]) -> type:
    """Return the polars frame kind the container uses (eager default)."""
    for frame in df.values():
        return pl.LazyFrame if isinstance(frame, pl.LazyFrame) else pl.DataFrame
    return pl.DataFrame


def _coerce_kind(
    frame: pl.DataFrame | pl.LazyFrame, kind: type
) -> pl.DataFrame | pl.LazyFrame:
    """Coerce a polars frame to `kind` (pl.DataFrame or pl.LazyFrame)."""
    if kind is pl.LazyFrame:
        return frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _require_dim(ds: xr.Dataset, dim: str, name: str) -> None:
    """Raise if `dim` is not a dimension of `ds`."""
    if dim not in ds.sizes:
        raise ValueError(
            f"'{dim}' (for variable '{name}') is not a dimension in ds. "
            f"Available: {list(ds.dims)}"
        )


def _ensure_frame(
    df: dict[Hashable, Any], ds: xr.Dataset, dim: str, kind: type
) -> pl.DataFrame | pl.LazyFrame:
    """Return df[dim], auto-creating a frame seeded with the dim coordinate."""
    if dim in df:
        return df[dim]
    seed = pl.DataFrame({dim: ds[dim].to_numpy()})
    return _coerce_kind(seed, kind)


def _columns(frame: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Column names for an eager or lazy frame."""
    return (
        frame.collect_schema().names()
        if isinstance(frame, pl.LazyFrame)
        else frame.columns
    )


def _attach_columns(frame: F, incoming: pl.DataFrame | pl.LazyFrame, key: str) -> F:
    """Left-join `incoming` onto `frame` on `key`. Incoming's overlapping
    non-key columns overwrite frame's. Result keeps frame's kind and rows."""
    if isinstance(frame, pl.LazyFrame):
        lazy_in: pl.LazyFrame = (
            incoming.lazy() if isinstance(incoming, pl.DataFrame) else incoming
        )
        overlap = [c for c in _columns(lazy_in) if c != key and c in _columns(frame)]
        base_lf: pl.LazyFrame = frame.drop(overlap) if overlap else frame
        return cast(F, base_lf.join(lazy_in, on=key, how="left"))
    else:
        eager_frame = cast(pl.DataFrame, frame)
        eager_in: pl.DataFrame = cast(
            pl.DataFrame,
            incoming.collect() if isinstance(incoming, pl.LazyFrame) else incoming,
        )
        overlap = [
            c for c in _columns(eager_in) if c != key and c in _columns(eager_frame)
        ]
        base_df: pl.DataFrame = eager_frame.drop(overlap) if overlap else eager_frame
        return cast(F, base_df.join(eager_in, on=key, how="left"))


def _is_dims(x: Any) -> bool:
    """True if x looks like an xarray dims spec: a str or a sequence of str."""
    return isinstance(x, str) or (
        isinstance(x, (tuple, list))
        and len(x) > 0
        and all(isinstance(d, str) for d in x)
    )


def _peel_1d(obj: xr.Dataset | xr.DataArray) -> tuple[xr.Dataset, list[xr.DataArray]]:
    """Split an xarray object into a Dataset of >=2-D / scalar data vars and a
    list of its 1-D data vars (as named DataArrays)."""
    if isinstance(obj, xr.DataArray):
        if obj.name is None:
            raise ValueError("Cannot merge an unnamed DataArray; set its .name.")
        ds = obj.to_dataset()
    else:
        ds = obj
    oned: list[xr.DataArray] = []
    keep: list[Hashable] = []
    for vname, da in ds.data_vars.items():
        if da.ndim == 1:
            oned.append(da)  # da.name already equals vname from items()
        else:
            keep.append(vname)
    return ds[keep], oned


def _infer_dim(frame: pl.DataFrame | pl.LazyFrame, ds: xr.Dataset) -> str:
    """Infer the target dim as the single frame column matching a ds dimension."""
    matches = [c for c in _columns(frame) if c in ds.sizes]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot infer target dimension for polars frame: expected exactly "
            f"one column matching a ds dimension, found {matches}. "
            f"Use frames={{dim: frame}} to be explicit."
        )
    return matches[0]


def _prepare_assign(name: str, value: Any, ds: xr.Dataset) -> tuple[Any, ...]:
    """Validate and route one assign value.

    Returns ('polars', dim, incoming_df) for 1-D data, or ('xarray', value)
    for >=2-D / scalar data. Raises ValueError on ambiguous/unsupported input.
    """
    if isinstance(value, xr.Dataset):
        raise ValueError(
            f"assign value for '{name}' is an xarray.Dataset (more than one "
            f"variable); a single name is ambiguous. Use merge() instead."
        )
    if isinstance(value, xr.DataArray):
        dims = tuple(str(d) for d in value.dims)
        if len(dims) == 1:
            dim = dims[0]
            _require_dim(ds, dim, name)
            key_vals = (
                value[dim].to_numpy() if dim in value.coords else ds[dim].to_numpy()
            )
            incoming = pl.DataFrame({dim: key_vals, name: value.to_numpy()})
            return ("polars", dim, incoming)
        return ("xarray", value)
    if isinstance(value, tuple) and len(value) == 2 and _is_dims(value[0]):
        raw_dims, data = value
        dims = (
            (raw_dims,)
            if isinstance(raw_dims, str)
            else tuple(str(d) for d in raw_dims)
        )
        if len(dims) == 1:
            dim = dims[0]
            _require_dim(ds, dim, name)
            arr = data.to_numpy() if isinstance(data, pl.Series) else np.asarray(data)
            incoming = pl.DataFrame({dim: ds[dim].to_numpy(), name: arr})
            return ("polars", dim, incoming)
        return ("xarray", (dims, data))
    if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
        return ("xarray", value)
    raise ValueError(
        f"assign value for '{name}' has no dimension information. Pass a "
        f"(dims, data) tuple or an xarray.DataArray."
    )
