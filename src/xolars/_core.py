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
