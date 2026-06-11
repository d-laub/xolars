"""xolars — an xarray Dataset paired with per-dimension Polars frames.

The :class:`Xolars` container keeps an :class:`xarray.Dataset` and one Polars
frame per dimension aligned to the Dataset's coordinate order, including under
``isel``/``sel`` selection and zarr+parquet round-trips.
"""

from __future__ import annotations

from xolars._core import Xolars

__all__ = ["Xolars"]
