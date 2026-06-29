"""xolars — an xarray Dataset paired with per-dimension Polars frames.

The :class:`Xolars` container keeps an :class:`xarray.Dataset` and one Polars
frame per dimension aligned to the Dataset's coordinate order, including under
``isel``/``sel`` selection and zarr+parquet round-trips.

Use :meth:`Xolars.assign` to add one or more named variables (1-D data routes to the
matching dimension's frame; N-D data routes to the Dataset) and
:meth:`Xolars.merge` to bring in whole xarray objects and Polars frames at once,
peeling 1-D variables out of xarray objects into Polars.
"""

from __future__ import annotations

from xolars._core import Xolars

__all__ = ["Xolars"]
