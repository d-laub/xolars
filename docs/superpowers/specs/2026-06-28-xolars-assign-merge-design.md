# Xolars: `assign` and `merge` — design

**Date:** 2026-06-28
**Status:** Approved, pending implementation plan

## Summary

Add two functional methods to `Xolars` for bringing 1-D and N-D data into the
container:

- `Xolars.assign(...)` — add or replace one or more **named** variables, mirroring
  `xarray.Dataset.assign`.
- `Xolars.merge(...)` — bring in whole xarray objects and polars frames at once,
  peeling 1-D data out of xarray objects and routing it to polars.

Both are pure/functional and return a **new** `Xolars`, consistent with the
existing frozen design (`isel`/`sel`/`collect` already return new instances).
`__setitem__` is intentionally **not** added: polars frames are immutable and the
container is frozen, so an in-place bracket setter would fight the design.

## Background

`Xolars` (`src/xolars/_core.py`) is an `attrs`-frozen, `Generic[F]` container
pairing one `xarray.Dataset` with one polars frame per dimension
(`df: Mapping[Hashable, F]`, `F = pl.DataFrame | pl.LazyFrame`). Each frame has a
column named exactly after its dimension (the join key) whose values must match
that dim's coordinates as a set; construction validates this and reorders rows to
the Dataset's coordinate order. `isel`/`sel` keep the frames aligned under
selection; `write`/`open` round-trip to zarr + parquet.

There is currently no ergonomic way to *add* data after construction. These two
methods fill that gap.

## Governing invariant

Every data variable is routed by its **dimensionality**:

| Dimensionality | Destination |
|---|---|
| 1-D (along exactly one dim `D`) | a column in `df[D]` |
| ≥2-D | a data variable in `ds` |
| 0-D / scalar | a scalar variable in `ds` |

When 1-D data targets a dim `D` that has no frame yet, `df[D]` is **auto-created**,
seeded with `D`'s coordinate values as the key column, then the new column is
attached.

Both methods build a new `ds` + `df` and finish with `evolve(self, ds=..., df=...)`,
which re-runs `__attrs_post_init__`. This **reuses the existing validation and
coord-order reorder for free**: as long as each `df[D]` keeps exactly `ds[D]`'s
coordinate set, validation passes.

### Frame-kind uniformity

`Xolars` is `Generic[F]` — every frame is the same kind (all eager or all lazy).
Auto-created and coerced frames must match the kind of existing frames:

- if `self.df` has frames → match their kind (lazy if lazy, else eager);
- if `self.df` is empty → default to eager `pl.DataFrame`.

Incoming polars data is coerced to the container's kind before joining.

## API 1: `assign`

```python
def assign(self, variables=None, /, **variables_kwargs) -> Self
```

Mirrors `xarray.Dataset.assign`. Accepts a mapping (positional) and/or keyword
arguments; keys are variable names, values are the data. Each value may be:

- an `xr.DataArray` — dims known from the array; **aligned to `ds` coordinates by
  label** before extraction;
- a **tuple `(dims, data)`** following the xarray idiom — `("gene_id", arr)` or
  `(("gene_id", "sample_id"), arr)` — taken **positionally** in `ds` coordinate
  order;
- a **scalar** → a 0-D variable in `ds`.

Routing follows the governing invariant by the value's resulting ndim. 1-D data is
attached positionally as a column (both the frame and `ds[D]` are kept in coord
order, so no join is needed); `DataArray` inputs are aligned by label first.

**Conflict policy:** a name that already exists (a `ds` variable or a frame column)
is **overwritten silently**, matching `xarray.Dataset.assign`.

**Errors:**

- value is an `xr.Dataset` → raise: *more than one variable; a single name is
  ambiguous — use `merge` instead*.
- value is a bare array / `pl.Series` with no dim information → raise: dim cannot be
  inferred; pass `(dims, data)` or a `DataArray` (same constraint as xarray).
- the resolved dim is not an existing `ds` dimension → raise (this method does not
  grow dimensions).

**Out of scope (YAGNI):** callable values (`lambda xo: ...`) that xarray supports.
May be added later if needed.

## API 2: `merge`

```python
def merge(self, *objects, frames=None) -> Self
#   *objects: xr.Dataset | xr.DataArray | pl.DataFrame | pl.LazyFrame
#   frames:   Mapping[str, F | Sequence[F]] | None
```

Inputs:

- **xarray `*objects`** (`Dataset` / `DataArray`): split into 1-D variables (routed
  to polars per the invariant) and ≥2-D variables (merged into `ds`). The
  multi-D merge keeps `ds` coordinates fixed (`join="left"`, conflicts overridden).
  `DataArray`s must be named.
- **bare polars `*objects`** (`DataFrame` / `LazyFrame`): the target dim is inferred
  as the single column whose name matches a `ds` dimension. **Raise if zero or more
  than one column matches** (ambiguous) — the user should use `frames=` to be
  explicit.
- **`frames=` keyword**: an explicit `Mapping[str, F | Sequence[F]]`. A value that is
  a single frame is joined into `df[dim]`; a value that is a **sequence** of frames
  means several frames all aligned to that same coordinate, joined in turn.

**Alignment — left-align:** for each target dim `D`,
`df[D].join(incoming, on=D, how="left")`:

- incoming covers a **subset** of coords → missing rows get null columns;
- incoming covers a **superset** → extra rows are dropped;
- `ds` coordinates never change, so the `df`-keyed-to-coords invariant holds.

Overlapping non-key columns from incoming **overwrite** the existing ones. The same
left-align/overwrite policy applies to 1-D variables peeled out of xarray objects
(auto-creating `df[D]` if absent).

Returns a new `Xolars`.

## Implementation

All changes land in `src/xolars/_core.py` (currently ~135 lines). New module-level
private helpers, alongside the existing `_reorder` / `_filter_df`:

- `_classify(dims) -> "polars" | "xarray"` — routing decision from a dim tuple;
- `_ensure_frame(df, ds, dim)` — return `df[dim]`, auto-creating a seeded frame of the
  correct kind if absent;
- `_attach_columns(frame, incoming, key)` — left-join with the overwrite policy;
- `_peel_1d(obj)` — separate 1-D variables from ≥2-D variables of an xarray object;
- `_coerce_kind(frame, like)` — coerce a polars frame to match the container's kind.

`assign` and `merge` are thin orchestrators over these helpers, both ending in
`evolve(self, ds=new_ds, df=new_df)` so construction-time validation runs on the
result.

## Testing

Extend `tests/test_xolars.py` (pytest), reusing its `_ds` / `_gene_df` /
`_sample_df` fixtures.

`assign`:

- 1-D into an existing frame, via `(dim, array)` (positional) and via a `DataArray`
  with shuffled coords (label-aligned) — column values land in `ds` coord order;
- 1-D auto-creates `df[D]` when the dim has no frame;
- ≥2-D value → `ds` data variable; scalar → `ds` scalar variable;
- overwrite an existing column and an existing `ds` variable;
- rejects an `xr.Dataset` value (points to `merge`);
- rejects a bare array / `pl.Series` with no dim info;
- rejects a dim not present in `ds`.

`merge`:

- `xr.Dataset` with mixed 1-D and 2-D variables → 1-D to polars, 2-D to `ds`;
- bare polars frame → dim inferred from the matching column;
- bare polars frame with zero / two matching columns → raises (ambiguous);
- `frames={dim: frame}` and `frames={dim: [f1, f2]}` (sequence);
- left-align subset (null columns) and superset (extra rows dropped);
- overwrite a conflicting column;
- preserves frame kind (lazy stays lazy, eager stays eager);
- result satisfies construction invariants (round-trips through validation).

## Docs

Refresh the `xolars` skill doc and the module docstring in `src/xolars/__init__.py`
to mention `assign` and `merge`.
