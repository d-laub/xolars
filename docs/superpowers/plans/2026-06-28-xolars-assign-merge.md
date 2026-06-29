# Xolars `assign` and `merge` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add functional `Xolars.assign` and `Xolars.merge` methods that route 1-D data into per-dimension Polars frames and N-D data into the xarray Dataset.

**Architecture:** Both methods are pure: they build a new `ds` + `df` dict and return a new instance via `evolve(self, ds=..., df=...)`, which re-runs `__attrs_post_init__` so the existing set-match validation and coord-order reorder are reused. Routing is by dimensionality (1-D → Polars, ≥2-D / scalar → xarray). Shared private module-level helpers do the polars-side join/auto-create/kind-coercion work.

**Tech Stack:** Python ≥3.11, xarray, polars, numpy, attrs (frozen, `Generic[F]`), pytest, uv.

## Global Constraints

- Container is `attrs`-frozen and `Generic[F]` where `F = pl.DataFrame | pl.LazyFrame`; **every frame is the same kind** (all eager or all lazy). Auto-created/coerced frames must match existing frames' kind; default eager `pl.DataFrame` when `df` is empty.
- Each `df[dim]` must always keep **exactly** `ds[dim]`'s coordinate set (validated by `__attrs_post_init__`). Left-align joins preserve this because they keep the base frame's rows.
- Conflict policy: a name that already exists (a `ds` variable or a frame column) is **overwritten silently**.
- `ds` coordinates are never grown by these methods; merged/assigned data is left-aligned to the fixed coords (subset → null, superset → dropped).
- All code lives in `src/xolars/_core.py`. Run tests with `uv run pytest`. Conventional-commit messages.

---

### Task 1: Polars-side helpers + `assign` routing 1-D data

**Files:**
- Modify: `src/xolars/_core.py` (add imports, module-level helpers, `assign` method on `Xolars`)
- Test: `tests/test_xolars.py`

**Interfaces:**
- Consumes: existing `Xolars(ds, df)`, `evolve`, `_reorder`; fixtures `_ds`, `_gene_df`, `_sample_df`.
- Produces (later tasks rely on these exact signatures):
  - `_frame_kind(df: Mapping[Hashable, Any]) -> type` → `pl.DataFrame` or `pl.LazyFrame`
  - `_coerce_kind(frame, kind) -> pl.DataFrame | pl.LazyFrame`
  - `_require_dim(ds: xr.Dataset, dim: str, name: str) -> None`
  - `_ensure_frame(df: dict, ds: xr.Dataset, dim: str, kind) -> pl.DataFrame | pl.LazyFrame`
  - `_attach_columns(frame: F, incoming, key: str) -> F` (left-join, incoming overwrites overlapping non-key columns)
  - `Xolars.assign(variables=None, /, **variables_kwargs) -> Self`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_xolars.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_xolars.py -k assign -v`
Expected: FAIL with `AttributeError: 'Xolars' object has no attribute 'assign'`

- [ ] **Step 3: Add imports and module-level helpers**

In `src/xolars/_core.py`, extend the `collections.abc` import to include `Sequence`:

```python
from collections.abc import Hashable, Mapping, Sequence
```

Add these helpers at module level (next to `_reorder` / `_filter_df`):

```python
def _frame_kind(df: Mapping[Hashable, Any]) -> type:
    """Return the polars frame kind the container uses (eager default)."""
    for frame in df.values():
        return pl.LazyFrame if isinstance(frame, pl.LazyFrame) else pl.DataFrame
    return pl.DataFrame


def _coerce_kind(frame: pl.DataFrame | pl.LazyFrame, kind: type) -> pl.DataFrame | pl.LazyFrame:
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
        incoming = incoming.lazy() if isinstance(incoming, pl.DataFrame) else incoming
    else:
        incoming = incoming.collect() if isinstance(incoming, pl.LazyFrame) else incoming
    overlap = [c for c in _columns(incoming) if c != key and c in _columns(frame)]
    base = frame.drop(overlap) if overlap else frame
    return cast(F, base.join(incoming, on=key, how="left"))
```

- [ ] **Step 4: Add the `_prepare_assign` helper and `assign` method**

Add the routing helper at module level:

```python
def _is_dims(x: Any) -> bool:
    """True if x looks like an xarray dims spec: a str or a sequence of str."""
    return isinstance(x, str) or (
        isinstance(x, (tuple, list)) and len(x) > 0 and all(isinstance(d, str) for d in x)
    )


def _prepare_assign(name: str, value: Any, ds: xr.Dataset):
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
            key_vals = value[dim].to_numpy() if dim in value.coords else ds[dim].to_numpy()
            incoming = pl.DataFrame({dim: key_vals, name: value.to_numpy()})
            return ("polars", dim, incoming)
        return ("xarray", value)
    if isinstance(value, tuple) and len(value) == 2 and _is_dims(value[0]):
        raw_dims, data = value
        dims = (raw_dims,) if isinstance(raw_dims, str) else tuple(str(d) for d in raw_dims)
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
```

Add the `assign` method to `Xolars` (place it after `collect`):

```python
    def assign(
        self,
        variables: Mapping[Hashable, Any] | None = None,
        /,
        **variables_kwargs: Any,
    ) -> Self:
        merged = dict(variables or {})
        merged.update(variables_kwargs)
        new_ds = self.ds
        new_df = dict(self.df)
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_xolars.py -k assign -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the full suite + type check**

Run: `uv run pytest tests/test_xolars.py -q && uv run pyrefly check`
Expected: all pass, no type errors.

- [ ] **Step 7: Commit**

```bash
git add src/xolars/_core.py tests/test_xolars.py
git commit -m "feat: add Xolars.assign routing 1-D data to polars frames"
```

---

### Task 2: `assign` — xarray routing (≥2-D, scalar) and DataArray label-align

**Files:**
- Modify: `src/xolars/_core.py` (no new code expected — verifies the `('xarray', ...)` branch and DataArray path written in Task 1)
- Test: `tests/test_xolars.py`

**Interfaces:**
- Consumes: `Xolars.assign` and `_prepare_assign` from Task 1.
- Produces: nothing new; this task is verification + any fixes the tests surface.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail or pass**

Run: `uv run pytest tests/test_xolars.py -k "assign_2d or assign_scalar or assign_dataarray or assign_mapping" -v`
Expected: these exercise branches from Task 1; they should PASS. If any FAIL, fix the relevant branch in `_prepare_assign` / `assign` (e.g. `np.isscalar` handling, or `new_ds.assign` tuple form) until green.

- [ ] **Step 3: Run the full suite + type check**

Run: `uv run pytest tests/test_xolars.py -q && uv run pyrefly check`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_xolars.py src/xolars/_core.py
git commit -m "test: cover assign xarray routing and DataArray label-align"
```

---

### Task 3: `assign` — error cases

**Files:**
- Modify: `src/xolars/_core.py` (only if a test surfaces a gap)
- Test: `tests/test_xolars.py`

**Interfaces:**
- Consumes: `Xolars.assign`, `_prepare_assign`, `_require_dim` from Task 1.
- Produces: nothing new.

- [ ] **Step 1: Write the failing tests**

```python
def test_assign_rejects_dataset_value():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    extra = xr.Dataset({"a": ("gene_id", [1, 2, 3]), "b": ("gene_id", [4, 5, 6])},
                       coords={"gene_id": ["ENSG001", "ENSG002", "ENSG003"]})
    with pytest.raises(ValueError, match="use merge"):
        xol.assign(thing=extra)


def test_assign_rejects_bare_array_no_dims():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    with pytest.raises(ValueError, match="no dimension information"):
        xol.assign(gc=np.array([0.4, 0.5, 0.6]))


def test_assign_rejects_unknown_dim():
    xol = Xolars(ds=_ds(), df={"gene_id": _gene_df()})
    with pytest.raises(ValueError, match="not a dimension"):
        xol.assign(x=("nope", [1, 2, 3]))
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_xolars.py -k "assign_rejects" -v`
Expected: PASS (the guards exist in `_prepare_assign`). If a message doesn't match, adjust the `match=` regex or the raised message text to agree.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest tests/test_xolars.py -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_xolars.py src/xolars/_core.py
git commit -m "test: cover assign error cases"
```

---

### Task 4: `merge` — xarray objects (peel 1-D, left-merge ≥2-D)

**Files:**
- Modify: `src/xolars/_core.py` (add `_peel_1d` helper and `Xolars.merge`)
- Test: `tests/test_xolars.py`

**Interfaces:**
- Consumes: `_frame_kind`, `_coerce_kind`, `_require_dim`, `_ensure_frame`, `_attach_columns` from Task 1.
- Produces:
  - `_peel_1d(obj: xr.Dataset | xr.DataArray) -> tuple[xr.Dataset, list[xr.DataArray]]`
  - `Xolars.merge(*objects, frames=None) -> Self`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_xolars.py -k merge -v`
Expected: FAIL with `AttributeError: 'Xolars' object has no attribute 'merge'`

- [ ] **Step 3: Add `_peel_1d` helper**

```python
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
            oned.append(da.rename(vname))
        else:
            keep.append(vname)
    return ds[keep], oned
```

- [ ] **Step 4: Add the `merge` method**

```python
    def merge(
        self,
        *objects: xr.Dataset | xr.DataArray | pl.DataFrame | pl.LazyFrame,
        frames: Mapping[str, Any] | None = None,
    ) -> Self:
        new_ds = self.ds
        new_df = dict(self.df)
        kind = _frame_kind(self.df)

        def route_frame(dim: str, frame: pl.DataFrame | pl.LazyFrame) -> None:
            nonlocal new_df
            _require_dim(new_ds, dim, dim)
            base = _ensure_frame(new_df, new_ds, dim, kind)
            new_df[dim] = _attach_columns(base, _coerce_kind(frame, kind), dim)

        def route_oned(da: xr.DataArray) -> None:
            dim = str(da.dims[0])
            name = str(da.name)
            key_vals = da[dim].to_numpy() if dim in da.coords else new_ds[dim].to_numpy()
            route_frame(dim, pl.DataFrame({dim: key_vals, name: da.to_numpy()}))

        for obj in objects:
            if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
                route_frame(_infer_dim(obj, new_ds), obj)
            elif isinstance(obj, (xr.Dataset, xr.DataArray)):
                keep, oned = _peel_1d(obj)
                for da in oned:
                    route_oned(da)
                if len(keep.data_vars) > 0:
                    overlap = [v for v in keep.data_vars if v in new_ds.data_vars]
                    base = new_ds.drop_vars(overlap) if overlap else new_ds
                    new_ds = xr.merge([base, keep], join="left", compat="override")
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
```

Also add the `_infer_dim` helper now (used by `merge`'s polars branch; exercised in Task 5):

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_xolars.py -k merge -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Run the full suite + type check**

Run: `uv run pytest tests/test_xolars.py -q && uv run pyrefly check`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/xolars/_core.py tests/test_xolars.py
git commit -m "feat: add Xolars.merge for xarray objects with 1-D peeling"
```

---

### Task 5: `merge` — polars frames (inference, `frames=`, left-align, kind)

**Files:**
- Modify: `src/xolars/_core.py` (only if a test surfaces a gap)
- Test: `tests/test_xolars.py`

**Interfaces:**
- Consumes: `Xolars.merge`, `_infer_dim`, `_attach_columns` from Task 4 / Task 1.
- Produces: nothing new.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_xolars.py -k merge -v`
Expected: PASS. If `test_merge_left_align_subset_nulls` fails on ordering, confirm `_attach_columns` left-joins onto the base frame (base rows are ds coord order) and that `__attrs_post_init__` reorders the result; fix only the helper, not the test.

- [ ] **Step 3: Run the full suite + type check**

Run: `uv run pytest tests/test_xolars.py -q && uv run pyrefly check`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_xolars.py src/xolars/_core.py
git commit -m "test: cover merge polars inference, frames=, and left-align"
```

---

### Task 6: Docs — module docstring and skill reference

**Files:**
- Modify: `src/xolars/__init__.py` (module docstring)
- Modify: `/Users/david/.claude/skills/xolars/SKILL.md` (public-surface reference)

**Interfaces:**
- Consumes: final `assign` / `merge` signatures from Tasks 1 & 4.
- Produces: documentation only.

- [ ] **Step 1: Update the module docstring**

In `src/xolars/__init__.py`, extend the existing docstring to mention the two new methods. Replace the closing of the docstring so it reads:

```python
"""xolars — an xarray Dataset paired with per-dimension Polars frames.

The :class:`Xolars` container keeps an :class:`xarray.Dataset` and one Polars
frame per dimension aligned to the Dataset's coordinate order, including under
``isel``/``sel`` selection and zarr+parquet round-trips.

Use :meth:`Xolars.assign` to add a single named variable (1-D data routes to the
matching dimension's frame; N-D data routes to the Dataset) and
:meth:`Xolars.merge` to bring in whole xarray objects and Polars frames at once,
peeling 1-D variables out of xarray objects into Polars.
"""
```

- [ ] **Step 2: Update the xolars skill reference**

In `/Users/david/.claude/skills/xolars/SKILL.md`, under the "Public surface" /
selection sections, add a short subsection documenting `assign` and `merge`.
Insert after the "Selection — isel / sel" section:

````markdown
## Adding data — assign / merge

Both are functional (return a **new** `Xolars`); 1-D data lives in the matching
dimension's Polars frame, N-D data lives in the Dataset.

```python
# assign one named variable (xarray.Dataset.assign idiom)
xo2 = xo.assign(gc=("gene_id", gc_array))                 # 1-D -> gene_id frame
xo2 = xo.assign(counts=(("gene_id", "sample_id"), mat))   # 2-D -> Dataset
xo2 = xo.assign(score=some_named_dataarray)               # aligned by label
# passing an xarray.Dataset value is an error -> use merge()

# merge whole objects; 1-D vars in xarray objects move to Polars
xo2 = xo.merge(other_dataset, gene_frame)                 # bare frame: dim inferred
xo2 = xo.merge(frames={"gene_id": [f1, f2]})              # explicit, aligned to gene_id
```

Conflicting names overwrite silently. Polars/xarray inputs are **left-aligned**
to the fixed coordinates (subset → nulls, superset → dropped); `ds` coordinates
are never grown.
````

- [ ] **Step 3: Sanity-check imports still work**

Run: `uv run python -c "import xolars; print(xolars.Xolars.assign, xolars.Xolars.merge)"`
Expected: prints both method objects without error.

- [ ] **Step 4: Commit**

```bash
git add src/xolars/__init__.py
git commit -m "docs: document Xolars.assign and Xolars.merge"
```

(The skill file under `~/.claude/skills/` is outside the repo; it is edited but not committed here.)

---

## Notes for the implementer

- `evolve(self, ds=..., df=...)` re-runs `__attrs_post_init__`, which **validates and reorders** every frame. You do not need to re-sort frames after a join — construction handles it. This is why left-joins onto the base frame are safe: the base already has exactly the dim's coordinates.
- Keep every frame the same kind. When in doubt, route incoming polars data through `_coerce_kind(frame, _frame_kind(self.df))` before joining.
- `np.isscalar(np.array(3))` is `False`, which is why `_prepare_assign` checks 0-D ndarrays separately.
- Do not weaken the construction invariants in `__attrs_post_init__`; the new methods are designed to satisfy them, not bypass them.
