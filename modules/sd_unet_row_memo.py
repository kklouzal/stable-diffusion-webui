"""Step-scoped record of the main-pass UNet calls, replayed by guidance passes on a subset of rows.

Perturbed-attention guidance (PAG) evaluates the UNet a second time per denoiser step on inputs that are
exact duplicates of main-pass rows, and only its conditional rows are consumed. Recording the real
main-pass calls lets that pass run exactly the cond rows of each call, with the main pass's own chunking,
instead of re-deriving A1111's batching rules.

Lifetime and ownership (one denoiser step, one thread):

1. The consumer arms a ``MainPassMemo`` on the CFG denoiser (``arm``) in its ``cfg_denoiser`` callback. That
   installs a ``MainPassRecorder`` over the instance's ``run_inner_model`` once per denoiser.
2. Every main-pass ``run_inner_model(x, sigma, cond)`` call appends a ``CallRecord`` holding its inputs by
   reference. A1111 always puts the cond rows first and calls contiguous row ranges in order, so a record's
   ``start`` is the running row count; the recorder rejects calls that are not row slices of one batch.
3. While a recorded call runs, the outermost UNet forward that understands the memo claims its slot
   (``claim``). Nested forwards (ControlNet reference passes, CUDA-graph warm-up/capture) see the claimed
   slot and do nothing. A claimant may flag the call as row-coupled (``row_subset_ok = False``).
4. The consumer takes the memo (``disarm``) in its ``cfg_denoised`` callback, replays rows ``[0, m)`` of
   each record inside ``replaying`` and clears the memo in ``finally``. An exception inside a recorded call
   drops the memo before it propagates; arming a later step replaces a memo that was never consumed.

The slot lives in a ``ContextVar``, so concurrent API workers never see each other's records.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

import torch


class CallRecord:
    """One main-pass ``run_inner_model`` call: rows ``[start, start + rows)`` of the step's CFG batch."""

    __slots__ = ("start", "rows", "x", "sigma", "cond", "row_subset_ok", "_sealed")

    def __init__(self, start: int, x: torch.Tensor, sigma: torch.Tensor, cond):
        self.start = start
        self.rows = int(x.shape[0])
        self.x = x
        self.sigma = sigma
        self.cond = cond
        # A claimant clears this when rows of this call are coupled (shared attention, per-call RNG over
        # all rows, caches sized by the first call), so a replay must evaluate every row of the call.
        self.row_subset_ok = True
        self._sealed = None

    def _input_tensors(self):
        tensors = [self.x, self.sigma]
        values = self.cond.values() if isinstance(self.cond, dict) else (self.cond,)
        for value in values:
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, (list, tuple)):
                tensors.extend(item for item in value if isinstance(item, torch.Tensor))
        return tensors

    def seal(self) -> None:
        """Snapshot input identity and version counters once the main call returned.

        The model's own in-place input edits (dtype casts of the cond dict, ControlNet Revision writing
        ``y``) happen during the call and are part of the snapshot.
        """
        self._sealed = [(tensor, tensor._version) for tensor in self._input_tensors()]

    def inputs_unchanged(self) -> bool:
        """True when no input was replaced or modified in place since ``seal``. Never synchronizes."""
        if self._sealed is None:
            return False
        current = self._input_tensors()
        return len(current) == len(self._sealed) and all(
            tensor is sealed and tensor._version == version
            for tensor, (sealed, version) in zip(current, self._sealed)
        )


class MainPassMemo:
    """The ordered main-pass calls of one denoiser step; ``n_cond`` leading rows are conditional."""

    def __init__(self, n_cond: int):
        if n_cond <= 0:
            raise ValueError(f"main-pass memo needs a positive cond row count, got {n_cond}")
        self.n_cond = int(n_cond)
        self.calls: list[CallRecord] = []
        self.rows = 0

    def add_call(self, x: torch.Tensor, sigma: torch.Tensor, cond) -> CallRecord:
        if self.calls:
            first = self.calls[0].x
            # Every A1111 CFG layout calls row slices of one x_in in order; anything else would make the
            # running row offset meaningless.
            if (
                x.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
                or x.stride() != first.stride()
                or x.storage_offset() != first.storage_offset() + self.rows * first.stride(0)
            ):
                raise RuntimeError(
                    "main-pass UNet call is not the next row slice of the step's CFG batch; "
                    "an extension called run_inner_model out of order while PAG was recording"
                )
        record = CallRecord(self.rows, x, sigma, cond)
        self.rows += record.rows
        self.calls.append(record)
        return record

    def clear(self) -> None:
        self.calls = []
        self.rows = 0


class MainPassRecorder:
    """Instance-level wrapper over ``CFGDenoiser.run_inner_model`` that records calls while armed."""

    def __init__(self, original):
        self.original = original
        self.memo: MainPassMemo | None = None

    def __call__(self, x, sigma, cond):
        memo = self.memo
        if memo is None:
            return self.original(x, sigma, cond)
        record = memo.add_call(x, sigma, cond)
        try:
            with recording(record):
                out = self.original(x, sigma, cond)
        except BaseException:
            self.memo = None
            memo.clear()
            raise
        record.seal()
        return out


def arm(denoiser, n_cond: int) -> MainPassMemo:
    """Start recording ``denoiser``'s main-pass calls for this step, replacing any unconsumed memo."""
    recorder = denoiser.__dict__.get("run_inner_model")
    if not isinstance(recorder, MainPassRecorder):
        recorder = MainPassRecorder(denoiser.run_inner_model)
        denoiser.run_inner_model = recorder
    recorder.memo = MainPassMemo(n_cond)
    return recorder.memo


def disarm(denoiser) -> MainPassMemo | None:
    """Stop recording and hand the step's memo to the caller, which owns clearing it."""
    recorder = denoiser.__dict__.get("run_inner_model") if denoiser is not None else None
    if not isinstance(recorder, MainPassRecorder):
        return None
    memo, recorder.memo = recorder.memo, None
    return memo


def uninstall(denoiser) -> bool:
    """Remove the recorder from ``denoiser``.

    Returns False, leaving everything in place, when another wrapper replaced the instance's
    ``run_inner_model`` after ``arm``; a disarmed recorder below it only passes calls through.
    """
    recorder = denoiser.__dict__.get("run_inner_model")
    if recorder is None:
        return True
    if not isinstance(recorder, MainPassRecorder):
        return False
    recorder.memo = None
    del denoiser.run_inner_model
    return True


class Slot:
    """Per-call claim token. ``rows`` is the number of rows the call evaluates."""

    __slots__ = ("rec", "rows", "replay", "claimed")

    def __init__(self, rec: CallRecord, rows: int, replay: bool):
        self.rec = rec
        self.rows = rows
        self.replay = replay
        self.claimed = False


_SLOT: contextvars.ContextVar[Slot | None] = contextvars.ContextVar("openclaw_unet_row_memo_slot", default=None)


def claim() -> Slot | None:
    """Return the current call's slot to the first (outermost) UNet forward that asks, else None."""
    slot = _SLOT.get()
    if slot is None or slot.claimed:
        return None
    slot.claimed = True
    return slot


@contextmanager
def _slot(slot: Slot):
    token = _SLOT.set(slot)
    try:
        yield slot
    finally:
        _SLOT.reset(token)


def recording(rec: CallRecord):
    return _slot(Slot(rec, rec.rows, replay=False))


def replaying(rec: CallRecord, rows: int):
    if not 0 < rows <= rec.rows:
        raise ValueError(f"cannot replay {rows} rows of a {rec.rows}-row main-pass call")
    return _slot(Slot(rec, rows, replay=True))


def _slice_rows(value, rows: int, m: int):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != rows:
            raise ValueError(f"conditioning tensor of shape {tuple(value.shape)} is not batch-aligned with a {rows}-row UNet call")
        return value[:m]
    if isinstance(value, list):
        return [_slice_rows(item, rows, m) for item in value]
    if isinstance(value, tuple):
        return tuple(_slice_rows(item, rows, m) for item in value)
    return value


def slice_cond_rows(cond, rows: int, m: int):
    """Leading ``m`` rows of a ``rows``-row conditioning dict (fresh containers, tensor views)."""
    if isinstance(cond, dict):
        return {key: _slice_rows(value, rows, m) for key, value in cond.items()}
    return _slice_rows(cond, rows, m)


# Attribute names set by extensions-builtin/hypertile on the diffusion wrapper and on each wrapped layer.
_HYPERTILE_LAYERS = "__webui_hypertile_layers"
_HYPERTILE_PARAMS = "__webui_hypertile_params"


def hypertile_unet_enabled(diffusion_wrapper) -> bool:
    """True when a hypertile-wrapped U-Net attention layer under ``diffusion_wrapper`` is enabled.

    Hypertile draws its tile layout from a private RNG on every wrapped attention call, so adding or
    skipping UNet calls (or encoder layers) while it is enabled shifts every later tile choice.
    """
    layers = getattr(diffusion_wrapper, _HYPERTILE_LAYERS, None)
    if not layers:
        return False
    for name in layers:
        params = getattr(diffusion_wrapper.get_submodule(name), _HYPERTILE_PARAMS, None)
        if params is not None and params.enabled:
            return True
    return False
