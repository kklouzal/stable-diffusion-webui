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
3. While a recorded call runs, the outermost UNet forward that understands the memo ("producer") claims
   its slot (``claim``). Nested forwards (ControlNet reference passes, CUDA-graph warm-up/capture) see the
   claimed slot and do nothing. A producer may flag the call as row-coupled (``row_subset_ok = False``) and,
   on an eager inference call on the recording stream that is not being captured into a CUDA graph
   (``can_record``), attach a ``Prefix``: references to work a replay of the same rows can reuse.
4. The consumer takes the memo (``disarm``) in its ``cfg_denoised`` callback, replays rows ``[0, m)`` of
   each record inside ``replaying`` and clears the memo in ``finally``. The producer that claims a replay
   slot reuses the prefix only after ``replay_prefix`` checked that the replay is fed the recorded inputs;
   otherwise it computes everything. An exception inside a recorded call drops the memo before it
   propagates; arming a later step replaces a memo that was never consumed.

The slot lives in a ``ContextVar``, so concurrent API workers never see each other's records.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

import torch


class CallRecord:
    """One main-pass ``run_inner_model`` call: rows ``[start, start + rows)`` of the step's CFG batch."""

    __slots__ = ("start", "rows", "cond_rows", "x", "sigma", "cond", "row_subset_ok", "prefix", "_sealed")

    def __init__(self, start: int, x: torch.Tensor, sigma: torch.Tensor, cond, cond_rows: int = 0):
        self.start = start
        self.rows = int(x.shape[0])
        self.cond_rows = cond_rows
        self.x = x
        self.sigma = sigma
        self.cond = cond
        # A claimant clears this when rows of this call are coupled (shared attention, per-call RNG over
        # all rows, caches sized by the first call), so a replay must evaluate every row of the call.
        self.row_subset_ok = True
        self.prefix: Prefix | None = None
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
        """Snapshot input identity (and version counters where tracked) once the main call returned.

        The model's own in-place input edits (dtype casts of the cond dict, ControlNet Revision writing
        ``y``) happen during the call and are part of the snapshot.
        """
        self._sealed = [(tensor, tensor_version(tensor)) for tensor in self._input_tensors()]

    def inputs_unchanged(self) -> bool:
        """True when no input was replaced, or modified in place where detectable, since ``seal``.

        Never synchronizes. Generation runs under ``torch.inference_mode()``, whose tensors have no version
        counter, so there only replacement is detected. No WebUI callback gets the main-pass inputs between
        the main call and the PAG pass (``cfg_denoised`` receives the outputs), so in-place edits there
        would need a reference kept from ``cfg_denoiser``.
        """
        if self._sealed is None:
            return False
        current = self._input_tensors()
        return len(current) == len(self._sealed) and all(
            tensor is sealed and tensor_version(tensor) == version
            for tensor, (sealed, version) in zip(current, self._sealed)
        )


def tensor_version(tensor: torch.Tensor):
    """In-place version counter, or None for inference tensors (they have none; reading it raises)."""
    return None if tensor.is_inference() else tensor._version


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
        cond_rows = max(0, min(int(x.shape[0]), self.n_cond - self.rows))
        record = CallRecord(self.rows, x, sigma, cond, cond_rows)
        self.rows += record.rows
        self.calls.append(record)
        return record

    def clear(self) -> None:
        for record in self.calls:
            record.prefix = None
        self.calls = []
        self.rows = 0


class MainPassRecorder:
    """Instance-level wrapper over ``CFGDenoiser.run_inner_model`` that records calls while armed.

    ``replaced`` is the instance attribute it shadows (another wrapper), or None for the class method.
    """

    def __init__(self, original, replaced):
        self.original = original
        self.replaced = replaced
        self.memo: MainPassMemo | None = None

    def __call__(self, x, sigma, cond):
        memo = self.memo
        if memo is None:
            return self.original(x, sigma, cond)
        try:
            record = memo.add_call(x, sigma, cond)
            with recording(record):
                out = self.original(x, sigma, cond)
            record.seal()
        except BaseException:
            self.memo = None
            memo.clear()
            raise
        return out


def arm(denoiser, n_cond: int) -> MainPassMemo:
    """Start recording ``denoiser``'s main-pass calls for this step, replacing any unconsumed memo."""
    recorder = denoiser.__dict__.get("run_inner_model")
    if not isinstance(recorder, MainPassRecorder):
        recorder = MainPassRecorder(denoiser.run_inner_model, recorder)
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
    if recorder.replaced is not None:
        denoiser.run_inner_model = recorder.replaced
    else:
        del denoiser.run_inner_model
    return True


class Prefix:
    """Producer payload kept from a recorded call: references to its tensors, never copies.

    ``owner`` identifies the producer and module that may reuse it. ``context`` and ``y`` are the
    conditioning tensors the recorded UNet call received; a replay must receive row-prefix views of the
    same tensors. ``state`` is producer-specific. Tensors in it whose leading dim is ``rows`` are cut to the
    replay's rows with ``rows_of``.
    """

    __slots__ = ("owner", "rows", "x_shape", "x_dtype", "x_device", "context", "y", "state")

    def __init__(self, owner, x: torch.Tensor, context, y, state):
        self.owner = owner
        self.rows = int(x.shape[0])
        self.x_shape = tuple(x.shape)
        self.x_dtype = x.dtype
        self.x_device = x.device
        self.context = context
        self.y = y
        self.state = state


class Slot:
    """Per-call claim token. ``rows`` is the number of rows the call evaluates."""

    __slots__ = ("rec", "rows", "replay", "stream", "claimed")

    def __init__(self, rec: CallRecord, rows: int, replay: bool):
        self.rec = rec
        self.rows = rows
        self.replay = replay
        # The stream the recorded call started on; tensors made on another stream (CUDA-graph warm-up) are never kept.
        self.stream = torch.cuda.current_stream(rec.x.device) if rec.x.is_cuda else None
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
    if rec.prefix is not None and not rec.inputs_unchanged():
        # Inputs were replaced or edited in place after the main call: nothing recorded may be reused.
        rec.prefix = None
    return _slot(Slot(rec, rows, replay=True))


def can_record(slot: Slot | None, x: torch.Tensor) -> bool:
    """Whether the producer holding ``slot`` may keep tensors of this call for a replay.

    Only eager inference calls of rows the consumer needs, on the stream the recorded call started on and
    outside CUDA-graph capture: graph warm-up runs on a side stream and capture allocates from the graph's
    private pool, neither of which may outlive the call.
    """
    if slot is None or slot.replay or slot.rec.cond_rows == 0 or torch.is_grad_enabled():
        return False
    if x.is_cuda:
        if torch.cuda.is_current_stream_capturing():
            return False
        if slot.stream is None or torch.cuda.current_stream(x.device) != slot.stream:
            return False
    return True


def _same_tensor_rows(t, ref, rows: int) -> bool:
    """``t`` is the first ``rows`` rows of ``ref`` (same storage, layout and dtype), or both are None."""
    if t is None or ref is None:
        return t is None and ref is None
    return (
        isinstance(t, torch.Tensor)
        and isinstance(ref, torch.Tensor)
        and t.data_ptr() == ref.data_ptr()
        and t.dtype == ref.dtype
        and t.shape[0] == rows
        and tuple(t.shape[1:]) == tuple(ref.shape[1:])
        and t.stride() == ref.stride()
    )


def attach_prefix(slot: Slot, prefix: Prefix) -> None:
    """Keep ``prefix`` for the replay if the UNet received the recorded call's own conditioning.

    A wrapper between the denoiser and the UNet that builds new conditioning per call (tiling, regional
    prompting) fails this check, so its calls are always recomputed.
    """
    cond = slot.rec.cond if isinstance(slot.rec.cond, dict) else {}
    if _same_tensor_rows(prefix.context, cond.get("crossattn"), slot.rec.rows) and _same_tensor_rows(prefix.y, cond.get("vector"), slot.rec.rows):
        slot.rec.prefix = prefix


def replay_prefix(slot: Slot, owner, x: torch.Tensor, context, y) -> Prefix | None:
    """The recorded prefix a replay call may reuse, or None to compute everything.

    Raises when the replay is fed the recorded conditioning but its shape contradicts the recording.
    """
    prefix = slot.rec.prefix
    if prefix is None or prefix.owner != owner:
        return None
    m = slot.rows
    if x.shape[0] != m:
        raise RuntimeError(f"UNet replay received {x.shape[0]} rows, the memo slot expects {m}")
    if not (_same_tensor_rows(context, prefix.context, m) and _same_tensor_rows(y, prefix.y, m)):
        return None
    if tuple(x.shape[1:]) != prefix.x_shape[1:] or x.dtype != prefix.x_dtype or x.device != prefix.x_device:
        raise RuntimeError(
            f"UNet replay input {tuple(x.shape)} {x.dtype} {x.device} does not match the recorded call "
            f"{prefix.x_shape} {prefix.x_dtype} {prefix.x_device}"
        )
    return prefix


def rows_of(t, rows: int, m: int):
    """First ``m`` rows of a recorded per-row tensor; single-row (broadcast) tensors and non-tensors pass through."""
    if not isinstance(t, torch.Tensor) or t.ndim == 0 or (t.shape[0] == 1 and rows != 1):
        return t
    if t.shape[0] != rows:
        raise RuntimeError(f"recorded tensor of shape {tuple(t.shape)} is not aligned with the {rows}-row call")
    return t[:m]


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
