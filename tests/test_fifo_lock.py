"""queue_lock (FIFOLock): arrival-order handoff, reentrancy for its owner, and no orphaned lock on an interrupted wait."""
import threading
import time
from types import SimpleNamespace

import pytest

from modules import call_queue, fifo_lock


def _try_from_other_thread(lock):
    result = []
    thread = threading.Thread(target=lambda: result.append(lock.acquire(blocking=False)))
    thread.start()
    thread.join()
    return result[0]


def _wait_for_waiters(lock, count):
    deadline = time.monotonic() + 10
    while len(lock._pending_threads) < count:
        assert time.monotonic() < deadline, "waiter never queued"
        time.sleep(0.001)


def test_owner_can_reacquire_and_others_wait_until_the_last_release():
    lock = fifo_lock.FIFOLock()
    with lock:
        with lock:  # before: a nested acquire by the owner deadlocked
            assert not _try_from_other_thread(lock)
        assert not _try_from_other_thread(lock)
    assert _try_from_other_thread(lock)  # the probe thread now holds it and exits without releasing


def test_queued_call_inside_a_held_queue_lock_does_not_deadlock():
    # Api.set_config holds queue_lock while option onchange callbacks run wrapped by wrap_queued_call.
    done = threading.Event()
    lock = fifo_lock.FIFOLock()

    def run():
        with lock:
            with lock:
                done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert done.wait(5)
    assert callable(call_queue.wrap_queued_call(lambda: None))


def test_release_by_a_thread_that_does_not_hold_it_raises():
    lock = fifo_lock.FIFOLock()
    with pytest.raises(RuntimeError):
        lock.release()
    lock.acquire()
    errors = []

    def other():
        try:
            lock.release()
        except RuntimeError as e:
            errors.append(e)

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert errors
    lock.release()


def test_release_hands_the_lock_to_the_oldest_waiter_not_to_a_newcomer():
    lock = fifo_lock.FIFOLock()
    order = []
    lock.acquire()

    def waiter(name):
        with lock:
            order.append(name)

    first = threading.Thread(target=waiter, args=("first",))
    first.start()
    _wait_for_waiters(lock, 1)
    second = threading.Thread(target=waiter, args=("second",))
    second.start()
    _wait_for_waiters(lock, 2)

    lock.release()
    # A thread arriving right after the release must queue behind the waiters (before: it usually won the race).
    assert not lock.acquire(blocking=False)
    first.join(10)
    second.join(10)
    assert order == ["first", "second"]
    assert lock.acquire(blocking=False)
    lock.release()


class _InterruptedEvent:
    """An Event whose wait() is interrupted, either before or after the lock was handed over."""

    def __init__(self, after_handoff):
        self._event = threading.Event()
        self.after_handoff = after_handoff

    def set(self):
        self._event.set()

    def wait(self):
        if self.after_handoff:
            self._event.wait()
        raise KeyboardInterrupt


@pytest.mark.parametrize("after_handoff", [False, True])
def test_interrupted_waiter_never_orphans_the_lock(monkeypatch, after_handoff):
    lock = fifo_lock.FIFOLock()
    lock.acquire()
    # Only fifo_lock's view of threading: the real module's Event backs every Thread.
    monkeypatch.setattr(fifo_lock, "threading", SimpleNamespace(Event=lambda: _InterruptedEvent(after_handoff), get_ident=threading.get_ident))
    interrupted = []

    def waiter():
        try:
            lock.acquire()
        except KeyboardInterrupt:
            interrupted.append(True)

    thread = threading.Thread(target=waiter)
    thread.start()
    if not after_handoff:
        thread.join(10)
    else:
        _wait_for_waiters(lock, 1)
    lock.release()
    thread.join(10)
    assert interrupted == [True]
    assert not lock._pending_threads
    assert _try_from_other_thread(lock)
