import threading
import collections


# reference: https://gist.github.com/vitaliyp/6d54dd76ca2c3cdfc1149d33007dc34a
class FIFOLock(object):
    """A reentrant lock granted in arrival order.

    release() hands the lock directly to the oldest waiter, so a thread arriving meanwhile cannot take it first
    (the previous version released and let the waiter re-acquire, which a newcomer could win). The owning thread
    may acquire it again (each acquire needs a release): e.g. a settings request holding queue_lock runs option
    onchange callbacks that are themselves wrapped with call_queue.wrap_queued_call."""

    def __init__(self):
        self._lock = threading.Lock()
        self._inner_lock = threading.Lock()
        self._pending_threads = collections.deque()
        self._owner = None
        self._count = 0

    def acquire(self, blocking=True):
        me = threading.get_ident()
        with self._inner_lock:
            if self._owner == me:
                self._count += 1
                return True

            if self._lock.acquire(False):
                self._owner, self._count = me, 1
                return True
            elif not blocking:
                return False

            release_event = threading.Event()
            self._pending_threads.append(release_event)

        try:
            release_event.wait()  # the releasing thread kept _lock held and handed it to us
        except BaseException:
            with self._inner_lock:
                if release_event in self._pending_threads:
                    self._pending_threads.remove(release_event)
                else:  # handed to us just as the wait was interrupted: pass it on
                    self._hand_off_locked()
            raise

        with self._inner_lock:
            self._owner, self._count = me, 1
        return True

    def _hand_off_locked(self):
        self._owner = None
        if self._pending_threads:
            self._pending_threads.popleft().set()
        else:
            self._lock.release()

    def release(self):
        with self._inner_lock:
            if self._owner != threading.get_ident():
                raise RuntimeError("cannot release un-acquired lock")

            self._count -= 1
            if not self._count:
                self._hand_off_locked()

    __enter__ = acquire

    def __exit__(self, t, v, tb):
        self.release()
