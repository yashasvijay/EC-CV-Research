import queue
from typing import TypeVar

T = TypeVar("T")


def _offer_latest(destination: queue.Queue[T], item: T) -> T | None:
    # One producer per queue. Consumers may race; Empty/Full are authoritative.
    try:
        destination.put_nowait(item)
        return None
    except queue.Full:
        try:
            evicted = destination.get_nowait()
            destination.task_done()
        except queue.Empty:
            evicted = None
        destination.put_nowait(item)
        return evicted
