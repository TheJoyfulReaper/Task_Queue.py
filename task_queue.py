"""
task_queue.py — In-Process Priority Task Queue with Worker Pool

Demonstrates:
  - Thread-safe priority queue with dead-letter handling
  - Worker pool pattern with graceful shutdown
  - Decorator-based task registration
  - CLI interface via argparse
  - Context managers for resource lifecycle
  - Comprehensive error handling and retry semantics
  - Dataclasses, enums, protocols, and type hints

Author: Christopher Hall
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, auto
from typing import Any, Callable, Protocol

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("taskq")


# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------

class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskState(IntEnum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    DEAD_LETTERED = auto()


@dataclass(order=True)
class Task:
    """
    Sortable task object — priority is the primary sort key,
    followed by creation timestamp for FIFO within the same priority.
    """
    priority: Priority = field(compare=True)
    created_at: float = field(compare=True, default_factory=time.monotonic)
    task_id: str = field(compare=False, default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = field(compare=False, default="")
    payload: dict[str, Any] = field(compare=False, default_factory=dict)
    state: TaskState = field(compare=False, default=TaskState.PENDING)
    attempts: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=3)
    result: Any = field(compare=False, default=None)
    error: str | None = field(compare=False, default=None)
    started_at: float | None = field(compare=False, default=None)
    completed_at: float | None = field(compare=False, default=None)

    @property
    def elapsed_ms(self) -> float | None:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "priority": self.priority.name,
            "state": self.state.name,
            "attempts": self.attempts,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Task Handler Protocol & Registry
# ---------------------------------------------------------------------------

class TaskHandler(Protocol):
    def __call__(self, payload: dict[str, Any]) -> Any: ...


class TaskRegistry:
    """Decorator-based handler registry — maps task names to callables."""

    _handlers: dict[str, TaskHandler] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(fn: TaskHandler) -> TaskHandler:
            cls._handlers[name] = fn
            log.debug("Registered handler: %s", name)
            return fn
        return decorator

    @classmethod
    def get(cls, name: str) -> TaskHandler | None:
        return cls._handlers.get(name)

    @classmethod
    def list_handlers(cls) -> list[str]:
        return sorted(cls._handlers.keys())


# ---------------------------------------------------------------------------
# Thread-Safe Priority Queue
# ---------------------------------------------------------------------------

class PriorityTaskQueue:
    """
    Min-heap backed priority queue with thread-safe access.
    Supports blocking get with timeout and graceful drain.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._heap: list[Task] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._maxsize = maxsize

    def put(self, task: Task) -> None:
        with self._not_empty:
            if self._maxsize > 0 and len(self._heap) >= self._maxsize:
                raise RuntimeError("Queue is full")
            heapq.heappush(self._heap, task)
            self._not_empty.notify()

    def get(self, timeout: float | None = None) -> Task | None:
        with self._not_empty:
            while not self._heap:
                if not self._not_empty.wait(timeout=timeout):
                    return None  # timed out
            return heapq.heappop(self._heap)

    def qsize(self) -> int:
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._heap) == 0


# ---------------------------------------------------------------------------
# Dead Letter Queue
# ---------------------------------------------------------------------------

class DeadLetterQueue:
    """Stores tasks that exhausted all retries for later inspection."""

    def __init__(self) -> None:
        self._tasks: list[Task] = []
        self._lock = threading.Lock()

    def push(self, task: Task) -> None:
        with self._lock:
            task.state = TaskState.DEAD_LETTERED
            self._tasks.append(task)
            log.warning("Dead-lettered task %s (%s): %s", task.task_id, task.name, task.error)

    def drain(self) -> list[Task]:
        with self._lock:
            tasks = list(self._tasks)
            self._tasks.clear()
            return tasks

    def size(self) -> int:
        with self._lock:
            return len(self._tasks)


# ---------------------------------------------------------------------------
# Worker Pool
# ---------------------------------------------------------------------------

class WorkerPool:
    """
    Fixed-size thread pool that processes tasks from a PriorityTaskQueue.

    Features:
      - Retry with configurable max attempts per task
      - Dead letter queue for permanently failed tasks
      - Graceful shutdown with drain timeout
      - Per-worker metrics collection
    """

    def __init__(
        self,
        queue: PriorityTaskQueue,
        dlq: DeadLetterQueue,
        num_workers: int = 4,
        poll_interval: float = 0.5,
    ) -> None:
        self._queue = queue
        self._dlq = dlq
        self._num_workers = num_workers
        self._poll_interval = poll_interval
        self._workers: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._completed: list[Task] = []
        self._completed_lock = threading.Lock()

    @contextmanager
    def running(self):
        """Context manager — starts workers on enter, shuts down on exit."""
        self.start()
        try:
            yield self
        finally:
            self.shutdown()

    def start(self) -> None:
        log.info("Starting %d workers…", self._num_workers)
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def shutdown(self, timeout: float = 10.0) -> None:
        log.info("Shutting down worker pool…")
        self._stop_event.set()
        deadline = time.monotonic() + timeout
        for t in self._workers:
            remaining = max(0, deadline - time.monotonic())
            t.join(timeout=remaining)
        log.info("All workers stopped.")

    @property
    def completed_tasks(self) -> list[Task]:
        with self._completed_lock:
            return list(self._completed)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            task = self._queue.get(timeout=self._poll_interval)
            if task is None:
                continue
            self._execute(task)

    def _execute(self, task: Task) -> None:
        handler = TaskRegistry.get(task.name)
        if handler is None:
            task.error = f"No handler registered for '{task.name}'"
            task.state = TaskState.FAILED
            self._dlq.push(task)
            return

        task.attempts += 1
        task.state = TaskState.RUNNING
        task.started_at = time.monotonic()

        try:
            result = handler(task.payload)
            task.completed_at = time.monotonic()
            task.state = TaskState.COMPLETED
            task.result = result
            log.info(
                "✓ %s [%s] completed in %.1fms",
                task.task_id,
                task.name,
                task.elapsed_ms,
            )
        except Exception as exc:
            task.completed_at = time.monotonic()
            task.error = str(exc)
            if task.attempts < task.max_retries:
                task.state = TaskState.PENDING
                log.warning(
                    "⟳ %s [%s] failed (attempt %d/%d): %s — re-queuing",
                    task.task_id,
                    task.name,
                    task.attempts,
                    task.max_retries,
                    exc,
                )
                self._queue.put(task)
                return
            else:
                task.state = TaskState.FAILED
                self._dlq.push(task)
                return

        with self._completed_lock:
            self._completed.append(task)


# ---------------------------------------------------------------------------
# Example Task Handlers
# ---------------------------------------------------------------------------

@TaskRegistry.register("send_email")
def handle_send_email(payload: dict[str, Any]) -> dict:
    """Simulate sending an email with random latency."""
    recipient = payload.get("to", "unknown")
    subject = payload.get("subject", "(no subject)")
    # Simulate network I/O
    latency = 0.05 + (hash(recipient) % 100) / 1000
    time.sleep(latency)
    # Simulate occasional failures (10% rate for demo)
    if hash(f"{recipient}{time.monotonic()}") % 10 == 0:
        raise ConnectionError(f"SMTP timeout sending to {recipient}")
    return {"delivered": True, "recipient": recipient, "subject": subject}


@TaskRegistry.register("process_image")
def handle_process_image(payload: dict[str, Any]) -> dict:
    """Simulate CPU-bound image processing."""
    filename = payload.get("filename", "unknown.png")
    width = payload.get("width", 1920)
    height = payload.get("height", 1080)
    # Simulate processing time proportional to pixel count
    pixels = width * height
    time.sleep(pixels / 50_000_000)  # ~40ms for 1080p
    return {"filename": filename, "processed": True, "pixels": pixels}


@TaskRegistry.register("sync_inventory")
def handle_sync_inventory(payload: dict[str, Any]) -> dict:
    """Simulate syncing inventory with an external API."""
    warehouse = payload.get("warehouse", "default")
    item_count = payload.get("items", 100)
    time.sleep(0.1)
    return {"warehouse": warehouse, "synced_items": item_count}


@TaskRegistry.register("generate_report")
def handle_generate_report(payload: dict[str, Any]) -> dict:
    """Simulate report generation."""
    report_type = payload.get("type", "monthly")
    time.sleep(0.2)
    return {
        "report_type": report_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pages": 42,
    }


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def run_demo(num_tasks: int, num_workers: int) -> None:
    """Run an interactive demo of the task queue system."""
    queue = PriorityTaskQueue(maxsize=1000)
    dlq = DeadLetterQueue()

    # Enqueue a mix of tasks at various priorities
    task_templates = [
        ("send_email", Priority.HIGH, {"to": "alice@example.com", "subject": "Q3 Report"}),
        ("send_email", Priority.NORMAL, {"to": "bob@example.com", "subject": "Meeting Notes"}),
        ("process_image", Priority.LOW, {"filename": "hero_banner.png", "width": 3840, "height": 2160}),
        ("process_image", Priority.NORMAL, {"filename": "thumbnail.jpg", "width": 640, "height": 480}),
        ("sync_inventory", Priority.CRITICAL, {"warehouse": "ATX-01", "items": 500}),
        ("sync_inventory", Priority.HIGH, {"warehouse": "ORL-02", "items": 250}),
        ("generate_report", Priority.BACKGROUND, {"type": "annual"}),
        ("generate_report", Priority.NORMAL, {"type": "weekly"}),
    ]

    log.info("Enqueueing %d tasks…", num_tasks)
    for i in range(num_tasks):
        name, prio, payload = task_templates[i % len(task_templates)]
        task = Task(name=name, priority=prio, payload=payload)
        queue.put(task)

    log.info("Queue size: %d", queue.qsize())

    with WorkerPool(queue, dlq, num_workers=num_workers).running() as pool:
        # Wait for queue to drain
        while not queue.is_empty():
            time.sleep(0.1)
        # Give workers time to finish in-flight tasks
        time.sleep(0.5)

    # Summary
    completed = pool.completed_tasks
    dead = dlq.drain()

    print("\n" + "=" * 64)
    print("  Task Queue Demo — Results Summary")
    print("=" * 64)
    print(f"  Total enqueued   : {num_tasks}")
    print(f"  Completed        : {len(completed)}")
    print(f"  Dead-lettered    : {len(dead)}")
    if completed:
        avg = sum(t.elapsed_ms or 0 for t in completed) / len(completed)
        print(f"  Avg latency      : {avg:.1f}ms")
    print("-" * 64)

    if dead:
        print("\n  Dead Letter Queue:")
        for t in dead[:5]:
            print(f"    {t.task_id} [{t.name}] — {t.error}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Priority Task Queue with Worker Pool — Portfolio Demo"
    )
    parser.add_argument(
        "--tasks", type=int, default=24, help="Number of tasks to enqueue"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )
    parser.add_argument(
        "--list-handlers",
        action="store_true",
        help="List all registered task handlers",
    )
    args = parser.parse_args()

    if args.list_handlers:
        print("Registered handlers:")
        for name in TaskRegistry.list_handlers():
            print(f"  • {name}")
        return

    run_demo(num_tasks=args.tasks, num_workers=args.workers)


if __name__ == "__main__":
    main()
