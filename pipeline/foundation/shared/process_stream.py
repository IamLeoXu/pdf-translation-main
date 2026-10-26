"""Stream child output on the caller thread while preserving timeout/error tails."""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence


def run_streamed_command(
    command: Sequence[str], *, timeout: float | None = None,
    cwd=None, env: dict[str, str] | None = None,
    on_stdout: Callable[[str], None] | None = None,
    on_stderr: Callable[[str], None] | None = None,
    on_tick: Callable[[], None] | None = None, tick_interval: float = 15.0,
    max_capture_chars: int = 65536, process_group: bool = False,
) -> subprocess.CompletedProcess:
    """Callbacks run on this thread so ContextVar event writers remain active."""
    grouped = bool(process_group and os.name == "posix")
    process = subprocess.Popen(
        list(command), cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        bufsize=1, start_new_session=grouped,
    )
    messages: queue.Queue = queue.Queue()
    captured = {"stdout": "", "stderr": ""}
    def read_pipe(name, pipe):
        try:
            for line in iter(pipe.readline, ""):
                messages.put((name, line))
        finally:
            messages.put((name, None))
    readers = [threading.Thread(target=read_pipe, args=(name, pipe), daemon=True)
               for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr))]
    for reader in readers:
        reader.start()
    started = time.monotonic()
    next_tick = started + max(0.05, tick_interval)
    closed = 0
    def stop():
        if process.poll() is None:
            if grouped:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
        process.wait()
    try:
        while closed < 2:
            now = time.monotonic()
            if timeout is not None and now - started >= timeout:
                stop()
                raise subprocess.TimeoutExpired(list(command), timeout, output=captured["stdout"], stderr=captured["stderr"])
            if on_tick is not None and now >= next_tick:
                on_tick()
                next_tick = now + max(0.05, tick_interval)
            try:
                name, line = messages.get(timeout=0.05)
            except queue.Empty:
                continue
            if line is None:
                closed += 1
                continue
            captured[name] = (captured[name] + line)[-max_capture_chars:]
            callback = on_stdout if name == "stdout" else on_stderr
            if callback is not None:
                callback(line)
            else:
                stream = sys.stdout if name == "stdout" else sys.stderr
                print(line, end="", file=stream, flush=True)
        remaining = None if timeout is None else max(0.001, timeout - (time.monotonic() - started))
        code = process.wait(timeout=remaining)
        return subprocess.CompletedProcess(list(command), code, captured["stdout"], captured["stderr"])
    except BaseException:
        stop()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=1)
        process.stdout.close()
        process.stderr.close()
