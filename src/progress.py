"""CLI progress display utilities.

Responsibilities:
    - Show the start/finish of every pipeline step with a phase marker.
    - Measure elapsed time automatically and report it as part of the summary.
    - Auto-downgrade to plain text when stderr is not a TTY.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator


# ANSI colour codes (only emitted when stderr is a TTY).
class _Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def _supports_color() -> bool:
    """True when stderr is a TTY (assumed to support ANSI)."""
    return sys.stderr.isatty()


def _colorize(text: str, color: str) -> str:
    """Wrap `text` in ANSI codes when the terminal supports it."""
    if _supports_color():
        return f"{color}{text}{_Color.RESET}"
    return text


class ProgressReporter:
    """Phase-by-phase progress reporter. Prints `[current/total] <description>`."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.current_step = 0
        self._phase_start: float = 0.0
        self._overall_start: float = time.monotonic()

    def _step_prefix(self) -> str:
        label = f"[{self.current_step}/{self.total_steps}]"
        return _colorize(label, _Color.BOLD + _Color.CYAN)

    def _check_mark(self) -> str:
        return _colorize("✓", _Color.GREEN)

    def _cross_mark(self) -> str:
        return _colorize("✗", _Color.RED)

    @contextmanager
    def step(self, description: str) -> Iterator["_StepContext"]:
        """Run a step as a context manager; auto-reports start and finish.

        Usage::

            with reporter.step("Loading CSV") as step:
                df = load_csv(...)
                step.set_summary(f"{len(df)} rows")
        """
        self.current_step += 1
        self._phase_start = time.monotonic()
        print(
            f"{self._step_prefix()} {description}...",
            file=sys.stderr,
            flush=True,
        )
        ctx = _StepContext(self)
        try:
            yield ctx
        except BaseException:
            elapsed = time.monotonic() - self._phase_start
            print(
                f"      {self._cross_mark()} failed ({elapsed:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            raise
        else:
            elapsed = time.monotonic() - self._phase_start
            suffix = f" {ctx.summary}" if ctx.summary else ""
            print(
                f"      {self._check_mark()} done ({elapsed:.1f}s){suffix}",
                file=sys.stderr,
                flush=True,
            )

    def total_elapsed(self) -> float:
        """Seconds elapsed since the ProgressReporter was created."""
        return time.monotonic() - self._overall_start

    def banner(self, title: str) -> None:
        """Print a prominent title banner at the start of a run."""
        bar = "=" * 60
        print(file=sys.stderr)
        print(_colorize(bar, _Color.DIM), file=sys.stderr)
        print(_colorize(f" {title}", _Color.BOLD), file=sys.stderr)
        print(_colorize(bar, _Color.DIM), file=sys.stderr, flush=True)

    def footer(self, message: str) -> None:
        """Print the final footer once all steps have finished."""
        elapsed = self.total_elapsed()
        print(file=sys.stderr)
        print(
            _colorize(
                f"All steps complete: {message} (total {elapsed:.1f}s)",
                _Color.BOLD + _Color.GREEN,
            ),
            file=sys.stderr,
            flush=True,
        )


class _StepContext:
    """Handle handed to the `step` block so callers can add detail info."""

    def __init__(self, owner: ProgressReporter) -> None:
        self._owner = owner
        self.summary: str = ""

    def set_summary(self, text: str) -> None:
        """Text appended to the ✓ line once the step finishes."""
        self.summary = text

    def detail(self, text: str) -> None:
        """Print an indented sub-line mid-step."""
        print(f"      {_colorize('·', _Color.DIM)} {text}", file=sys.stderr, flush=True)
