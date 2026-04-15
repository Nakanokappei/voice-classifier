"""CLI progress display utilities.

Design goal: keep the terminal as quiet as possible.

- Banner + final footer are printed normally.
- Each step occupies exactly one visible line in the final scrollback:

      [3/5] Search clustering candidates — selected kmeans k=10 (0.7s)

  To achieve this we print an ephemeral "in progress" line first, let any
  nested `tqdm` progress bars run (they already use ``leave=False`` so they
  clean up after themselves), then overwrite the in-progress line with the
  completion line using ANSI cursor controls.

- Sub-details that used to print on dedicated `·` lines are collapsed into
  the completion summary (or dropped entirely). ``step.detail(...)`` is kept
  as a logger-only call so callers don't have to change.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


# ANSI colour codes (only emitted when stderr is a TTY).
class _Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


# ANSI cursor controls used to overwrite the in-progress line after tqdm bars
# have torn themselves down.
_CURSOR_UP_ONE = "\x1b[1A"
_CLEAR_LINE = "\x1b[2K"
_CARRIAGE_RETURN = "\r"


def _supports_ansi() -> bool:
    """True when stderr is a TTY (we only use ANSI in that case)."""
    return sys.stderr.isatty()


def _colorize(text: str, color: str) -> str:
    """Wrap ``text`` in ANSI codes when the terminal supports them."""
    if _supports_ansi():
        return f"{color}{text}{_Color.RESET}"
    return text


class ProgressReporter:
    """Phase-by-phase progress reporter with a single line per step."""

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
        """Run a step as a context manager; prints exactly one final line.

        Usage::

            with reporter.step("Load CSV") as step:
                df = load_csv(...)
                step.set_summary(f"{len(df)} rows")
        """
        self.current_step += 1
        self._phase_start = time.monotonic()

        # Ephemeral "in progress" line. On TTYs we will overwrite this later.
        in_progress = f"{self._step_prefix()} {description}..."
        print(in_progress, file=sys.stderr, flush=True)

        ctx = _StepContext(self, description=description)
        try:
            yield ctx
        except BaseException:
            elapsed = time.monotonic() - self._phase_start
            final_line = (
                f"{self._step_prefix()} {self._cross_mark()} {description} "
                f"({elapsed:.1f}s) failed"
            )
            self._replace_previous_line(final_line)
            raise
        else:
            elapsed = time.monotonic() - self._phase_start
            # Summaries typically begin with "→" already; avoid double separators.
            summary = f" {ctx.summary}" if ctx.summary else ""
            final_line = (
                f"{self._step_prefix()} {self._check_mark()} {description} "
                f"({elapsed:.1f}s){summary}"
            )
            self._replace_previous_line(final_line)

    def _replace_previous_line(self, final_line: str) -> None:
        """Overwrite the "in progress" line with the completion line.

        On ANSI-capable terminals, move the cursor up one line, clear it,
        then write the new line. Anything tqdm printed in between is already
        gone because tqdm uses ``leave=False``.

        On non-TTY output (pipes, CI logs) the in-progress line stays; we
        just print the completion line on a new row so both are visible in
        the log. This is fine for machine-readable logs where line count
        doesn't matter.
        """
        if _supports_ansi():
            # Walk the cursor back up over the in-progress line, clear it,
            # then print the final version.
            sys.stderr.write(_CURSOR_UP_ONE + _CARRIAGE_RETURN + _CLEAR_LINE)
            print(final_line, file=sys.stderr, flush=True)
        else:
            print(final_line, file=sys.stderr, flush=True)

    def total_elapsed(self) -> float:
        """Seconds elapsed since the ProgressReporter was created."""
        return time.monotonic() - self._overall_start

    def banner(self, title: str) -> None:
        """Print a title banner at the start of a run."""
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
    """Handle exposed inside a ``ProgressReporter.step`` block."""

    def __init__(self, owner: ProgressReporter, description: str) -> None:
        self._owner = owner
        self.description = description
        self.summary: str = ""

    def set_summary(self, text: str) -> None:
        """Text appended to the completion line."""
        self.summary = text

    def detail(self, text: str) -> None:
        """Record a detail line to the logger only.

        The old implementation printed these as indented ``· ...`` lines,
        which doubled the scrollback per step. We now rely on the summary
        (or log inspection) to surface details, keeping the terminal quiet.
        """
        logger.info("  · %s", text)
