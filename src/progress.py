"""CLI progress display utilities.

Design goals:
    - Keep the terminal quiet: exactly one visible line per completed step
      in the final scrollback.
    - Animate an inline spinner while a step is running so the user sees
      activity even when the step produces no other output.
    - Handle ``KeyboardInterrupt`` (Ctrl-C) gracefully with a clear
      "Cancelled" marker instead of a generic failure.

On a TTY, the in-progress line is overwritten with the completion line via
ANSI cursor controls. On non-TTYs (pipes, CI logs), both the in-progress
and completion lines are retained so the timeline remains reconstructible.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


# ANSI colour codes; only used when stderr is a TTY.
class _Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


# ANSI cursor controls for in-place line overwrite.
_CURSOR_UP_ONE = "\x1b[1A"
_CLEAR_LINE = "\x1b[2K"
_CARRIAGE_RETURN = "\r"
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"

# Braille spinner frames — widely supported and visually compact.
_SPINNER_FRAMES: tuple[str, ...] = (
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
)
_SPINNER_INTERVAL_SEC: float = 0.08


def _supports_ansi() -> bool:
    """True when stderr is a TTY (we only emit ANSI in that case)."""
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

    def _cancel_mark(self) -> str:
        return _colorize("⏹", _Color.YELLOW)

    @contextmanager
    def step(self, description: str) -> Iterator["_StepContext"]:
        """Run a step as a context manager.

        Usage::

            with reporter.step("Load CSV") as step:
                df = load_csv(...)
                step.set_summary(f"{len(df)} rows")
        """
        self.current_step += 1
        self._phase_start = time.monotonic()

        # Start the animated spinner on stderr (TTY only).
        spinner = _SpinnerThread(
            description=description,
            step_prefix=self._step_prefix(),
        ) if _supports_ansi() else None
        if spinner is not None:
            spinner.start()
        else:
            # Non-TTY: print a static "in progress" line so piped logs still
            # show the step boundary.
            print(
                f"{self._step_prefix()} {description}...",
                file=sys.stderr,
                flush=True,
            )

        ctx = _StepContext(self, description=description)
        try:
            yield ctx
        except KeyboardInterrupt:
            elapsed = time.monotonic() - self._phase_start
            if spinner is not None:
                spinner.stop()
            final_line = (
                f"{self._step_prefix()} {self._cancel_mark()} {description} "
                f"({elapsed:.1f}s) cancelled by user"
            )
            self._emit_final_line(final_line, had_spinner=spinner is not None)
            raise
        except BaseException:
            elapsed = time.monotonic() - self._phase_start
            if spinner is not None:
                spinner.stop()
            final_line = (
                f"{self._step_prefix()} {self._cross_mark()} {description} "
                f"({elapsed:.1f}s) failed"
            )
            self._emit_final_line(final_line, had_spinner=spinner is not None)
            raise
        else:
            elapsed = time.monotonic() - self._phase_start
            if spinner is not None:
                spinner.stop()
            # Summaries typically begin with "→"; avoid duplicate separators.
            summary = f" {ctx.summary}" if ctx.summary else ""
            final_line = (
                f"{self._step_prefix()} {self._check_mark()} {description} "
                f"({elapsed:.1f}s){summary}"
            )
            self._emit_final_line(final_line, had_spinner=spinner is not None)

    def _emit_final_line(self, final_line: str, *, had_spinner: bool) -> None:
        """Write the completion line, overwriting the spinner line on TTYs.

        - TTY + spinner: the spinner already cleared its line via \\r\\x1b[2K
          before exiting. Write the completion line on the same row.
        - TTY without spinner (shouldn't happen currently): fall through to
          the ANSI overwrite used by the earlier implementation.
        - Non-TTY: print on a new row; both in-progress and completion lines
          survive in the log.
        """
        if _supports_ansi():
            if had_spinner:
                # Spinner already left us at column 0 of an empty line.
                print(final_line, file=sys.stderr, flush=True)
            else:
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

    def cancel_footer(self) -> None:
        """Print a visible footer when the run was cancelled by Ctrl-C."""
        elapsed = self.total_elapsed()
        print(file=sys.stderr)
        print(
            _colorize(
                f"Run cancelled by user (elapsed {elapsed:.1f}s)",
                _Color.BOLD + _Color.YELLOW,
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
        doubling the scrollback per step. We now rely on the summary
        (or log inspection) to surface details.
        """
        logger.info("  · %s", text)


class _SpinnerThread(threading.Thread):
    """Animated spinner that rewrites its stderr line on a timer.

    The spinner runs on a daemon thread so an uncaught exception in the main
    thread (or ``KeyboardInterrupt``) still lets the process exit cleanly.
    """

    def __init__(self, description: str, step_prefix: str) -> None:
        super().__init__(daemon=True, name="progress-spinner")
        self._description = description
        self._step_prefix = step_prefix
        self._stop_event = threading.Event()
        self._frame_index = 0

    def run(self) -> None:
        # Hide the cursor while we draw; tqdm bars from nested code will still
        # flash it back on, which is fine — we hide it again on our cadence.
        sys.stderr.write(_HIDE_CURSOR)
        try:
            # Emit the initial frame so the user sees activity immediately.
            self._draw_current_frame()
            while not self._stop_event.wait(_SPINNER_INTERVAL_SEC):
                self._frame_index = (self._frame_index + 1) % len(_SPINNER_FRAMES)
                self._draw_current_frame()
        finally:
            # Clear the spinner line and restore the cursor before returning.
            sys.stderr.write(_CARRIAGE_RETURN + _CLEAR_LINE + _SHOW_CURSOR)
            sys.stderr.flush()

    def _draw_current_frame(self) -> None:
        """Repaint the spinner line on stderr."""
        frame = _colorize(
            _SPINNER_FRAMES[self._frame_index], _Color.CYAN
        )
        sys.stderr.write(
            _CARRIAGE_RETURN
            + _CLEAR_LINE
            + f"{self._step_prefix} {frame} {self._description}..."
        )
        sys.stderr.flush()

    def stop(self) -> None:
        """Stop the spinner and wait for its last repaint to finish."""
        self._stop_event.set()
        self.join(timeout=1.0)
