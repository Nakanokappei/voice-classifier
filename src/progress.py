"""CLI 実行状況の表示ユーティリティ.

責務:
    - 各パイプラインステップの開始・完了を視覚的に表示
    - 経過時間を自動計測してサマリに含める
    - TTY 環境では色付き、非TTYではプレーンテキストに自動切替
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator


# ANSI カラーコード（サポート環境のみ使用）
class _Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def _supports_color() -> bool:
    """stderr が TTY で、環境が色表示をサポートするか."""
    return sys.stderr.isatty()


def _colorize(text: str, color: str) -> str:
    """色付け（非対応環境ではそのまま）."""
    if _supports_color():
        return f"{color}{text}{_Color.RESET}"
    return text


class ProgressReporter:
    """ステップ進行表示. `[current/total] <description>` 形式で出力."""

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
        """ステップ開始〜終了をコンテキストマネージャで記述.

        使い方::

            with reporter.step("CSVを読み込み中") as step:
                df = load_csv(...)
                step.detail(f"{len(df)}件読込")
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
                f"      {self._cross_mark()} 失敗 ({elapsed:.1f}秒)",
                file=sys.stderr,
                flush=True,
            )
            raise
        else:
            elapsed = time.monotonic() - self._phase_start
            suffix = f" {ctx.summary}" if ctx.summary else ""
            print(
                f"      {self._check_mark()} 完了 ({elapsed:.1f}秒){suffix}",
                file=sys.stderr,
                flush=True,
            )

    def total_elapsed(self) -> float:
        """開始からの経過秒数."""
        return time.monotonic() - self._overall_start

    def banner(self, title: str) -> None:
        """目立つ見出しを表示."""
        bar = "=" * 60
        print(file=sys.stderr)
        print(_colorize(bar, _Color.DIM), file=sys.stderr)
        print(_colorize(f" {title}", _Color.BOLD), file=sys.stderr)
        print(_colorize(bar, _Color.DIM), file=sys.stderr, flush=True)

    def footer(self, message: str) -> None:
        """完了時のフッタ."""
        elapsed = self.total_elapsed()
        print(file=sys.stderr)
        print(
            _colorize(
                f"全工程完了: {message} (総経過 {elapsed:.1f}秒)",
                _Color.BOLD + _Color.GREEN,
            ),
            file=sys.stderr,
            flush=True,
        )


class _StepContext:
    """ステップ内でサブ情報を記録するハンドル."""

    def __init__(self, owner: ProgressReporter) -> None:
        self._owner = owner
        self.summary: str = ""

    def set_summary(self, text: str) -> None:
        """ステップ完了時に `完了` 行の後ろに付くサマリ文."""
        self.summary = text

    def detail(self, text: str) -> None:
        """実行中の補足情報をインデント付きで表示."""
        print(f"      {_colorize('·', _Color.DIM)} {text}", file=sys.stderr, flush=True)
