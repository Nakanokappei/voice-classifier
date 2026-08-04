"""Collect a support report describing why the pipeline will not run.

This module is deliberately different from the rest of ``src``: it imports
nothing outside the standard library. The most common thing to go wrong on a
customer machine is that the dependencies are missing or were installed into
the wrong interpreter, and a diagnostic that needs pandas to start cannot
report on that.

    python src\\diagnose.py

It writes 診断結果_YYYYMMDD_HHMMSS.txt next to the project folder and prints
the path. The customer sends that one file back.

The report never contains the API key (only its length), and never contains
any cell of the input CSV (only column names and a row count), because it
travels by email.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import socket
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent

# The offline bundle ships cp314 wheels, so anything else fails at install time.
EXPECTED_PYTHON = (3, 14)
DEFAULT_API_VERSION = "2024-10-21"
HTTP_TIMEOUT = 20

# Verdicts. Kept short so the report stays scannable in a fixed-width mail body.
OK, WARN, NG, INFO = "OK  ", "注意", "NG  ", "    "

# Modules whose importability maps onto a user-visible capability.
KEY_MODULES = [
    "numpy", "pandas", "sklearn", "openai", "dotenv", "tqdm", "markdown",
    "hdbscan", "hnswlib", "igraph", "leidenalg",
]

# Executed inside the *target* interpreter so the report describes the
# environment pipeline.py actually runs in, not the one launching this script.
PROBE = r"""
import importlib, json, sys
try:
    import importlib.metadata as md
except ImportError:
    md = None
names, mods = json.loads(sys.argv[1]), json.loads(sys.argv[2])
out = {
    "version": list(sys.version_info[:3]),
    "version_string": sys.version.split()[0],
    "executable": sys.executable,
    "bits": 64 if sys.maxsize > 2 ** 32 else 32,
    "versions": {},
    "imports": {},
}
for n in names:
    try:
        out["versions"][n] = md.version(n) if md else None
    except Exception:
        out["versions"][n] = None
for m in mods:
    try:
        importlib.import_module(m)
        out["imports"][m] = None
    except Exception as exc:
        out["imports"][m] = f"{type(exc).__name__}: {exc}"
# The pipeline's HTTP client trusts this bundle rather than the OS store, so
# the diagnosis needs it to reproduce what the pipeline will actually see.
try:
    import certifi
    out["certifi"] = certifi.where()
except Exception:
    out["certifi"] = None
print(json.dumps(out))
"""


class Report:
    """Accumulates report lines and remembers verdicts for the summary."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.verdicts: dict[str, str] = {}

    def section(self, title: str) -> None:
        self.lines += ["", "=" * 72, title, "=" * 72]

    def check(self, key: str, status: str, label: str, detail: str = "") -> None:
        """Record one verdict. `key` is what the summary logic looks up."""
        self.verdicts[key] = status
        self.lines.append(f"[{status}] {label}" + (f"  {detail}" if detail else ""))

    def mark(self, key: str, status: str) -> None:
        """Record a verdict for the summary without printing a line for it."""
        self.verdicts[key] = status

    def note(self, text: str = "") -> None:
        self.lines.append(text)

    def failed(self, *keys: str) -> bool:
        return any(self.verdicts.get(k) == NG for k in keys)


def probe(interpreter: Path | str, packages: list[str]) -> dict | None:
    """Run the probe script under `interpreter`, returning its findings.

    Returns None when the interpreter cannot be started at all, which is itself
    the answer to "why does nothing work".
    """
    try:
        result = subprocess.run(
            [str(interpreter), "-c", PROBE, json.dumps(packages), json.dumps(KEY_MODULES)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def locked_packages() -> list[str]:
    """Package names the bundle pins, falling back to requirements.txt."""
    lock = REPO / "tools" / "offline-lock.txt"
    if lock.exists():
        return [line.split("==")[0] for line in lock.read_text(encoding="utf-8").split()]

    names = []
    requirements = REPO / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                names.append(_split_requirement(line))
    return names


def _split_requirement(line: str) -> str:
    for separator in (">=", "==", "<=", "~=", ">", "<"):
        if separator in line:
            return line.split(separator)[0].strip()
    return line


def read_env_file(path: Path) -> dict[str, str]:
    """Parse .env without python-dotenv, which may not be installed yet."""
    values = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def azure_request(url: str, api_key: str, payload: dict,
                  ca_bundle: str | None = None) -> tuple[int | None, str]:
    """POST to Azure OpenAI and return (status, explanation).

    The status code is the diagnosis: the body is irrelevant and is not read
    into the report. A 400 still proves the deployment resolved, so it counts
    as reachable rather than as a configuration error.

    `ca_bundle` selects which certificate authorities to trust. That matters
    because this script and the pipeline do not share a TLS trust store: urllib
    uses the Windows certificate store, while the openai SDK goes through httpx
    and certifi's bundle. A network that re-signs TLS traffic is therefore
    trusted by one and rejected by the other.
    """
    context = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else None
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT, context=context) as response:
            return response.status, "正常に応答しました"
    except urllib.error.HTTPError as exc:
        return exc.code, {
            400: "デプロイには到達しています（リクエスト内容の差異。設定としては正常）",
            401: "APIキーが違います（手順4をやり直してください）",
            403: "アクセスが拒否されました（キーの権限、またはネットワーク制限）",
            404: "デプロイ名が見つかりません（.env のデプロイ名を確認してください）",
            429: "レート制限中です（設定は正しい。しばらく待って再実行）",
        }.get(exc.code, f"HTTP {exc.code} が返りました")
    except urllib.error.URLError as exc:
        # A certificate failure is not a blocked port; it usually means the
        # network is inspecting TLS with its own certificate authority.
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            return None, ("TLS証明書の検証に失敗しました"
                          "（社内プロキシによるSSL検査の可能性が高いです）")
        return None, f"接続できません（{exc.reason}）"
    except (TimeoutError, socket.timeout):
        return None, "タイムアウトしました（社内ネットワークで遮断されている可能性）"
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never crash
        return None, f"{type(exc).__name__}: {exc}"


def check_environment(report: Report) -> None:
    """Operating system, interpreter, console encoding."""
    report.section("1. 実行環境")
    report.note(f"OS            : {platform.platform()}")
    report.note(f"マシン        : {platform.machine()}")
    report.note(f"実行した場所  : {Path.cwd()}")
    report.note(f"プログラム位置: {REPO}")
    report.note(f"PYTHONUTF8    : {os.environ.get('PYTHONUTF8', '(未設定)')}")
    report.note(f"標準出力の文字コード: {sys.stdout.encoding}")
    try:
        usage = shutil.disk_usage(REPO)
        report.note(f"ディスク空き  : {usage.free / 1024 ** 3:.1f} GB")
    except OSError:
        report.note("ディスク空き  : 取得できませんでした")


def check_interpreters(report: Report, packages: list[str]) -> dict | None:
    """Compare the venv interpreter against the required version.

    The setup guide runs the pipeline through .venv\\Scripts\\python.exe, so the
    venv is the environment that matters even when this script was started by
    a different Python.
    """
    report.section("2. Python と仮想環境（.venv）")

    venv_python = REPO / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():                     # POSIX layout, for dev machines
        venv_python = REPO / ".venv" / "bin" / "python"

    report.note(f"この診断を動かしたPython: {sys.version.split()[0]}  ({sys.executable})")

    if not venv_python.exists():
        report.check(
            "venv", NG, "仮想環境 .venv がありません",
            "手順3（部品の組み込み）がまだ済んでいないか、失敗しています",
        )
        found = probe(sys.executable, packages)
    else:
        report.check("venv", OK, "仮想環境 .venv があります", str(venv_python))
        found = probe(venv_python, packages)
        if found is None:
            report.check("venv_runs", NG, ".venv のPythonを起動できません",
                         "仮想環境が壊れています。.venv フォルダを削除して手順3をやり直してください")
            return None
        report.check("venv_runs", OK, ".venv のPythonは起動します")

    if found is None:
        report.check("python_version", NG, "Pythonの情報を取得できませんでした")
        return None

    version = tuple(found["version"][:2])
    report.note(f"実際に使われるPython    : {found['version_string']}  ({found['bits']}bit)")
    if version == EXPECTED_PYTHON:
        report.check("python_version", OK, f"Pythonのバージョンは {found['version_string']} です")
    else:
        report.check(
            "python_version", NG,
            f"Pythonのバージョンが違います（{found['version_string']}）",
            f"同梱の部品は Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]} 専用です",
        )
    if found["bits"] != 64:
        report.check("python_bits", NG, "32bit版のPythonです", "64bit版を入れ直してください")
    return found


def check_packages(report: Report, found: dict | None, packages: list[str]) -> None:
    """List what is missing, and translate that into lost capability."""
    report.section("3. 必要な部品（パッケージ）")
    if found is None:
        report.check("packages", NG, "確認できませんでした（Pythonが起動しないため）")
        return

    missing = [name for name in packages if not found["versions"].get(name)]
    if missing:
        report.check("packages", NG, f"{len(missing)}個の部品が入っていません",
                     "手順3をやり直してください")
        report.note("  入っていないもの: " + ", ".join(missing))
    else:
        report.check("packages", OK, f"必要な部品 {len(packages)}個はすべて入っています")

    report.note("")
    report.note("主要な部品の読み込み確認:")
    for module, error in found["imports"].items():
        report.note(f"  {'OK  ' if error is None else 'NG  '} {module}"
                    + (f"  {error}" if error else ""))

    # These two mirror the optional-dependency handling in src/tuner.py.
    report.note("")
    if found["imports"].get("hdbscan") is None:
        report.check("hdbscan", OK, "HDBSCAN による分類が使えます")
    else:
        report.check("hdbscan", WARN, "HDBSCAN が使えません", "分類の選択肢が減ります")

    leiden = [m for m in ("hnswlib", "igraph", "leidenalg") if found["imports"].get(m) is not None]
    if leiden:
        report.check("leiden", WARN, "Leiden による分類が使えません",
                     f"読み込めない部品: {', '.join(leiden)}")
    else:
        report.check("leiden", OK, "Leiden による分類が使えます")


def check_settings(report: Report, env: dict[str, str]) -> None:
    """Validate .env without ever printing the key itself."""
    report.section("4. Azureの設定（.env）")
    env_path = REPO / ".env"
    if not env_path.exists():
        report.check("env", NG, ".env ファイルがありません", "手順4を実施してください")
        return

    report.check("env", OK, ".env ファイルはあります", str(env_path))

    key = env.get("AZURE_OPENAI_API_KEY", "")
    if not key:
        report.check("api_key", NG, "APIキーが空です")
    elif "★" in key or key.startswith("your-"):
        report.check("api_key", NG, "APIキーが書き換えられていません",
                     "★の部分が残ったままです。手順4をやり直してください")
    elif key.startswith("http"):
        report.check("api_key", NG, "APIキーの欄にエンドポイントが入っています",
                     "キーとエンドポイントが逆になっています")
    else:
        report.check("api_key", OK, f"APIキーは設定されています（{len(key)}文字）")

    endpoint = env.get("AZURE_OPENAI_ENDPOINT", "")
    if not endpoint or "★" in endpoint:
        report.check("endpoint", NG, "エンドポイントが設定されていません")
    elif not endpoint.startswith("https://"):
        report.check("endpoint", NG, "エンドポイントの形式が違います", endpoint)
    else:
        report.check("endpoint", OK, "エンドポイント", endpoint)

    for label, name in [
        ("埋め込み", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        ("ラベル付け", "AZURE_OPENAI_NAMER_DEPLOYMENT"),
        ("解説コメント", "AZURE_OPENAI_ADVISOR_DEPLOYMENT"),
    ]:
        value = env.get(name, "")
        if value:
            report.check(f"deploy_{name}", OK, f"デプロイ名（{label}）", value)
        else:
            report.check(f"deploy_{name}", NG, f"デプロイ名（{label}）が空です")


def check_connectivity(report: Report, env: dict[str, str],
                       ca_bundle: str | None = None) -> None:
    """Resolve, connect, and exercise each deployment.

    Each call sends a couple of tokens of dummy text and reads only the status
    code, so no customer data leaves the machine during diagnosis.
    """
    report.section("5. Azureへの通信")
    endpoint = env.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = env.get("AZURE_OPENAI_API_KEY", "")
    if not endpoint.startswith("https://") or not api_key or "★" in api_key:
        report.check("azure", NG, "設定が未完成のため通信を確認できませんでした",
                     "先に手順4をやり直してください")
        return

    host = urlparse(endpoint).hostname or ""
    try:
        address = socket.gethostbyname(host)
        report.check("dns", OK, f"名前解決 {host}", address)
    except OSError as exc:
        report.check("dns", NG, f"名前解決できません {host}", str(exc))
        report.note("  → エンドポイントの綴り、または社内ネットワークのDNS制限を確認してください")
        return

    try:
        with socket.create_connection((host, 443), timeout=HTTP_TIMEOUT):
            report.check("tcp", OK, "443番ポートに接続できます")
    except OSError as exc:
        report.check("tcp", NG, "443番ポートに接続できません", str(exc))
        report.note("  → ファイアウォール／プロキシで遮断されている可能性があります")
        return

    version = env.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)

    # Trust whatever the pipeline trusts, so a pass here means the pipeline can
    # connect and a failure here is a failure it will hit too.
    if ca_bundle:
        report.note(f"  （本体プログラムと同じ証明書で確認します: {ca_bundle}）")

    embedding = env.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    if embedding:
        status, message = azure_request(
            f"{endpoint}/openai/deployments/{embedding}/embeddings?api-version={version}",
            api_key, {"input": "test"}, ca_bundle=ca_bundle,
        )
        report.check("azure_embedding", OK if status == 200 else NG,
                     f"埋め込みデプロイ「{embedding}」", message)
        # Distinguish a TLS problem from a wrong key or deployment name, so the
        # summary does not send the customer to redo step 4 for nothing.
        if status is None and "TLS証明書" in message:
            report.mark("tls", NG)

    for label, name in [("ラベル付け", "AZURE_OPENAI_NAMER_DEPLOYMENT"),
                        ("解説コメント", "AZURE_OPENAI_ADVISOR_DEPLOYMENT")]:
        deployment = env.get(name, "")
        if not deployment:
            continue
        status, message = azure_request(
            f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}",
            api_key, {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
            ca_bundle=ca_bundle,
        )
        # 400 means the deployment resolved and only the request shape differed,
        # which is a pass for configuration purposes.
        good = status in (200, 400, 429)
        report.check(f"azure_{name}", OK if good else NG,
                     f"{label}デプロイ「{deployment}」", message)


def check_input_data(report: Report) -> None:
    """Report CSV structure only -- never the contents."""
    report.section("6. 入力データ")
    report.note("※ 個人情報保護のため、CSVの中身（セルの値）は一切出力していません。")
    report.note("")

    input_dir = REPO / "data" / "input"
    if not input_dir.exists():
        report.check("input", NG, "data\\input フォルダがありません", "手順5を実施してください")
        return

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        report.check("input", NG, "data\\input にCSVがありません", "手順5を実施してください")
        return

    report.check("input", OK, f"CSVが {len(csv_files)}個あります")
    for path in csv_files:
        report.note(f"  {path.name}  ({path.stat().st_size:,} バイト)")
        header, encoding, rows = read_csv_shape(path)
        if header is None:
            report.note("    読み取れませんでした（文字コード不明、または破損）")
            continue
        report.note(f"    文字コード: {encoding} / データ行数: {rows}")
        report.note(f"    列名: {', '.join(header)}")
        if "問い合わせ内容" not in header:
            report.check("text_col", WARN, "「問い合わせ内容」列がありません",
                         "実行コマンドの --text-col を上の列名のどれかに変えてください")


def read_csv_shape(path: Path) -> tuple[list[str] | None, str, int]:
    """Return (column names, encoding, row count), trying Japanese encodings."""
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            with path.open(encoding=encoding, newline="") as handle:
                header = next(csv.reader(handle), None)
                if header is None:
                    return None, encoding, 0
                rows = sum(1 for _ in handle)
            return header, encoding, rows
        except (UnicodeDecodeError, csv.Error):
            continue
        except OSError:
            break
    return None, "不明", 0


def check_last_run(report: Report) -> None:
    """Surface only error lines from the most recent run log."""
    report.section("7. 直近の実行結果")
    output_dir = REPO / "data" / "output"
    if not output_dir.exists():
        report.check("last_run", INFO, "まだ一度も実行されていません")
        return

    runs = sorted((d for d in output_dir.iterdir() if d.is_dir()),
                  key=lambda d: d.stat().st_mtime)
    if not runs:
        report.check("last_run", INFO, "まだ一度も実行されていません")
        return

    latest = runs[-1]
    stamp = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    report.check("last_run", INFO, f"最後の実行: {latest.name}", stamp)
    report.note("  生成されたファイル: " + ", ".join(sorted(p.name for p in latest.iterdir())))

    log = latest / "run.log"
    if not log.exists():
        return

    # Only error-ish lines: the full log can quote customer text.
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    errors = [l for l in lines
              if any(marker in l for marker in ("ERROR", "CRITICAL", "Traceback", "Error:"))]
    if errors:
        report.check("last_run_errors", NG, f"ログにエラーが {len(errors)}件あります")
        report.note("")
        for line in errors[-15:]:
            report.note(f"  {line[:300]}")
    else:
        report.check("last_run_errors", OK, "ログにエラーはありません")


def summarise(report: Report) -> None:
    """Turn the verdicts into a ranked list of likely causes.

    Ordered by dependency: a wrong Python explains missing packages, which
    explains everything downstream, so only the first applicable cause is worth
    acting on first.
    """
    report.section("8. 考えられる原因（上から順に確認してください）")

    causes: list[str] = []
    if report.failed("python_version"):
        causes.append("Pythonのバージョンが違います。手順1のリンクから Python 3.14 を入れ直し、"
                      "その後 .venv フォルダを削除して手順3からやり直してください。")
    if report.failed("python_bits"):
        causes.append("32bit版のPythonが入っています。64bit版を入れ直してください。")
    if report.failed("venv", "venv_runs"):
        causes.append("仮想環境（.venv）がない、または壊れています。手順3をやり直してください。")
    if report.failed("packages"):
        causes.append("必要な部品が入っていません。手順3をやり直してください。")
    if report.failed("env", "api_key", "endpoint"):
        causes.append("Azureの接続情報（.env）が正しくありません。手順4をやり直してください。")
    if report.failed("dns", "tcp"):
        causes.append("Azureに通信できません。社内ネットワークの制限が原因の可能性が高いです。"
                      "情報システム部門に、上記エンドポイントへのHTTPS通信の許可を確認してください。")
    if report.failed("tls"):
        causes.append("TLS証明書の検証に失敗しています。社内プロキシがSSL通信を検査している"
                      "環境で起きます。設定の誤りではないため手順4をやり直しても直りません。"
                      "情報システム部門に、このPCでのプロキシ証明書の扱いを確認してください。")
    elif report.failed("azure_embedding", "azure_AZURE_OPENAI_NAMER_DEPLOYMENT",
                       "azure_AZURE_OPENAI_ADVISOR_DEPLOYMENT"):
        causes.append("Azureのデプロイ名またはAPIキーが違います。上の「5. Azureへの通信」の"
                      "メッセージに従って手順4をやり直してください。")
    if report.failed("input"):
        causes.append("分析するCSVが置かれていません。手順5を実施してください。")

    if causes:
        for index, cause in enumerate(causes, 1):
            report.note(f"{index}. {cause}")
    elif report.failed("last_run_errors"):
        report.note("環境の設定に問題は見つかりませんでした。")
        report.note("実行時にエラーが出ています。「7. 直近の実行結果」のエラー行を確認してください。")
    else:
        report.note("環境に問題は見つかりませんでした。")
        report.note("それでも動かない場合は、実行時に画面に出た赤い文字を")
        report.note("このレポートと一緒に共有してください。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="動作しないときの環境診断レポートを作ります",
    )
    parser.add_argument("--no-azure", action="store_true",
                        help="Azureへの通信確認を行わない")
    parser.add_argument("--output", type=Path, default=None,
                        help="レポートの出力先ファイル")
    args = parser.parse_args()

    # The console may be cp932; a diagnostic that dies printing Japanese is
    # worse than useless.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    report = Report()
    now = datetime.now()
    report.note("voice-classifier 診断レポート")
    report.note(f"作成日時: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    report.note("このファイルをそのまま担当者に送ってください。")
    report.note("APIキーとCSVの中身は含まれていません。")

    packages = locked_packages()
    check_environment(report)
    found = check_interpreters(report, packages)
    check_packages(report, found, packages)

    env = read_env_file(REPO / ".env")
    check_settings(report, env)
    if args.no_azure:
        report.section("5. Azureへの通信")
        report.check("azure", INFO, "--no-azure が指定されたため確認していません")
    else:
        check_connectivity(report, env, (found or {}).get("certifi"))

    check_input_data(report)
    check_last_run(report)
    summarise(report)

    text = "\n".join(report.lines) + "\n"
    destination = args.output or REPO / f"診断結果_{now.strftime('%Y%m%d_%H%M%S')}.txt"
    # utf-8-sig so Windows メモ帳 and Excel both read the Japanese correctly.
    destination.write_text(text, encoding="utf-8-sig")

    print(text)
    print("=" * 72)
    print("レポートを次の場所に保存しました。このファイルを担当者に送ってください：")
    print(f"  {destination}")
    print("=" * 72)


if __name__ == "__main__":
    main()
