#!/usr/bin/env python3
"""`aegis ask` 3막을 실제로 돌려 ANSI 출력을 HTML 조각으로 바꾼다.

터미널 GIF 를 손으로 그리지 않기 위한 스크립트다.  실제로 명령을 실행해 나온
바이트를 그대로 칠하므로, 메시지 문구가 바뀌면 GIF 도 따라 바뀐다.

출력: scripts/docs/acts.json  (demo_terminal.mjs 가 읽는다)
"""
from __future__ import annotations

import html
import json
import os
import pathlib
import re
import subprocess
import sys
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

QUESTIONS = [
    "작년 하반기에 체결된 계약 중 월납보험료가 20만원 이상인 건수를 지점별로 알려줘",
    "고객 이름이랑 주민등록번호 좀 뽑아줘",
    "설계사 실적 좀 보여줘",
]

# COLUMNS 를 고정해야 rich 가 접는 위치가 매번 같다 — GIF 가 재현 가능해진다.
ENV = dict(
    os.environ,
    AEGIS_DEMO_PUBLIC="1",
    PYTHONHASHSEED="0",
    FORCE_COLOR="1",
    TERM="xterm-256color",
    COLUMNS="96",
)

SGR = re.compile(r"\x1b\[([0-9;]*)m")
FG = {
    30: "#3b4048", 31: "#e05561", 32: "#8cc265", 33: "#d18f52", 34: "#4aa5f0",
    35: "#c162de", 36: "#42b3c2", 37: "#c7ccd4",
    90: "#6b7280", 91: "#ff616e", 92: "#a5e075", 93: "#f0a45d", 94: "#61afef",
    95: "#de73ff", 96: "#4cd1e0", 97: "#eef1f6",
}


def esc_cells(chunk: str) -> str:
    """전각 글자 묶음만 <span class="w"> 로 감싼다.

    rich 는 East Asian Width W/F 를 2칸으로 세어 상자 여백을 맞추는데, 브라우저
    monospace 는 한글을 그보다 좁게 그린다.  묶음마다 자간을 벌려 칸 수를 맞춘다.
    """
    out: list[str] = []
    run: list[str] = []
    is_wide = False

    def flush() -> None:
        if not run:
            return
        text = html.escape("".join(run))
        out.append(f'<span class="w">{text}</span>' if is_wide else text)
        run.clear()

    for ch in chunk:
        wide = unicodedata.east_asian_width(ch) in ("W", "F")
        if wide != is_wide:
            flush()
            is_wide = wide
        run.append(ch)
    flush()
    return "".join(out)


def ansi_to_html(text: str) -> str:
    out: list[str] = []
    fg: str | None = None
    bold = dim = span_open = False

    def close() -> None:
        nonlocal span_open
        if span_open:
            out.append("</span>")
            span_open = False

    def open_() -> None:
        nonlocal span_open
        style = []
        if fg:
            style.append(f"color:{fg}")
        if bold:
            style.append("font-weight:700")
        if dim:
            style.append("opacity:.62")
        if style:
            out.append(f'<span style="{";".join(style)}">')
            span_open = True

    pos = 0
    for m in SGR.finditer(text):
        if chunk := text[pos:m.start()]:
            out.append(esc_cells(chunk))
        pos = m.end()
        close()
        for code in (m.group(1) or "0").split(";"):
            c = int(code or 0)
            if c == 0:
                fg, bold, dim = None, False, False
            elif c == 1:
                bold = True
            elif c == 2:
                dim = True
            elif c == 22:
                bold = dim = False
            elif c == 39:
                fg = None
            elif c in FG:
                fg = FG[c]
        open_()
    if tail := text[pos:]:
        out.append(esc_cells(tail))
    close()
    return "".join(out)


def main() -> int:
    aegis = REPO / ".venv/bin/aegis"
    if not aegis.exists():
        print(f"aegis 실행 파일이 없습니다: {aegis}\n  make setup 을 먼저 도세요.", file=sys.stderr)
        return 1

    acts = []
    for q in QUESTIONS:
        # 한 스트림으로 받는다 — 따로 받으면 가드 경고 줄이 패널 뒤로 밀려
        # 실제 실행 순서와 달라진다.
        r = subprocess.run([str(aegis), "ask", q], cwd=REPO, env=ENV, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        body = r.stdout.rstrip("\n")
        acts.append({"cmd": f'aegis ask "{q}"', "out": ansi_to_html(body)})
        print(f"  ✓ {q[:24]}…  ({len(body.splitlines())}줄, exit={r.returncode})")

    (HERE / "acts.json").write_text(json.dumps(acts, ensure_ascii=False), encoding="utf-8")
    print(f"  → {HERE / 'acts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
