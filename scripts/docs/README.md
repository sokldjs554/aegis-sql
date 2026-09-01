# 문서 캡처 도구

README 의 화면 캡처와 데모 GIF 를 **지금 코드에서 다시 만드는** 스크립트입니다.
손으로 찍지 않는 이유는 하나입니다 — 캡처는 코드보다 먼저 낡습니다. 콘솔을
고칠 때마다 손으로 다시 찍으면 결국 안 찍게 되고, README 가 없는 기능을
보여주게 됩니다. 실제로 이 저장소에서 한 번 그런 일이 있었습니다.

산출물은 `docs/images/` 에만 커밋합니다. 중간 파일(`acts.json`, 프레임 PNG)은
버전 관리하지 않습니다.

## 준비

본체와 별개인 개발 전용 의존성입니다. 저장소를 쓰는 데는 필요 없습니다.

```bash
npm install playwright && npx playwright install chromium
```

`ffmpeg` 은 PATH 에 있으면 쓰고, 없으면 `imageio-ffmpeg` 가 들고 있는 바이너리를
찾습니다. 둘 다 없으면 `FFMPEG=/경로/ffmpeg` 로 지정하세요.

## 쓰는 법

```bash
make serve &                                   # 다른 창에서 (기본 8000)

# ① 화면 캡처 12장 → docs/images/console-*.png
node scripts/docs/capture.mjs

# ② 웹 콘솔 히어로 GIF
node scripts/docs/demo_console.mjs
scripts/docs/gif.sh /tmp/aegis-frames/console docs/images/console-demo.gif 820

# ③ 터미널 GIF (aegis ask 3막)
python3 scripts/docs/build_terminal.py         # 실제로 명령을 돌려 ANSI 를 받는다
node    scripts/docs/demo_terminal.mjs
scripts/docs/gif.sh /tmp/aegis-frames/terminal docs/images/terminal-demo.gif
```

환경변수: `AEGIS_URL`(기본 `http://127.0.0.1:8000/`), `CHROMIUM_PATH`,
`FRAME_DIR`, `FFMPEG`.

## 설계 메모

- **캡처마다 새 페이지를 연다.** 한 페이지에서 이어 찍으면 "이번 세션" 이력이
  화면마다 다르게 쌓여 캡처들의 레이아웃이 어긋납니다.
- **`capture.mjs` 는 검증도 한다.** 마스킹 배지가 0개거나, 권한 대조 버튼이
  없거나, 샌드박스가 `PII_FORBIDDEN` 을 내지 않으면 실패합니다 — 기능이 조용히
  깨진 채로 캡처만 갱신되는 것을 막습니다. 페이지 콘솔 에러가 있어도 실패합니다.
- **GIF 는 영상이 아니라 프레임 목록으로 굽는다.** 정지 구간을 "프레임 하나 +
  긴 지연"으로 넣을 수 있어 파일이 작고(2.1MB → 0.3MB), 손실 압축 잡음이 없어
  글자가 더 또렷합니다.
- **터미널 GIF 는 실제 출력을 그대로 칠한다.** `aegis ask` 를 정말로 실행해 나온
  ANSI 를 HTML 로 옮기므로, 메시지 문구가 바뀌면 GIF 도 따라 바뀝니다.
  `rich` 가 2칸으로 세는 전각 글자는 브라우저에서 더 좁게 그려져 상자 테두리가
  어긋나는데, `demo_terminal.mjs` 가 그 폭을 직접 재서 자간으로 보정합니다.
