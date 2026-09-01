#!/usr/bin/env bash
# 프레임 목록(list.txt)을 GIF 로 굽는다.
#
# 영상 대신 프레임 목록을 쓰는 이유: 정지 구간을 "프레임 하나 + 긴 지연"으로
# 넣을 수 있어 파일이 훨씬 작고, 손실 압축 잡음이 없어 글자가 또렷하다.
# (이 저장소에서 2.1MB → 0.3MB, 화질은 오히려 개선됐다.)
#
#   scripts/docs/gif.sh <프레임디렉터리> <출력.gif> [가로폭]
set -euo pipefail
DIR=${1:?프레임 디렉터리}; OUT=${2:?출력 gif}; WIDTH=${3:-0}
FF=${FFMPEG:-$(command -v ffmpeg || python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')}

if [ "$WIDTH" -gt 0 ]; then
  SCALE="scale=${WIDTH}:-1:flags=lanczos"
  GEN="${SCALE},palettegen=max_colors=64:stats_mode=diff"
  USE="[0:v]${SCALE}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle"
else
  GEN="palettegen=max_colors=64:stats_mode=diff"
  USE="[0:v][1:v]paletteuse=dither=none:diff_mode=rectangle"
fi

"$FF" -v error -y -f concat -safe 0 -i "$DIR/list.txt" -vf "$GEN" "$DIR/pal.png"
"$FF" -v error -y -f concat -safe 0 -i "$DIR/list.txt" -i "$DIR/pal.png" \
      -lavfi "$USE" -loop 0 "$OUT"
echo "  ✓ $OUT  ($(du -h "$OUT" | cut -f1))"
