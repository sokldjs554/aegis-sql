/* 터미널 데모 GIF 의 프레임을 찍는다.  build_terminal.py 를 먼저 도세요.
 *
 *   python3 scripts/docs/build_terminal.py
 *   node    scripts/docs/demo_terminal.mjs
 *   scripts/docs/gif.sh /tmp/aegis-frames/terminal docs/images/terminal-demo.gif
 */
import { chromium } from "playwright";
import { readFileSync, writeFileSync, mkdirSync, rmSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FR = process.env.FRAME_DIR || "/tmp/aegis-frames/terminal";
rmSync(FR, { recursive: true, force: true });
mkdirSync(FR, { recursive: true });

const acts = JSON.parse(readFileSync(path.join(HERE, "acts.json"), "utf8"));
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const p = await browser.newPage({ viewport: { width: 900, height: 580 } });
await p.goto("file://" + path.join(HERE, "terminal.html"));

/* 이 브라우저·폰트에서 한글이 ASCII 몇 칸으로 그려지는지 재서, rich 가 세는
   2칸이 되도록 자간을 정한다.  폰트가 바뀌어도 상자 테두리가 어긋나지 않는다. */
const gap = await p.evaluate(() => {
  const pre = document.querySelector("pre");
  const probe = document.createElement("span");
  probe.style.whiteSpace = "pre";
  pre.appendChild(probe);
  const w = (s) => { probe.textContent = s; return probe.getBoundingClientRect().width / s.length; };
  const em = parseFloat(getComputedStyle(pre).fontSize);
  const g = (2 * w("x".repeat(40)) - w("가".repeat(40))) / em;
  probe.remove();
  document.documentElement.style.setProperty("--wide-gap", `${g.toFixed(5)}em`);
  return g;
});
console.log(`  전각 자간 보정 ${gap.toFixed(5)}em`);
await p.waitForTimeout(300);

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const list = [];
let n = 0;
let buf = "";

const frame = async (dur) => {
  const f = `f${String(++n).padStart(4, "0")}.png`;
  await p.screenshot({ path: path.join(FR, f) });
  list.push({ f, dur });
};
const draw = async (dur) => { await p.evaluate((x) => window.render(x), buf); await frame(dur); };

buf = '<span class="p">$</span> ';
await draw(0.7);

for (const a of acts) {
  const chars = [...a.cmd];
  for (let i = 0; i < chars.length; i += 2) {   // 2글자씩 = 타이핑 느낌
    buf += esc(chars.slice(i, i + 2).join(""));
    await draw(0.07);
  }
  await frame(0.45);                             // Enter 직전 멈칫
  buf += "\n";              await draw(0.30);    // 실행 중
  buf += a.out + "\n";      await draw(2.10);    // 결과 — 읽을 시간
  buf += '\n<span class="p">$</span> '; await draw(0.35);
}
await frame(1.60);

// concat 디먹서는 마지막 파일을 한 번 더 적어 줘야 그 지연이 반영된다.
writeFileSync(path.join(FR, "list.txt"),
  list.map((x) => `file '${x.f}'\nduration ${x.dur}`).join("\n") +
  `\nfile '${list[list.length - 1].f}'\n`);
console.log(`  프레임 ${n}개 · ${list.reduce((s, x) => s + x.dur, 0).toFixed(1)}초 → ${FR}`);
await browser.close();
