/* 웹 콘솔 히어로 GIF 의 프레임을 찍는다.  make serve 를 먼저 띄우세요.
 *
 *   node scripts/docs/demo_console.mjs
 *   scripts/docs/gif.sh /tmp/aegis-frames/console docs/images/console-demo.gif 820
 */
import { chromium } from "playwright";
import { writeFileSync, mkdirSync, rmSync } from "fs";
import path from "path";

const FR = process.env.FRAME_DIR || "/tmp/aegis-frames/console";
const URL_ = process.env.AEGIS_URL || "http://127.0.0.1:8000/";
rmSync(FR, { recursive: true, force: true });
mkdirSync(FR, { recursive: true });

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const p = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
const errs = [];
p.on("pageerror", (e) => errs.push(e.message));
await p.goto(URL_, { waitUntil: "networkidle" });
await p.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));
await p.waitForTimeout(700);

const list = [];
let n = 0;
const frame = async (dur) => {
  const f = `c${String(++n).padStart(4, "0")}.png`;
  await p.screenshot({ path: path.join(FR, f) });
  list.push({ f, dur });
};

const act = async (q, hold) => {
  const chars = [...q];
  await p.evaluate(() => { document.querySelector("#q").value = ""; });
  for (let i = 0; i < chars.length; i += 2) {
    await p.evaluate((s) => { document.querySelector("#q").value = s; }, chars.slice(0, i + 2).join(""));
    await frame(0.07);
  }
  await frame(0.40);
  await p.click("#run");
  for (let i = 0; i < 4; i++) { await p.waitForTimeout(110); await frame(0.13); }  // 스테퍼가 지나가는 모습
  await p.waitForFunction(
    () => document.querySelector("#out")?.children.length > 0 && !document.querySelector("#run").disabled,
    { timeout: 30000 });
  await p.waitForTimeout(260);
  await frame(hold);
};

await frame(0.9);                                                     // 첫 화면
await act("작년 하반기 계약 중 월납보험료 20만원 이상 건수를 지점별로", 2.4);  // ① 실행
await act("고객 이름이랑 주민등록번호 좀 뽑아줘", 2.5);                  // ② 차단
await act("채널별 실적 알려줘", 2.8);                                    // ③ 되묻기

writeFileSync(path.join(FR, "list.txt"),
  list.map((x) => `file '${x.f}'\nduration ${x.dur}`).join("\n") +
  `\nfile '${list[list.length - 1].f}'\n`);
console.log(`  프레임 ${n}개 · ${list.reduce((s, x) => s + x.dur, 0).toFixed(1)}초 → ${FR}`);
await browser.close();
if (errs.length) { console.error("페이지 에러:", errs); process.exit(1); }
