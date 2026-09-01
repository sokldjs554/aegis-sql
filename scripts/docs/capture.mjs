/* README 의 콘솔 캡처를 현재 코드로 다시 찍는다.
 *
 * 왜 스크립트로 두는가: 캡처가 코드보다 먼저 낡는다.  콘솔을 고칠 때마다
 * 손으로 다시 찍으면 결국 안 찍게 되고, README 가 없는 기능을 보여주게 된다.
 *
 *   make serve &                       # 다른 창에서 (기본 8000)
 *   node scripts/docs/capture.mjs      # AEGIS_URL 로 주소 바꿀 수 있음
 */
import { chromium } from "playwright";
import { fileURLToPath } from "url";
import path from "path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(HERE, "../../docs/images");
const URL_ = process.env.AEGIS_URL || "http://127.0.0.1:8000/";
const W = 1280;
const H = 1600;                       // 잘라내기 전 여유 — 실제 높이는 내용에 맞춘다

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const errs = [];

const fresh = async (theme = "light") => {
  // 캡처마다 새 페이지를 연다 — 안 그러면 "이번 세션" 이력이 화면마다 다르게 쌓인다.
  const p = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
  p.on("pageerror", (e) => errs.push(`${e.message}`));
  await p.goto(URL_, { waitUntil: "networkidle" });
  await p.evaluate((t) => document.documentElement.setAttribute("data-theme", t), theme);
  await p.waitForTimeout(400);        // 스키마 프리페치(마스킹 배지) + 테마 반영
  return p;
};

const ask = async (p, q) => {
  await p.evaluate((qq) => { document.querySelector("#q").value = qq; }, q);
  await p.click("#run");
  await p.waitForFunction(
    () => document.querySelector("#out")?.children.length > 0 && !document.querySelector("#run").disabled,
    { timeout: 40000 });
  await p.waitForTimeout(450);
};

const tab = async (p, label) => {
  for (const t of await p.$$(".tab")) {
    if ((await t.innerText()).includes(label)) { await t.click(); await p.waitForTimeout(350); return; }
  }
  throw new Error(`탭을 찾지 못했습니다: ${label}`);
};

/* 화면 아래 남는 빈 공간을 잘라낸다. from 을 주면 그 요소 위부터. */
const shot = async (p, name, { from = null } = {}) => {
  const y = from ? Math.max(0, (await (await p.$(from)).boundingBox()).y - 10) : 0;
  const end = await p.evaluate(() => Math.max(
    ...[".drawer", "footer", "#out", "#gout", "body"]
      .map((s) => document.querySelector(s))
      .filter(Boolean)
      .map((e) => e.getBoundingClientRect().bottom)));
  const height = Math.min(H - y, Math.max(200, Math.ceil(end - y) + 14));
  await p.screenshot({ path: path.join(OUT, `${name}.png`), clip: { x: 0, y, width: W, height } });
  console.log(`  ✓ ${name}  ${W}×${height}`);
};

let p;

// 첫 화면
p = await fresh(); await shot(p, "console-empty"); await p.close();

// 질의 결과 · 트레이스
p = await fresh();
await ask(p, "작년 하반기에 체결된 계약 중 월납보험료가 20만원 이상인 건수를 지점별로 알려줘");
await shot(p, "console-query");
await tab(p, "트레이스"); await shot(p, "console-trace");
await p.close();

// 축·눈금이 있는 그래프
p = await fresh();
await ask(p, "채널별 계약 건수를 많은 순으로 보여줘");
{ const t = await p.$$(".viewtog button"); if (!t[1]) throw new Error("그래프 토글 없음"); await t[1].click(); }
await p.waitForTimeout(450);
await shot(p, "console-chart", { from: "#out" });
await p.close();

// 마스킹 배지
p = await fresh();
await ask(p, "고객 테이블 전체를 조회해줘");
{
  const n = await p.$$eval("th .lock", (ns) => ns.length).catch(() => 0);
  if (!n) throw new Error("마스킹 배지가 하나도 없습니다 — /v1/schema 프리페치를 확인하세요");
  console.log(`    마스킹 배지 ${n}개`);
}
await shot(p, "console-masking", { from: "#out" });
await p.close();

// 거버넌스 차단 · 되묻기
p = await fresh(); await ask(p, "고객 이름이랑 주민등록번호 좀 뽑아줘");
await shot(p, "console-governance"); await p.close();
p = await fresh(); await ask(p, "채널별 실적 알려줘");
await shot(p, "console-clarify"); await p.close();

// 행 정책 · 권한 대조
p = await fresh();
{
  const o = await p.$$eval("#ctxBranch option", (ns) => ns.map((n) => n.value));
  if (o.length < 2) throw new Error("지점 컨텍스트 선택지가 없습니다 — /v1/policy 확인");
  await p.selectOption("#ctxBranch", o[1]);
}
await ask(p, "지점별 신계약 건수 상위 5개를 알려줘");
await shot(p, "console-row-policy");
{
  let hit = false;
  for (const btn of await p.$$(".cmp button")) {
    if ((await btn.innerText()).includes("비교")) { await btn.click(); hit = true; break; }
  }
  if (!hit) throw new Error("권한 대조 버튼이 없습니다");
  await p.waitForTimeout(3000);
  await shot(p, "console-compare", { from: ".cmp" });
}
await p.close();

// 스키마 사전 · 런타임·재현
p = await fresh();
await p.click("#schemaBtn"); await p.waitForTimeout(800);
await shot(p, "console-schema");
for (const t of await p.$$(".dtab")) {
  if ((await t.innerText()).includes("런타임")) { await t.click(); break; }
}
await p.waitForTimeout(450);
await shot(p, "console-runtime");
await p.close();

// 거버넌스 샌드박스 — 실제 스키마의 반출금지 컬럼을 쓴다(없는 컬럼은 통과한다)
p = await fresh();
await p.click("#modeGuard"); await p.waitForTimeout(350);
await p.evaluate(() => {
  document.querySelector("#gsql").value =
    "SELECT CUST_NM, RRNO_ENC, TELNO FROM TB_CUST WHERE VIP_GRD_CD = 'VIP'";
});
await p.click("#gcheck");
await p.waitForFunction(() => document.querySelector("#gout")?.children.length > 0, { timeout: 20000 });
await p.waitForTimeout(600);
{
  const txt = await p.$eval("#gout", (n) => n.innerText);
  if (!txt.includes("PII_FORBIDDEN")) throw new Error("샌드박스가 차단하지 않았습니다");
}
await shot(p, "console-sandbox"); await p.close();

// 다크 모드
p = await fresh("dark");
await ask(p, "작년 하반기에 체결된 계약 중 월납보험료가 20만원 이상인 건수를 지점별로 알려줘");
await shot(p, "console-query-dark"); await p.close();

await browser.close();
if (errs.length) { console.error("\n페이지 에러:", errs); process.exit(1); }
console.log("\n캡처 완료 — 페이지 에러 없음");
