#!/usr/bin/env python3
"""Deterministically build the AEGIS-SQL demo database.

The generator is seeded, so every developer, CI run and evaluation report sees
byte-identical data — which is what makes the benchmark's gold SQL results
reproducible.  Distributions are deliberately non-uniform (regional skew,
seasonal claim peaks, channel-correlated delinquency, grade-correlated premium)
so that aggregate questions have interesting, non-degenerate answers.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260824
TODAY = date(2026, 8, 24)
MARKETING_CONSENT_RATE = 0.40

# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

CODES: dict[str, list[tuple[str, str, str]]] = {
    "GNDR": [("M", "남성", "주민등록 성별 남"), ("F", "여성", "주민등록 성별 여")],
    "CTRT_STAT": [
        ("01", "정상", "보험료 납입 중이거나 납입완료된 유효계약"),
        ("02", "실효", "보험료 미납으로 효력이 상실된 계약"),
        ("03", "해지", "계약자 요청 또는 회사 사유로 해지된 계약"),
        ("04", "만기", "보험기간이 종료된 계약"),
        ("05", "부활", "실효 후 부활 처리된 계약"),
    ],
    "PAY_CYCL": [
        ("01", "월납", "매월 납입"), ("02", "3개월납", "분기 납입"),
        ("03", "6개월납", "반기 납입"), ("04", "연납", "연 1회 납입"),
        ("05", "일시납", "계약 시 전액 납입"),
    ],
    "CHNL": [
        ("10", "대면", "설계사 대면 모집"), ("20", "텔레마케팅", "TM 아웃바운드 모집"),
        ("30", "온라인", "다이렉트/CM 채널"), ("40", "방카슈랑스", "은행 창구 모집"),
        ("50", "GA", "독립법인대리점 모집"),
    ],
    "PROD_TYP": [
        ("L", "종신보험", "사망보장 중심 장기보험"), ("H", "건강보험", "질병 입원/수술 보장"),
        ("C", "암보험", "암 진단/치료 보장"), ("A", "상해보험", "재해/상해 보장"),
        ("P", "연금보험", "노후 연금 수령"), ("S", "저축보험", "저축성 상품"),
        ("D", "운전자보험", "자동차 사고 형사/행정 비용 보장"),
    ],
    "CLM_TYP": [
        ("01", "입원", "입원일당/입원의료비"), ("02", "통원", "통원의료비"),
        ("03", "수술", "수술비 담보"), ("04", "진단", "진단금 담보"),
        ("05", "사망", "사망보험금"), ("06", "장해", "후유장해보험금"),
        ("07", "실손", "실손의료비"),
    ],
    "CLM_STAT": [
        ("10", "접수", "청구서류 접수 완료"), ("20", "심사중", "지급심사 진행 중"),
        ("30", "지급완료", "보험금 전액 지급"), ("40", "부지급", "면책 등 사유로 미지급"),
        ("50", "일부지급", "청구액 일부만 지급"),
    ],
    "PAY_MTHD": [
        ("01", "자동이체", "은행 계좌 자동이체"), ("02", "신용카드", "카드 정기결제"),
        ("03", "지로", "지로 납부"), ("04", "가상계좌", "가상계좌 입금"),
    ],
    "VIP_GRD": [
        ("V1", "플래티넘", "연납환산보험료 최상위 고객"), ("V2", "골드", "우수 고객"),
        ("V3", "실버", "일반 우량 고객"), ("V4", "일반", "기본 등급"),
    ],
    "UW_RSLT": [
        ("A", "표준체승낙", "가입조건 변경 없이 승낙"), ("B", "할증승낙", "보험료 할증 조건부 승낙"),
        ("C", "부담보승낙", "특정 부위/질병 부담보 조건부 승낙"),
        ("D", "거절", "인수 거절"), ("E", "보류", "추가 서류 요청으로 보류"),
    ],
    "TCKT_CTGY": [
        ("01", "보험금", "보험금 지급 관련 문의/민원"), ("02", "계약변경", "수익자/주소 등 변경"),
        ("03", "납입", "보험료 납입 관련"), ("04", "해지", "해지/환급금 문의"),
        ("05", "상품문의", "상품 내용 문의"), ("06", "불완전판매", "설명의무 위반 민원"),
    ],
    "PROC_STAT": [
        ("01", "접수", "티켓 접수"), ("02", "처리중", "담당자 배정 후 처리 중"),
        ("03", "완료", "처리 완료"), ("04", "반려", "요건 미비로 반려"),
    ],
    "RGN": [
        ("11", "서울", "서울특별시"), ("26", "부산", "부산광역시"), ("27", "대구", "대구광역시"),
        ("28", "인천", "인천광역시"), ("29", "광주", "광주광역시"), ("30", "대전", "대전광역시"),
        ("41", "경기", "경기도"), ("43", "충북", "충청북도"), ("46", "전남", "전라남도"),
        ("47", "경북", "경상북도"), ("48", "경남", "경상남도"),
    ],
    "AGNT_GRD": [
        ("S", "수석", "수석 FP"), ("A", "선임", "선임 FP"),
        ("B", "주임", "주임 FP"), ("C", "사원", "신입 FP"),
    ],
}

SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "홍"]
GIVEN_M = ["민준", "서준", "도윤", "예준", "시우", "하준", "지호", "준우", "준서", "건우", "현우", "우진", "선우", "연우", "정우"]
GIVEN_F = ["서연", "서윤", "지우", "서현", "하윤", "민서", "지유", "윤서", "채원", "지민", "수아", "다은", "은서", "예은", "소율"]

REGION_CITY = {
    "11": ("서울특별시", ["강남구", "서초구", "송파구", "마포구", "노원구", "성북구", "은평구"]),
    "26": ("부산광역시", ["해운대구", "부산진구", "동래구", "남구", "사하구"]),
    "27": ("대구광역시", ["수성구", "달서구", "북구", "중구"]),
    "28": ("인천광역시", ["연수구", "남동구", "부평구", "서구"]),
    "29": ("광주광역시", ["서구", "북구", "광산구"]),
    "30": ("대전광역시", ["유성구", "서구", "중구"]),
    "41": ("경기도", ["성남시 분당구", "수원시 영통구", "고양시 일산동구", "용인시 기흥구", "안양시 동안구", "화성시"]),
    "43": ("충청북도", ["청주시 흥덕구", "충주시"]),
    "46": ("전라남도", ["여수시", "순천시", "목포시"]),
    "47": ("경상북도", ["포항시 남구", "구미시", "경주시"]),
    "48": ("경상남도", ["창원시 성산구", "김해시", "진주시"]),
}
REGION_WEIGHT = {"11": 22, "41": 26, "26": 7, "27": 5, "28": 6, "29": 3, "30": 3, "43": 3, "46": 4, "47": 5, "48": 6}

BRANCH_NAMES = [
    "강남", "서초", "송파", "여의도", "종로", "마포", "분당", "판교", "수원", "일산", "용인", "부천",
    "해운대", "서면", "대구수성", "인천송도", "광주상무", "대전둔산", "청주", "창원", "전주", "포항", "울산", "제주",
]
PROD_PREFIX = ["한아름", "행복드림", "미래설계", "든든플러스", "The좋은", "다이렉트", "실속형", "프리미엄", "New", "스마트"]
PROD_SUFFIX = {
    "L": ["종신보험", "변액종신보험", "정기보험"],
    "H": ["건강보험", "종합건강보험", "간편심사건강보험"],
    "C": ["암보험", "3대질병보험", "유방암케어보험"],
    "A": ["상해보험", "레저상해보험", "일상생활배상보험"],
    "P": ["연금보험", "변액연금보험", "즉시연금보험"],
    "S": ["저축보험", "적립보험", "교육자금보험"],
    "D": ["운전자보험", "안심운전자보험"],
}
HOSPITALS = [
    "서울대학교병원", "세브란스병원", "서울아산병원", "삼성서울병원", "서울성모병원", "고려대안암병원",
    "분당서울대병원", "아주대병원", "부산대학교병원", "경북대학교병원", "충남대학교병원", "전남대학교병원",
    "한마음정형외과", "굿모닝내과의원", "새봄이비인후과", "연세필외과", "미래재활의학과",
]
DIAG = ["C50", "C16", "C34", "C18", "I21", "I63", "J18", "K35", "K80", "M51", "M17", "S72", "S52", "E11", "N20", "H25", "A09", "F32"]
CVRG_CATALOG = [
    ("CV01", "일반상해사망", 100_000_000), ("CV02", "일반상해후유장해", 100_000_000),
    ("CV03", "질병사망", 50_000_000), ("CV04", "암진단비", 30_000_000),
    ("CV05", "뇌혈관질환진단비", 20_000_000), ("CV06", "허혈성심장질환진단비", 20_000_000),
    ("CV07", "질병입원일당", 100_000), ("CV08", "상해입원일당", 100_000),
    ("CV09", "수술비", 5_000_000), ("CV10", "실손의료비(급여)", 50_000_000),
    ("CV11", "실손의료비(비급여)", 30_000_000), ("CV12", "골절진단비", 500_000),
    ("CV13", "치아보철치료비", 3_000_000), ("CV14", "간병인지원일당", 150_000),
    ("CV15", "운전자벌금", 30_000_000),
]
TICKET_TEXT = {
    "01": ["보험금 지급이 지연되는 이유를 알고 싶습니다.", "청구한 실손 보험금이 일부만 지급되었습니다.", "서류를 다시 제출해야 하나요?"],
    "02": ["주소를 변경하고 싶습니다.", "수익자를 배우자로 변경 요청드립니다.", "자동이체 계좌를 바꾸고 싶어요."],
    "03": ["이번 달 보험료가 두 번 출금되었습니다.", "납입 유예 신청이 가능한가요?", "카드 결제일을 변경하고 싶습니다."],
    "04": ["해지환급금이 얼마인지 알려주세요.", "계약을 해지하려면 어떻게 해야 하나요?", "해지 철회가 가능한지 문의드립니다."],
    "05": ["갱신 시 보험료가 얼마나 오르나요?", "이 상품의 면책기간이 궁금합니다.", "특약을 추가할 수 있나요?"],
    "06": ["설계사가 설명하지 않은 조건이 있습니다.", "가입 당시 안내와 실제 보장이 다릅니다.", "자필서명을 하지 않았습니다."],
}


def ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def weighted_choice(rng: random.Random, mapping: dict[str, float]) -> str:
    keys = list(mapping)
    return rng.choices(keys, weights=[mapping[k] for k in keys], k=1)[0]


def rand_date(rng: random.Random, start: date, end: date) -> date:
    span = (end - start).days
    return start + timedelta(days=rng.randint(0, max(0, span)))


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def build(conn: sqlite3.Connection, scale: float = 1.0) -> dict[str, int]:
    rng = random.Random(SEED)
    counts: dict[str, int] = {}
    cur = conn.cursor()

    # -- 공통코드 -------------------------------------------------------- #
    rows = []
    for grp, entries in CODES.items():
        for i, (cd, nm, desc) in enumerate(entries):
            rows.append((grp, cd, nm, desc, i + 1, "Y"))
    cur.executemany("INSERT INTO TB_COMM_CD VALUES (?,?,?,?,?,?)", rows)
    counts["TB_COMM_CD"] = len(rows)

    # -- 지점 ------------------------------------------------------------ #
    branches = []
    for i, nm in enumerate(BRANCH_NAMES):
        rgn = weighted_choice(rng, REGION_WEIGHT)
        open_d = rand_date(rng, date(2005, 1, 1), date(2022, 12, 31))
        cls_d = ymd(rand_date(rng, date(2024, 1, 1), TODAY)) if rng.random() < 0.08 else None
        branches.append((f"BR{i + 1:03d}", f"{nm}지점", rgn, ymd(open_d), cls_d, rng.randint(3, 22)))
    cur.executemany("INSERT INTO TB_BRCH VALUES (?,?,?,?,?,?)", branches)
    counts["TB_BRCH"] = len(branches)
    branch_ids = [b[0] for b in branches]

    # -- 설계사 ---------------------------------------------------------- #
    n_agents = int(320 * scale)
    agents = []
    for i in range(n_agents):
        gender = rng.choice("MF")
        nm = rng.choice(SURNAMES) + rng.choice(GIVEN_M if gender == "M" else GIVEN_F)
        hire = rand_date(rng, date(2012, 1, 1), date(2026, 3, 1))
        rsgn = ymd(rand_date(rng, hire + timedelta(days=200), TODAY)) if rng.random() < 0.22 else None
        grd = rng.choices(["S", "A", "B", "C"], weights=[8, 22, 40, 30])[0]
        agents.append((
            f"AG{i + 1:05d}", nm, rng.choice(branch_ids), ymd(hire), rsgn, grd,
            f"LIC-{rng.randint(100000, 999999)}", f"010-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}",
        ))
    cur.executemany("INSERT INTO TB_AGNT VALUES (?,?,?,?,?,?,?,?)", agents)
    counts["TB_AGNT"] = len(agents)
    agent_ids = [a[0] for a in agents]

    # -- 고객 ------------------------------------------------------------ #
    n_cust = int(5200 * scale)
    customers = []
    for i in range(n_cust):
        gender = rng.choices("MF", weights=[49, 51])[0]
        nm = rng.choice(SURNAMES) + rng.choice(GIVEN_M if gender == "M" else GIVEN_F)
        brdt = rand_date(rng, date(1945, 1, 1), date(2008, 12, 31))
        rgn = weighted_choice(rng, REGION_WEIGHT)
        city, gus = REGION_CITY[rgn]
        addr = f"{city} {rng.choice(gus)} {rng.choice(['테헤란로','중앙로','한밭대로','번영로','새터로','문화로'])} {rng.randint(1, 400)}"
        join = rand_date(rng, date(2010, 1, 1), TODAY)
        grade = rng.choices(["V1", "V2", "V3", "V4"], weights=[3, 9, 23, 65])[0]
        rrn = f"{brdt.strftime('%y%m%d')}-{'1' if gender == 'M' else '2'}******"
        customers.append((
            f"CU{i + 1:07d}", nm, f"ENC:{abs(hash((rrn, i))) % (10**16):016d}", ymd(brdt), gender,
            f"{rng.randint(1,63):02d}{rng.randint(100,999):03d}", addr,
            f"010-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}",
            f"user{i + 1}@{rng.choice(['naver.com','gmail.com','daum.net','kakao.com'])}",
            ymd(join), grade, "Y" if rng.random() < MARKETING_CONSENT_RATE else "N", rgn,
        ))
    cur.executemany("INSERT INTO TB_CUST VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", customers)
    counts["TB_CUST"] = len(customers)

    # -- 상품 ------------------------------------------------------------ #
    products = []
    pi = 0
    for typ, suffixes in PROD_SUFFIX.items():
        for suf in suffixes:
            for pre in rng.sample(PROD_PREFIX, 2):
                pi += 1
                start = rand_date(rng, date(2014, 1, 1), date(2025, 6, 30))
                end = ymd(rand_date(rng, start + timedelta(days=400), TODAY)) if rng.random() < 0.3 else None
                base = {"L": 90_000, "H": 55_000, "C": 42_000, "A": 22_000, "P": 180_000, "S": 250_000, "D": 15_000}[typ]
                products.append((
                    f"PD{pi:04d}", f"{pre}{suf}", typ, ymd(start), end,
                    int(base * rng.uniform(0.7, 1.6) // 1000 * 1000),
                    rng.choice([0, 15, 20, 30]), rng.choice([65, 70, 80, 100]),
                    "Y" if typ in {"H", "C", "A", "D"} and rng.random() < 0.6 else "N",
                ))
    cur.executemany("INSERT INTO TB_PROD VALUES (?,?,?,?,?,?,?,?,?)", products)
    counts["TB_PROD"] = len(products)

    # -- 계약 ------------------------------------------------------------ #
    n_ctrt = int(13000 * scale)
    contracts, coverages, payments, uws = [], [], [], []
    cvrg_seq = 0
    pay_seq = 0
    grade_prm = {"V1": 3.2, "V2": 2.0, "V3": 1.35, "V4": 1.0}
    for i in range(n_ctrt):
        cust = customers[rng.randrange(len(customers))]
        prod = products[rng.randrange(len(products))]
        chnl = rng.choices(["10", "20", "30", "40", "50"], weights=[46, 9, 17, 11, 17])[0]
        agnt = None if chnl == "30" else rng.choice(agent_ids)
        ctrt_dt = rand_date(rng, date(2016, 1, 1), TODAY - timedelta(days=15))
        pay_term = rng.choice([5, 10, 15, 20, 20, 30])
        expr = date(min(ctrt_dt.year + rng.choice([10, 20, 20, 30, 40]), 2099), ctrt_dt.month, min(ctrt_dt.day, 28))
        age_days = (TODAY - ctrt_dt).days
        # status distribution depends on tenure and channel (TM/GA lapse more)
        lapse_bias = {"10": 1.0, "20": 1.8, "30": 1.25, "40": 0.85, "50": 1.6}[chnl]
        r = rng.random() * (1.0 / lapse_bias) + (age_days / 4200.0) * 0.10
        if expr <= TODAY:
            stat = "04"
        elif r < 0.66:
            stat = "01"
        elif r < 0.78:
            stat = "02"
        elif r < 0.90:
            stat = "03"
        else:
            stat = "05"
        term_dt = ymd(rand_date(rng, ctrt_dt + timedelta(days=90), TODAY)) if stat == "03" else None
        mon_prm = int(prod[5] * grade_prm[cust[10]] * rng.uniform(0.6, 2.4) // 100 * 100)
        insd = int(mon_prm * rng.uniform(180, 900) // 10000 * 10000)
        ctrt_no = f"CT{ctrt_dt.year}{i + 1:07d}"
        contracts.append((
            ctrt_no, cust[0], prod[0], agnt, ymd(ctrt_dt), ymd(expr), stat, mon_prm, insd,
            rng.choices(["01", "02", "03", "04", "05"], weights=[72, 6, 5, 12, 5])[0],
            pay_term, chnl, term_dt, 1 if stat == "05" else 0,
        ))

        # 담보 2~5개
        for cvrg in rng.sample(CVRG_CATALOG, rng.randint(2, 5)):
            cvrg_seq += 1
            amt = int(cvrg[2] * rng.uniform(0.2, 1.2) // 10000 * 10000)
            coverages.append((
                f"CV{cvrg_seq:08d}", ctrt_no, cvrg[0], cvrg[1], amt,
                max(500, int(amt * rng.uniform(0.0004, 0.0016) // 100 * 100)),
                ymd(ctrt_dt), ymd(expr),
            ))

        # 수납 이력 (최대 24회 최근분)
        n_pay = min(24, max(1, age_days // 30))
        for k in range(n_pay):
            pay_seq += 1
            due = ctrt_dt + timedelta(days=30 * (k + 1))
            if due > TODAY:
                break
            dlq = rng.random() < (0.05 * lapse_bias)
            dlq_days = rng.randint(1, 75) if dlq else 0
            payments.append((
                f"PY{pay_seq:09d}", ctrt_no, ymd(due + timedelta(days=dlq_days)), ymd(due), mon_prm,
                rng.choices(["01", "02", "03", "04"], weights=[64, 24, 5, 7])[0],
                "Y" if dlq else "N", dlq_days,
            ))

        # 인수심사
        risk = int(max(0, min(100, rng.gauss(38, 18))))
        uw_rslt = "A" if risk < 45 else "B" if risk < 62 else "C" if risk < 78 else "D" if risk < 90 else "E"
        uws.append((
            f"UW{i + 1:07d}", ctrt_no, ymd(ctrt_dt - timedelta(days=rng.randint(1, 12))), uw_rslt, risk,
            f"EMP{rng.randint(1000, 1999)}",
            {"A": "표준체 승낙", "B": "위험도 반영 할증 적용", "C": "특정부위 부담보 조건", "D": "고지사항 부적격", "E": "추가 검진 서류 필요"}[uw_rslt],
        ))

    cur.executemany("INSERT INTO TB_CTRT VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", contracts)
    cur.executemany("INSERT INTO TB_CVRG VALUES (?,?,?,?,?,?,?,?)", coverages)
    cur.executemany("INSERT INTO TB_PAY VALUES (?,?,?,?,?,?,?,?)", payments)
    cur.executemany("INSERT INTO TB_UW VALUES (?,?,?,?,?,?,?)", uws)
    counts.update({"TB_CTRT": len(contracts), "TB_CVRG": len(coverages), "TB_PAY": len(payments), "TB_UW": len(uws)})

    # -- 보험금청구 ------------------------------------------------------ #
    claims = []
    active = [c for c in contracts if c[6] in {"01", "04", "05"}]
    n_clm = int(9500 * scale)
    for i in range(n_clm):
        ctrt = active[rng.randrange(len(active))]
        ctrt_dt = date(int(ctrt[4][:4]), int(ctrt[4][4:6]), int(ctrt[4][6:]))
        acdn = rand_date(rng, max(ctrt_dt + timedelta(days=90), date(2019, 1, 1)), TODAY - timedelta(days=5))
        # seasonal peak: winter respiratory / summer accidents
        if acdn.month in (12, 1, 2) and rng.random() < 0.35:
            typ = rng.choice(["01", "02", "07"])
        else:
            typ = rng.choices(["01", "02", "03", "04", "05", "06", "07"], weights=[18, 25, 14, 12, 2, 4, 25])[0]
        clm_dt = acdn + timedelta(days=rng.randint(1, 60))
        if clm_dt > TODAY:
            clm_dt = TODAY
        base_amt = {"01": 900_000, "02": 180_000, "03": 3_200_000, "04": 12_000_000,
                    "05": 60_000_000, "06": 18_000_000, "07": 420_000}[typ]
        clm_amt = int(base_amt * rng.uniform(0.35, 2.6) // 1000 * 1000)
        fraud = int(max(0, min(100, rng.gauss(22, 17) + (25 if rng.random() < 0.05 else 0))))
        age_d = (TODAY - clm_dt).days
        if age_d < 7:
            stat = rng.choices(["10", "20"], weights=[65, 35])[0]
        elif age_d < 21:
            stat = rng.choices(["20", "30", "50", "40"], weights=[35, 45, 12, 8])[0]
        else:
            stat = rng.choices(["30", "50", "40"], weights=[74, 16, 10])[0]
        if stat == "30":
            pay_amt, dedt = clm_amt, ymd(min(TODAY, clm_dt + timedelta(days=rng.randint(3, 25))))
        elif stat == "50":
            pay_amt, dedt = int(clm_amt * rng.uniform(0.25, 0.85) // 1000 * 1000), ymd(min(TODAY, clm_dt + timedelta(days=rng.randint(5, 35))))
        else:
            pay_amt, dedt = 0, None
        claims.append((
            f"CL{clm_dt.year}{i + 1:07d}", ctrt[0], typ, ymd(acdn), ymd(clm_dt), dedt,
            clm_amt, pay_amt, stat, rng.choice(HOSPITALS) if typ != "05" else None,
            rng.choice(DIAG), fraud,
        ))
    cur.executemany("INSERT INTO TB_CLM VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", claims)
    counts["TB_CLM"] = len(claims)

    # -- 상담/민원 ------------------------------------------------------- #
    tickets = []
    n_tckt = int(4300 * scale)
    for i in range(n_tckt):
        ctrt = contracts[rng.randrange(len(contracts))]
        cust_id = ctrt[1]
        rcpt = rand_date(rng, date(2023, 1, 1), TODAY)
        ctgy = rng.choices(["01", "02", "03", "04", "05", "06"], weights=[31, 16, 18, 14, 15, 6])[0]
        age_d = (TODAY - rcpt).days
        stat = rng.choices(["01", "02", "03", "04"], weights=[40, 35, 20, 5])[0] if age_d < 5 else \
            rng.choices(["02", "03", "04"], weights=[10, 82, 8])[0]
        cmpl = ymd(min(TODAY, rcpt + timedelta(days=rng.randint(0, 14)))) if stat in {"03", "04"} else None
        satis = rng.choices([1, 2, 3, 4, 5], weights=[6, 9, 20, 34, 31])[0] if stat == "03" else None
        tickets.append((
            f"TK{i + 1:07d}", cust_id, ctrt[0] if rng.random() < 0.82 else None, ymd(rcpt), cmpl,
            rng.choices(["10", "20", "30", "40", "50"], weights=[18, 34, 33, 8, 7])[0],
            ctgy, stat, rng.choice(TICKET_TEXT[ctgy]), satis,
        ))
    cur.executemany("INSERT INTO TB_CS_TCKT VALUES (?,?,?,?,?,?,?,?,?,?)", tickets)
    counts["TB_CS_TCKT"] = len(tickets)

    conn.commit()
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the AEGIS-SQL demo SQLite database")
    ap.add_argument("--out", default=str(ROOT / "data" / "demo" / "aegis_demo.sqlite"))
    ap.add_argument("--scale", type=float, default=1.0, help="row-count multiplier (CI uses 0.25)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        if not args.force:
            print(f"[skip] {out} already exists (use --force to rebuild)")
            return 0
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(out)
    conn.executescript((ROOT / "data" / "demo" / "schema.sql").read_text(encoding="utf-8"))
    counts = build(conn, scale=args.scale)
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    total = sum(counts.values())
    width = max(len(k) for k in counts)
    print(f"[ok] built {out} ({out.stat().st_size / 1_048_576:.1f} MB)")
    for k in sorted(counts):
        print(f"     {k:<{width}}  {counts[k]:>8,}")
    print(f"     {'TOTAL':<{width}}  {total:>8,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
