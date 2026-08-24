-- =====================================================================
-- AEGIS-SQL demo schema — "한화라이프 스타일" 국내 보험사 레거시 코어 DB (축약판)
-- ---------------------------------------------------------------------
-- 이 스키마는 의도적으로 국내 금융/보험 IT의 실제 관행을 재현한다:
--   1) 물리 테이블/컬럼명은 영문 대문자 약어    (TB_CTRT.CTRT_STAT_CD)
--   2) 의미는 한글 주석(데이터 사전)에만 존재    (계약상태코드)
--   3) 날짜는 DATE 타입이 아니라 CHAR(8) 'YYYYMMDD' 문자열
--   4) 금액은 원(KRW) 단위 정수
--   5) 코드값은 전부 공통코드 테이블(TB_COMM_CD)과 조인해야 사람이 읽을 수 있음
--   6) 개인식별정보(주민등록번호/연락처/주소)가 운영 테이블에 그대로 존재
-- 이 6가지가 바로 "LLM에 스키마만 던지면 되는" 순진한 Text-to-SQL이
-- 국내 금융권에서 무너지는 지점이고, AEGIS-SQL이 해결하려는 문제다.
-- 컬럼 뒤 '--' 주석은 introspector가 데이터 사전으로 파싱한다.
-- =====================================================================

-- 공통코드
CREATE TABLE TB_COMM_CD (
    CD_GRP      TEXT    NOT NULL,               -- 코드그룹
    CD          TEXT    NOT NULL,               -- 코드값
    CD_NM       TEXT    NOT NULL,               -- 코드명
    CD_DESC     TEXT,                           -- 코드설명
    SORT_ORD    INTEGER NOT NULL DEFAULT 0,     -- 정렬순서
    USE_YN      TEXT    NOT NULL DEFAULT 'Y',   -- 사용여부
    PRIMARY KEY (CD_GRP, CD)
);

-- 지점
CREATE TABLE TB_BRCH (
    BRCH_CD     TEXT    NOT NULL PRIMARY KEY,   -- 지점코드
    BRCH_NM     TEXT    NOT NULL,               -- 지점명
    RGN_CD      TEXT    NOT NULL,               -- 지역코드
    OPEN_DT     TEXT    NOT NULL,               -- 지점개설일자
    CLS_DT      TEXT,                           -- 지점폐쇄일자
    MNG_EMP_CNT INTEGER NOT NULL DEFAULT 0      -- 관리직원수
);

-- 설계사
CREATE TABLE TB_AGNT (
    AGNT_ID     TEXT    NOT NULL PRIMARY KEY,   -- 설계사ID
    AGNT_NM     TEXT    NOT NULL,               -- 설계사명
    BRCH_CD     TEXT    NOT NULL,               -- 소속지점코드
    HIRE_DT     TEXT    NOT NULL,               -- 위촉일자
    RSGN_DT     TEXT,                           -- 해촉일자
    GRD_CD      TEXT    NOT NULL,               -- 설계사등급코드
    LIC_NO      TEXT,                           -- 자격증번호
    TELNO       TEXT,                           -- 설계사연락처
    FOREIGN KEY (BRCH_CD) REFERENCES TB_BRCH(BRCH_CD)
);

-- 고객
CREATE TABLE TB_CUST (
    CUST_ID     TEXT    NOT NULL PRIMARY KEY,   -- 고객ID
    CUST_NM     TEXT    NOT NULL,               -- 고객명
    RRNO_ENC    TEXT,                           -- 주민등록번호암호화
    BRDT        TEXT    NOT NULL,               -- 생년월일
    GNDR_CD     TEXT    NOT NULL,               -- 성별코드
    ZIP_CD      TEXT,                           -- 우편번호
    ADDR        TEXT,                           -- 주소
    TELNO       TEXT,                           -- 휴대전화번호
    EMAIL       TEXT,                           -- 이메일주소
    JOIN_DT     TEXT    NOT NULL,               -- 최초거래일자
    VIP_GRD_CD  TEXT    NOT NULL DEFAULT 'V4',  -- 고객등급코드
    MKT_AGR_YN  TEXT    NOT NULL DEFAULT 'N',   -- 마케팅수신동의여부
    RGN_CD      TEXT                            -- 거주지역코드
);

-- 상품
CREATE TABLE TB_PROD (
    PROD_CD     TEXT    NOT NULL PRIMARY KEY,   -- 상품코드
    PROD_NM     TEXT    NOT NULL,               -- 상품명
    PROD_TYP_CD TEXT    NOT NULL,               -- 상품유형코드
    SALE_STRT_DT TEXT   NOT NULL,               -- 판매개시일자
    SALE_END_DT TEXT,                           -- 판매종료일자
    BASE_PRM    INTEGER NOT NULL,               -- 기준보험료
    MIN_AGE     INTEGER NOT NULL DEFAULT 0,     -- 최소가입연령
    MAX_AGE     INTEGER NOT NULL DEFAULT 100,   -- 최대가입연령
    RNW_YN      TEXT    NOT NULL DEFAULT 'N'    -- 갱신형여부
);

-- 계약
CREATE TABLE TB_CTRT (
    CTRT_NO     TEXT    NOT NULL PRIMARY KEY,   -- 계약번호
    CUST_ID     TEXT    NOT NULL,               -- 계약자고객ID
    PROD_CD     TEXT    NOT NULL,               -- 상품코드
    AGNT_ID     TEXT,                           -- 모집설계사ID
    CTRT_DT     TEXT    NOT NULL,               -- 계약체결일자
    EXPR_DT     TEXT,                           -- 만기일자
    CTRT_STAT_CD TEXT   NOT NULL,               -- 계약상태코드
    MON_PRM     INTEGER NOT NULL,               -- 월납보험료
    TOT_INSD_AMT INTEGER NOT NULL,              -- 총가입금액
    PAY_CYCL_CD TEXT    NOT NULL,               -- 납입주기코드
    PAY_TERM_YR INTEGER NOT NULL,               -- 납입기간년수
    CHNL_CD     TEXT    NOT NULL,               -- 모집채널코드
    TERM_DT     TEXT,                           -- 해지일자
    RVIV_CNT    INTEGER NOT NULL DEFAULT 0,     -- 부활횟수
    FOREIGN KEY (CUST_ID) REFERENCES TB_CUST(CUST_ID),
    FOREIGN KEY (PROD_CD) REFERENCES TB_PROD(PROD_CD),
    FOREIGN KEY (AGNT_ID) REFERENCES TB_AGNT(AGNT_ID)
);

-- 담보
CREATE TABLE TB_CVRG (
    CVRG_ID     TEXT    NOT NULL PRIMARY KEY,   -- 담보ID
    CTRT_NO     TEXT    NOT NULL,               -- 계약번호
    CVRG_CD     TEXT    NOT NULL,               -- 담보코드
    CVRG_NM     TEXT    NOT NULL,               -- 담보명
    INSD_AMT    INTEGER NOT NULL,               -- 담보가입금액
    CVRG_PRM    INTEGER NOT NULL,               -- 담보보험료
    STRT_DT     TEXT    NOT NULL,               -- 담보개시일자
    END_DT      TEXT,                           -- 담보종료일자
    FOREIGN KEY (CTRT_NO) REFERENCES TB_CTRT(CTRT_NO)
);

-- 수납
CREATE TABLE TB_PAY (
    PAY_ID      TEXT    NOT NULL PRIMARY KEY,   -- 수납ID
    CTRT_NO     TEXT    NOT NULL,               -- 계약번호
    PAY_DT      TEXT    NOT NULL,               -- 수납일자
    DUE_DT      TEXT    NOT NULL,               -- 납입기일
    PAY_AMT     INTEGER NOT NULL,               -- 수납금액
    PAY_MTHD_CD TEXT    NOT NULL,               -- 납입방법코드
    DLQ_YN      TEXT    NOT NULL DEFAULT 'N',   -- 연체여부
    DLQ_DAYS    INTEGER NOT NULL DEFAULT 0,     -- 연체일수
    FOREIGN KEY (CTRT_NO) REFERENCES TB_CTRT(CTRT_NO)
);

-- 보험금청구
CREATE TABLE TB_CLM (
    CLM_NO      TEXT    NOT NULL PRIMARY KEY,   -- 청구번호
    CTRT_NO     TEXT    NOT NULL,               -- 계약번호
    CLM_TYP_CD  TEXT    NOT NULL,               -- 청구유형코드
    ACDN_DT     TEXT    NOT NULL,               -- 사고일자
    CLM_DT      TEXT    NOT NULL,               -- 청구접수일자
    DEDT_DT     TEXT,                           -- 보험금지급일자
    CLM_AMT     INTEGER NOT NULL,               -- 청구금액
    PAY_AMT     INTEGER NOT NULL DEFAULT 0,     -- 지급금액
    CLM_STAT_CD TEXT    NOT NULL,               -- 청구상태코드
    HOSP_NM     TEXT,                           -- 병원명
    DIAG_CD     TEXT,                           -- 진단코드
    FRAUD_SCR   INTEGER NOT NULL DEFAULT 0,     -- 이상징후점수
    FOREIGN KEY (CTRT_NO) REFERENCES TB_CTRT(CTRT_NO)
);

-- 인수심사
CREATE TABLE TB_UW (
    UW_ID       TEXT    NOT NULL PRIMARY KEY,   -- 인수심사ID
    CTRT_NO     TEXT    NOT NULL,               -- 계약번호
    UW_DT       TEXT    NOT NULL,               -- 심사일자
    UW_RSLT_CD  TEXT    NOT NULL,               -- 인수심사결과코드
    RISK_SCR    INTEGER NOT NULL,               -- 위험점수
    UW_EMP_ID   TEXT,                           -- 심사자사번
    RMRK        TEXT,                           -- 심사의견
    FOREIGN KEY (CTRT_NO) REFERENCES TB_CTRT(CTRT_NO)
);

-- 고객상담/민원
CREATE TABLE TB_CS_TCKT (
    TCKT_ID     TEXT    NOT NULL PRIMARY KEY,   -- 상담티켓ID
    CUST_ID     TEXT    NOT NULL,               -- 고객ID
    CTRT_NO     TEXT,                           -- 관련계약번호
    RCPT_DT     TEXT    NOT NULL,               -- 접수일자
    CMPL_DT     TEXT,                           -- 처리완료일자
    CHNL_CD     TEXT    NOT NULL,               -- 접수채널코드
    CTGY_CD     TEXT    NOT NULL,               -- 민원유형코드
    PROC_STAT_CD TEXT   NOT NULL,               -- 처리상태코드
    CNTN        TEXT,                           -- 상담내용
    SATIS_SCR   INTEGER,                        -- 만족도점수
    FOREIGN KEY (CUST_ID) REFERENCES TB_CUST(CUST_ID),
    FOREIGN KEY (CTRT_NO) REFERENCES TB_CTRT(CTRT_NO)
);

CREATE INDEX IX_CTRT_CUST  ON TB_CTRT(CUST_ID);
CREATE INDEX IX_CTRT_DT    ON TB_CTRT(CTRT_DT);
CREATE INDEX IX_CTRT_STAT  ON TB_CTRT(CTRT_STAT_CD);
CREATE INDEX IX_CLM_CTRT   ON TB_CLM(CTRT_NO);
CREATE INDEX IX_CLM_DT     ON TB_CLM(CLM_DT);
CREATE INDEX IX_PAY_CTRT   ON TB_PAY(CTRT_NO);
CREATE INDEX IX_CVRG_CTRT  ON TB_CVRG(CTRT_NO);
CREATE INDEX IX_TCKT_CUST  ON TB_CS_TCKT(CUST_ID);
