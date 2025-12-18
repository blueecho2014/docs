# API 및 데이터 흐름 정의서(요약본) — 1차년도 “구조 설계(Agent+RAG)” 증빙용

## 1. 목적
- 1차년도에 수행한 “LLM 기반 대화형 설명 인터페이스 시스템 구조 설계(Agent + RAG)”의 **검증 가능한 증빙**으로,
  - (1) 주요 컴포넌트 간 **데이터 흐름**
  - (2) 핵심 API의 **입·출력 스키마(요약)**
  - (3) 운영/로그 관점의 **비기능 요구**
  를 명세한다.

## 2. 시스템 구성(요약)
- 클라이언트: **xUI(대화형 설명 UI)**
- 백엔드: **Platform API**, 인증/권한, 로그/모니터링
- AI 서비스: **xAI 예측**, **해석/설명(SHAP)**, **sLLM**, **LLM Agent**, **RAG Retriever**
- 데이터: 학습/로그 DB, 분석 DB, 해석/설명 구조화 DB, Vector DB, 문서 저장소

> 구성도는 `docs/부록-B2_네트워크구성도_요약(mermaid).md`(부록 B-2) 참조.

## 3. 데이터 흐름(시나리오)
### 3.1 시나리오 A: “예측 결과 조회 → 근거 기반 설명”
1) 사용자(xUI)에서 특정 학습자/문항/세션을 선택
2) xUI → Platform API: 예측 요청
3) Platform API → xAI 예측 서비스: 예측 수행(확률/점수 포함)
4) Platform API → 해석/설명 서비스: SHAP 계산(또는 사전 계산 결과 조회)
5) 해석/설명 서비스 → sLLM/Agent: SHAP 요약 + 사용자 맥락을 전달하여 설명문 생성
6) Agent → RAG Retriever: 필요한 도메인 지식(가이드/교육 문서/FAQ 등) 검색
7) RAG Retriever → Vector DB: Top-k 문서/근거 반환
8) Agent: 근거 인용 포함 설명 생성 → Platform API → xUI 응답

### 3.2 시나리오 B: “대화형 질의(Why?/How?)”
1) 사용자 질문(Why 틀렸어?/어떤 개념이 취약해?) 입력
2) xUI → `POST /chat/explain`
3) Agent가 현재 컨텍스트(예측/SHAP/학습이력) + RAG 근거를 결합해 답변
4) 응답에는 **근거(문서/규칙/해석 결과)**를 포함

## 4. 핵심 API 정의(요약)
> 아래는 “구조 설계 증빙” 목적의 요약 명세이며, 2차년도에 상세 스펙(오류코드/페이징/권한 정책)을 확정한다.

### 4.1 예측 API
**Endpoint**
- `POST /api/v1/predict`

**Request (예시)**
- `member_id`: int
- `rcset_id`: int
- `question_id`: int
- `features`: object (duration, Difficulty, Discrimination 등)

**Response (예시)**
- `prediction`: int (0/1 또는 등급)
- `probability`: float
- `model_version`: string
- `trace_id`: string

### 4.2 해석(설명) API — SHAP 기반
**Endpoint**
- `POST /api/v1/explain`

**Request (예시)**
- `trace_id` 또는 `input`: object
- `top_n`: int (기본 20)

**Response (예시)**
- `top_features`: array[{feature: string, mean_abs_shap: float}]
- `artifacts`: object
  - `topn_csv`: string (파일명/경로)
  - `topn_png`: string (파일명/경로)
- `explain_version`: string

### 4.3 대화형 설명 API — Agent + RAG
**Endpoint**
- `POST /api/v1/chat/explain`

**Request (예시)**
- `question`: string
- `context`:
  - `member_id`, `rcset_id`, `question_id`
  - `prediction`, `top_features`

**Response (예시)**
- `answer`: string
- `citations`: array[{source_id: string, snippet: string}]
- `trace_id`: string

## 5. 로그/모니터링(비기능 요구) — 요약
- **추적성**: 모든 요청에 `trace_id` 부여(예측→해석→대화 응답까지 연결)
- **감사 로그**: 사용자/권한/조회 범위(학습자 식별자) 기록
- **품질 지표**: 응답시간, 실패율, RAG 히트율, 근거 포함 비율

## 6. 1차년도 산출물(연계 증빙)
- 1차년도 파일럿 결과 산출물: `shap_top20_member*.csv/png`, `example*.json`
- 관련 문서: 부록 B-1(시스템 설계서 요약), 부록 B-2(구성도), 부록 B-3(xAI 모델 정의서)


