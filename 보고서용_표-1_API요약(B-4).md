# 표 1. 핵심 API 요약(1차년도 구조 설계 증빙, 부록 B-4 기반)

아래 표는 HWP에 **표로 붙여넣기** 용도로 정리한 요약본입니다.

| 구분 | API(Endpoint) | 목적 | 입력(요약) | 출력(요약) | 비고 |
|---|---|---|---|---|---|
| 예측 | `POST /api/v1/predict` | 학습자/문항 맥락 기반 예측 수행 | `member_id`, `rcset_id`, `question_id`, `features` | `prediction`, `probability`, `model_version`, `trace_id` | 모든 요청에 `trace_id` 부여 |
| 해석(SHAP) | `POST /api/v1/explain` | 예측 근거(Top-N) 산출/조회 | `trace_id` 또는 `input`, `top_n` | `top_features[]`, `artifacts(topn_csv/topn_png)`, `explain_version` | SHAP Top-N CSV/PNG 산출 |
| 대화형 설명 | `POST /api/v1/chat/explain` | Agent+RAG 기반 질의응답/설명 생성 | `question`, `context(member/문항/예측/Top feature)` | `answer`, `citations[]`, `trace_id` | 근거(citations) 포함 응답 |

| 운영/비기능 항목 | 정의(요약) |
|---|---|
| 추적성 | 예측→해석→대화 응답 전 구간을 `trace_id`로 연결 |
| 감사 로그 | 사용자/권한/조회 범위(학습자 식별자) 기록 |
| 품질 지표 | 응답시간, 실패율, RAG hit율, 근거 포함률 |


