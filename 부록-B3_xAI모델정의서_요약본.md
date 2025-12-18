# xAI 모델 정의서(요약본) — 1차년도 최소 증빙용

## 1. 목적
- 1차년도에 수행한 xAI 기반 예측/설명(해석) 모델 개발 내용을 “모델 정의서” 형태로 정리하여 연차보고서 증빙(부록)으로 첨부

## 2. 문제 정의(1차년도 파일럿)
- **목표**: 학습자의 문항 반응 데이터를 기반으로 정답 여부(또는 성취/취약 관련 타겟)를 예측하고, 예측 근거를 SHAP 기반으로 설명
- **타겟(현 구현)**: `answer`(정답 여부) → 이진 분류

## 3. 데이터/피처 정의
### 3.1 원천 데이터
- `data/toctoc_1216.csv`: 학습 로그/답안 데이터
- `data/irt_1216_all.csv`: IRT 파라미터 데이터

### 3.2 전처리/결합
- `rcset_id`, `question_id` 기준 inner join
- 불필요 컬럼 제거(세부 컬럼은 `Xai_ex.py` 참조)
- `member_id`, `question_id`는 충돌 방지를 위해 **연속 정수로 재매핑**

### 3.3 입력 피처(모델 입력 X)
- 식별/세션: `member_id`, `rcset_id`, `question_id`
- 행동/로그: `duration`, `response_count` 등
- 학습 맥락: `grade_cd`, `semstr_cd`, `lsn_cd`, `mode_id`, `conts_dtl_qitem_type_se_cd` 등
- IRT: `Discrimination`, `Difficulty`

## 4. 모델 정의
- **모델 타입**: RandomForestClassifier (scikit-learn)
- **주요 하이퍼파라미터(현 구현)**: `n_estimators=100`, `random_state=42`
- **학습/검증**: train/test split (test_size=0.2, random_state=42)

## 5. 성능 지표(현 구현)
- **Accuracy**: 0.90760759261228
- **AUC**: 0.662563032913503

> 주: 본 값은 1차년도 파일럿 결과이며, 계획서 성능지표(수준진단/취약진단/해석 모델 정확도 등)로의 정합화 및 공인시험성적서 기반 공식 검증은 2차년도에 수행 예정.

## 6. 설명(해석) 모델 정의
- **설명 기법**: SHAP(TreeExplainer)
- **설명 대상**: 분류 모델의 class=1(정답) 예측에 대한 feature 기여도
- **집계 방식(Top-N)**: feature별 \(mean(|SHAP|)\) 기준 상위 N개 산출

### 6.1 설명 산출물(현 구현)
- Top-20 CSV/PNG
  - `shap_top20_member0.csv`, `shap_top20_member0.png`
  - `shap_top20_member1.csv`, `shap_top20_member1.png`
- 입력 샘플 JSON(재현/검증용)
  - `example1.json`(member_id==0), `example2.json`(member_id==1)

## 7. 재현 방법(간단)
1) 필요한 패키지 설치: `shap`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
2) 프로젝트 루트에서 실행:
   - `py -3 Xai_ex.py`
3) 결과 파일 생성 확인:
   - `shap_top20_member*.csv/png`, `example*.json`

## 8. 2차년도 확장/고도화 항목(요약)
- **최적화**: 지표 정합화, 평가셋 고정, 모델 튜닝/비교
- **경량화**: 압축/증류/추론 최적화 및 응답시간 목표 달성
- **설명 고도화**: sLLM/Agent 기반 근거 제시형 설명, RAG 연계, SaaS 운영화


