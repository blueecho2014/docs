# 부록 B-6: XAI 결과 LLM 기반 해석 보고서

> 본 문서는 1차년도 수행 결과인 XAI 모델의 SHAP 중요도 분석 및 ROC 커브 결과를 LLM으로 해석한 증적 자료입니다.

**생성 일시**: 2025년 12월 22일 04:27:20  
**LLM 모델**: Google Gemini / gemini-2.5-flash  
**분석 대상**: SHAP 중요도 Top-20 (member0, member1), ROC 커브 (AUC=0.8669579415213406)

---

## 1. 입력 데이터 요약

### 1.1 모델 성능 지표

- **정확도(Accuracy)**: 0.9076
- **AUC**: 0.8670


### 1.2 ROC 커브
- **AUC**: 0.8670
- **FPR/TPR 좌표 수**: 102개


### 1.3 SHAP 중요도 Top-10 요약

#### member_id=0 샘플
| 순위 | 피처 | mean(|SHAP|) |
|:---:|:---|:---:|
| 1 | Difficulty | 0.051220 |
| 2 | member_id | 0.049035 |
| 3 | duration | 0.024684 |
| 4 | mode_id | 0.018927 |
| 5 | type | 0.018196 |
| 6 | Discrimination | 0.017703 |
| 7 | edu_crs_factor_sort_sn | 0.011036 |
| 8 | question_id | 0.010144 |
| 9 | response_count | 0.009913 |
| 10 | rcset_id | 0.007209 |

#### member_id=1 샘플
| 순위 | 피처 | mean(|SHAP|) |
|:---:|:---|:---:|
| 1 | Difficulty | 0.067111 |
| 2 | member_id | 0.034997 |
| 3 | mode_id | 0.022652 |
| 4 | duration | 0.019831 |
| 5 | type | 0.019745 |
| 6 | Discrimination | 0.015124 |
| 7 | edu_crs_factor_sort_sn | 0.010804 |
| 8 | question_id | 0.008877 |
| 9 | response_count | 0.008487 |
| 10 | conts_dtl_qitem_type_se_cd | 0.006514 |


---

## 2. LLM 기반 해석 결과

## 교육 데이터 분석을 위한 XAI 모델 분석 보고서

### 서론
본 보고서는 교육 데이터 분석을 위해 개발된 XAI(설명 가능한 AI) 모델의 성능 지표와 SHAP(SHapley Additive exPlanations) 중요도 분석 결과를 바탕으로, 모델의 예측 메커니즘을 이해하고 교육 현장에서 활용 가능한 인사이트

---

## 3. 참고 자료

- SHAP 중요도 상세 데이터: `shap_top20_member0.csv`, `shap_top20_member1.csv`
- ROC 커브 데이터: `roc_curve.json`
- SHAP 중요도 시각화: `shap_top20_member0.png`, `shap_top20_member1.png`
- ROC 커브 시각화: `roc_curve.png`

---

**주**: 본 해석은 LLM(Google Gemini/gemini-2.5-flash)을 활용하여 자동 생성되었으며, 
교육 데이터 분석 전문가의 관점에서 XAI 결과를 해석한 내용입니다. 
실제 교육 현장 적용 시 추가 검증이 필요할 수 있습니다.
