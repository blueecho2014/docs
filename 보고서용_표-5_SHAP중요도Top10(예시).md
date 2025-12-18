# 표 5. SHAP 중요도 Top-10(절대값 평균) 예시

아래 표는 HWP에 **표로 붙여넣기** 용도로 정리한 요약본입니다.  
(산출물: `shap_top20_member0.csv`, `shap_top20_member1.csv`)

## (A) member0 예시 — Top 10
| 순위 | Feature | mean(|SHAP|) |
|---:|---|---:|
| 1 | Difficulty | 0.05122 |
| 2 | member_id | 0.04904 |
| 3 | duration | 0.02468 |
| 4 | mode_id | 0.01893 |
| 5 | type | 0.01820 |
| 6 | Discrimination | 0.01770 |
| 7 | edu_crs_factor_sort_sn | 0.01104 |
| 8 | question_id | 0.01014 |
| 9 | response_count | 0.00991 |
| 10 | rcset_id | 0.00721 |

## (B) member1 예시 — Top 10
| 순위 | Feature | mean(|SHAP|) |
|---:|---|---:|
| 1 | Difficulty | 0.06711 |
| 2 | member_id | 0.03500 |
| 3 | mode_id | 0.02265 |
| 4 | duration | 0.01983 |
| 5 | type | 0.01975 |
| 6 | Discrimination | 0.01512 |
| 7 | edu_crs_factor_sort_sn | 0.01080 |
| 8 | question_id | 0.00888 |
| 9 | response_count | 0.00849 |
| 10 | conts_dtl_qitem_type_se_cd | 0.00651 |

> 주: 상기 값은 내부 파일럿 데이터에서 특정 사용자 샘플(member_id=0/1)에 대한 예시이며, 2차년도에 평가셋 고정 후 지표를 공식화할 예정.


