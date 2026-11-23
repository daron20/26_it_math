# 분류 단계 평가 스크립트

모델 종류와 상관없이, 아래 형식의 CSV만 만들면 같은 기준으로 평가한다.

```
python eval.py --gt gt.csv --preds preds_A.csv preds_B.csv --out results
```

동작 확인용 가짜 데이터: `python make_fake.py --out sample`

## 1. 정답 파일 `gt.csv` (정답 물체 하나 = 한 행)

| 컬럼 | 필수 | 내용 |
|---|---|---|
| object_id | ✔ | 물체 ID (예측 파일과 연결하는 키) |
| scene_id | ✔ | 장면 ID |
| split | ✔ | `val` / `test` (`train` 행은 무시) — 장면 단위로 나눌 것 |
| y_true | ✔ | 페트·캔·유리·종이컵·기타 |
| hard_case |  | 투명 PET·검은 물체면 True |
| overlap |  | L0~L3 |

## 2. 예측 파일 (모델마다 하나, adapter가 만든다)

| 컬럼 | 내용 |
|---|---|
| model | 모델 이름 (없으면 파일 이름) |
| object_id | 짝지은 정답 물체 ID. **정답에 없는 예측(오검출)은 빈 칸** |
| scene_id | 장면 ID (오검출 행은 필수) |
| detected | 못 찾았으면 False (분류 모델은 항상 True) |
| p_페트 … p_기타 | 5클래스 확률 (합이 1이 아니면 자동 정규화) |

- 같은 물체가 여러 프레임에 나오면 **안정화한 최종 결과 1행**만 넣는다.
- 검출형 모델: 정답 마스크와 IoU ≥ 0.5로 1:1 매칭해 object_id를 붙인다.
- 정답 물체에 예측 행이 없으면 미검출로 센다.

## 3. 하는 일

1. 판정: 최고 확률 < threshold → 확인필요, 아니면 최고 클래스 (기타 포함)
2. val에서 threshold를 0~1로 바꿔 가며, 재활용 4클래스 precision이 모두 `--target` 이상인 값 중 확인필요 비율이 가장 낮은 값을 고른다 (모델마다 따로)
3. 그 값을 test에 그대로 적용해 보고한다

## 4. 출력 (`--out` 폴더)

| 파일 | 내용 |
|---|---|
| report.md | 모델 비교표 + 모델별 클래스 표(n, ±오차, Wilson 구간), 투명·검정 행, 겹침 단계별, 운영점 비교 |
| pr_<모델>.png | test PR 곡선 + 운영점, threshold별 확인필요 비율 |
| confusion_<모델>.png/csv | 혼동행렬 (확인필요·미검출 열, 배경 행), 유리→페트 강조 |
| compare.csv / summary.json | 모델 비교용 숫자 |
| sweep_<모델>_val/test.csv | threshold별 전체 수치 |

## 5. 옵션

| 옵션 | 기본값 |
|---|---|
| --target | 0.90 (운영점 기준) |
| --extra-targets | 0.95 (비교용 운영점) |
| --classes | 페트,캔,유리,종이컵,기타 |
| --recycle | 페트,캔,유리,종이컵 |
| --highlight | 유리:페트 |
| --grid-step | 0.005 |

클래스가 3종으로 줄면 `--classes`, `--recycle`만 바꾸면 된다.
