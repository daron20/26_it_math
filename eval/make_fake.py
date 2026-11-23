"""eval.py 동작 확인용 가짜 데이터 생성기. 실제 평가에는 쓰지 않는다.

    python make_fake.py --out sample
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["페트", "캔", "유리", "종이컵", "기타"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample")
    ap.add_argument("--scenes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 정답: 장면 단위로 70/15/15 분할
    split = rng.choice(["train", "val", "test"], a.scenes, p=[.7, .15, .15])
    gt = []
    for s in range(a.scenes):
        lv = rng.choice(["L0", "L1", "L2", "L3"], p=[.4, .3, .2, .1])
        for k in range(rng.integers(3, 9)):
            y = rng.choice(CLASSES, p=[.32, .22, .16, .18, .12])
            hard = (y in ("페트", "유리") and rng.random() < .35) or rng.random() < .05
            gt.append(dict(object_id=f"s{s:03d}_o{k}", scene_id=f"s{s:03d}", split=split[s],
                           y_true=y, hard_case=hard, overlap=lv))
    gt = pd.DataFrame(gt)
    gt.to_csv(out / "gt.csv", index=False, encoding="utf-8-sig")

    # 모델 두 개: 품질만 다르게
    for name, q in [("model_a", 3.2), ("model_b", 2.4)]:
        rows = []
        for r in gt.itertuples():
            miss_p = {"L0": .01, "L1": .02, "L2": .05, "L3": .12}[r.overlap]
            if rng.random() < miss_p:
                rows.append(dict(model=name, object_id=r.object_id, scene_id=r.scene_id, detected=False))
                continue
            qq = q * (0.6 if r.hard_case else 1.0) * (0.8 if r.overlap in ("L2", "L3") else 1.0)
            logit = rng.normal(0, 1, 5)
            logit[CLASSES.index(r.y_true)] += qq
            if r.y_true == "유리":            # 유리를 페트로 헷갈리는 경향
                logit[0] += 0.8
            p = np.exp(logit) / np.exp(logit).sum()
            rows.append(dict(model=name, object_id=r.object_id, scene_id=r.scene_id, detected=True,
                             **{f"p_{c}": v for c, v in zip(CLASSES, p)}))
        # 오검출: 정답 없는 곳에서 나온 예측 (object_id 비움)
        for s in gt.scene_id.unique():
            if rng.random() < .1:
                logit = rng.normal(0, 1.2, 5)
                logit[4] += 1.0
                p = np.exp(logit) / np.exp(logit).sum()
                rows.append(dict(model=name, object_id="", scene_id=s, detected=True,
                                 **{f"p_{c}": v for c, v in zip(CLASSES, p)}))
        pd.DataFrame(rows).to_csv(out / f"preds_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"→ {out}: gt.csv, preds_model_a.csv, preds_model_b.csv")


if __name__ == "__main__":
    main()
