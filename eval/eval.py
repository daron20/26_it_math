#!/usr/bin/env python
"""분류 단계 평가 스크립트 (모델 무관).

정답 파일 하나와 모델별 예측 파일(표준 형식)을 받아,
val에서 운영점(threshold)을 고르고 test에서 지표를 보고한다.

    python eval.py --gt gt.csv --preds preds_a.csv preds_b.csv --out results

입력 형식은 README.md 참고.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REVIEW = "확인필요"   # threshold 미만 → 사람에게 넘김
MISS = "미검출"       # 검출 단계에서 놓친 정답 물체
BG = "배경"           # 정답이 없는 곳에서 나온 예측(오검출)


# ───────────────────────── 입력 ─────────────────────────

def load_inputs(gt_path, pred_paths, classes):
    gt = pd.read_csv(gt_path, dtype={"object_id": str, "scene_id": str})
    need = {"object_id", "scene_id", "split", "y_true"}
    if missing := need - set(gt.columns):
        sys.exit(f"[오류] 정답 파일에 컬럼이 없음: {sorted(missing)}")
    if gt.object_id.duplicated().any():
        sys.exit("[오류] 정답 파일에 object_id 중복이 있음")
    bad = set(gt.y_true) - set(classes)
    if bad:
        sys.exit(f"[오류] 정답에 모르는 클래스: {sorted(bad)} (허용: {classes})")
    gt["split"] = gt["split"].str.lower()
    gt["hard_case"] = gt.get("hard_case", False)
    gt["hard_case"] = gt["hard_case"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])
    gt["overlap"] = gt.get("overlap", "").fillna("").astype(str) if "overlap" in gt else ""

    pcols = [f"p_{c}" for c in classes]
    frames = []
    for p in pred_paths:
        df = pd.read_csv(p, dtype={"object_id": str, "scene_id": str})
        if "model" not in df:
            df["model"] = Path(p).stem
        if missing := set(pcols) - set(df.columns):
            sys.exit(f"[오류] {p}: 확률 컬럼이 없음 {sorted(missing)}")
        if "detected" not in df:
            df["detected"] = True
        df["detected"] = df["detected"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        frames.append(df)
    return gt, pd.concat(frames, ignore_index=True)


def build_rows(gt, pm, classes):
    """한 모델의 예측을 정답과 합쳐 '평가 행' 표를 만든다.

    - 정답 물체마다 한 행. 예측이 없거나 detected=False면 미검출.
    - 정답에 없는 object_id(또는 빈 값)를 가진 예측은 y_true=배경인 오검출 행.
    """
    pcols = [f"p_{c}" for c in classes]
    pm = pm[pm.detected].copy()
    pm["object_id"] = pm["object_id"].fillna("").astype(str)
    known = set(gt.object_id)

    matched = pm[pm.object_id.isin(known)]
    if matched.object_id.duplicated().any():
        dups = matched.object_id[matched.object_id.duplicated()].unique()[:5]
        sys.exit(f"[오류] 한 정답 물체에 예측이 둘 이상: {list(dups)} … adapter에서 1:1로 짝지어야 함")

    rows = gt.merge(matched[["object_id"] + pcols], on="object_id", how="left")
    rows["detected"] = rows[pcols].notna().all(axis=1)

    fp = pm[~pm.object_id.isin(known)]
    if len(fp):
        if "scene_id" not in fp or fp.scene_id.isna().any():
            sys.exit("[오류] 오검출 행(정답에 없는 object_id)은 scene_id가 있어야 split을 알 수 있음")
        scene_split = gt.drop_duplicates("scene_id").set_index("scene_id")["split"]
        fp = fp.assign(y_true=BG, split=fp.scene_id.map(scene_split),
                       hard_case=False, overlap="", detected=True)
        fp = fp[fp.split.notna()]
        rows = pd.concat([rows, fp[rows.columns]], ignore_index=True)

    P = rows[pcols].to_numpy(float)
    det = rows["detected"].to_numpy()
    s = P[det].sum(1, keepdims=True)
    if (s <= 0).any():
        sys.exit("[오류] 확률 합이 0 이하인 행이 있음")
    P[det] = P[det] / s           # 합이 1이 아니면 정규화
    return rows.reset_index(drop=True), P, det


# ───────────────────────── 판정·지표 ─────────────────────────

def decide(P, det, classes, t):
    """판정 규칙: 최고 확률 < t → 확인필요, 아니면 최고 클래스(기타 포함)."""
    pred = np.full(len(P), MISS, dtype=object)
    if det.any():
        Pd = P[det]
        lab = np.array(classes, dtype=object)[Pd.argmax(1)]
        lab[Pd.max(1) < t] = REVIEW
        pred[det] = lab
    return pred


def class_metrics(y, pred, recycle):
    out = {}
    for c in recycle:
        tp = int(((y == c) & (pred == c)).sum())
        n_pred = int((pred == c).sum())
        n_true = int((y == c).sum())
        out[c] = dict(tp=tp, n_pred=n_pred, n_true=n_true,
                      P=tp / n_pred if n_pred else np.nan,
                      R=tp / n_true if n_true else np.nan)
    return out


def rates(y, pred):
    seen = pred != MISS                          # 분류기가 판정한 물체(오검출 포함)
    real = y != BG                               # 실제 정답 물체
    return dict(review=(pred == REVIEW).sum() / max(seen.sum(), 1),
                miss=((pred == MISS) & real).sum() / max(real.sum(), 1),
                n_seen=int(seen.sum()), n_real=int(real.sum()),
                n_bg=int((y == BG).sum()))


def min_p(m):
    ps = [v["P"] for v in m.values()]
    return np.nan if any(np.isnan(ps)) else min(ps)   # 예측 0개인 클래스가 있으면 기준 미달


def f_beta(m, beta=0.5):
    b2, fs = beta ** 2, []
    for v in m.values():
        p, r = v["P"], v["R"]
        fs.append(0.0 if np.isnan(p) or np.isnan(r) or (b2 * p + r) == 0
                  else (1 + b2) * p * r / (b2 * p + r))
    return float(np.mean(fs))


def wald(p, n):
    return np.nan if np.isnan(p) or n == 0 else 1.96 * np.sqrt(p * (1 - p) / n)


def wilson(p, n, z=1.96):
    if np.isnan(p) or n == 0:
        return (np.nan, np.nan)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def sweep(y, P, det, classes, recycle, grid):
    recs = []
    for t in grid:
        pred = decide(P, det, classes, t)
        m = class_metrics(y, pred, recycle)
        row = dict(t=t, review=rates(y, pred)["review"], minP=min_p(m))
        for c in recycle:
            row[f"P_{c}"], row[f"R_{c}"] = m[c]["P"], m[c]["R"]
        recs.append(row)
    return pd.DataFrame(recs)


def pick_threshold(sw, target):
    """minP ≥ target 중 확인필요 비율이 가장 낮은 t. 없으면 minP가 가장 높은 t (met=False)."""
    ok = sw[sw.minP >= target]
    if len(ok):
        r = ok.sort_values(["review", "t"]).iloc[0]
        return float(r.t), True
    if sw.minP.notna().any():
        return float(sw.loc[sw.minP.idxmax(), "t"]), False
    return 0.5, False


def subgroup_row(y, pred, mask, recycle):
    """부분집합(투명·검정, 겹침 단계 등)을 재활용 4클래스 묶어서(micro) 집계."""
    yy, pp = y[mask], pred[mask]
    pred_rec = np.isin(pp, recycle)
    true_rec = np.isin(yy, recycle)
    correct = (yy == pp) & pred_rec
    n_pred, n_true = int(pred_rec.sum()), int(true_rec.sum())
    seen = pp != MISS
    return dict(n=int(mask.sum()), n_true=n_true, n_pred=n_pred,
                P=correct.sum() / n_pred if n_pred else np.nan,
                R=correct.sum() / n_true if n_true else np.nan,
                review=(pp == REVIEW).sum() / max(seen.sum(), 1))


def confusion(y, pred, classes):
    rows = list(classes) + ([BG] if (y == BG).any() else [])
    cols = list(classes) + [REVIEW] + ([MISS] if (pred == MISS).any() else [])
    cm = pd.DataFrame(0, index=rows, columns=cols)
    for a, b in zip(y, pred):
        cm.loc[a, b] += 1
    return cm


# ───────────────────────── 출력 ─────────────────────────

def pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def setup_font():
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for n in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        if n in names:
            plt.rcParams["font.family"] = n
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_confusion(cm, path, title, highlight):
    fig, ax = plt.subplots(figsize=(1.1 * cm.shape[1] + 2, 0.8 * cm.shape[0] + 1.6))
    norm = cm.div(cm.sum(1).replace(0, 1), axis=0)
    ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.iat[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=10,
                    color="white" if norm.iat[i, j] > 0.6 else "black")
    ax.set_xticks(range(cm.shape[1]), cm.columns)
    ax.set_yticks(range(cm.shape[0]), cm.index)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    for a, b in highlight:
        if a in cm.index and b in cm.columns:
            i, j = cm.index.get_loc(a), cm.columns.get_loc(b)
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="red", lw=2.5))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(sw_test, ops, recycle, path, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, c in enumerate(recycle):
        d = sw_test[[f"R_{c}", f"P_{c}"]].dropna()
        a1.plot(d[f"R_{c}"], d[f"P_{c}"], color=colors[k], label=c)
    markers = ["o", "s", "^", "D"]
    for k, (target, t) in enumerate(ops.items()):
        r = sw_test.iloc[(sw_test.t - t).abs().argmin()]
        for j, c in enumerate(recycle):
            a1.scatter(r[f"R_{c}"], r[f"P_{c}"], marker=markers[k % 4], color=colors[j],
                       edgecolor="black", zorder=3, s=45,
                       label=f"운영점 P≥{target:.0%} (t={t:.3f})" if j == 0 else None)
        a2.axvline(t, ls="--", color="gray")
        a2.text(t, 0.98, f" P≥{target:.0%}", va="top", fontsize=8)
    a1.set_xlabel("Recall")
    a1.set_ylabel("Precision")
    a1.set_xlim(0, 1.02)
    a1.set_ylim(0, 1.02)
    a1.grid(alpha=.3)
    a1.legend(fontsize=8, loc="lower left")
    a1.set_title("PR 곡선 (공통 threshold를 바꿔 가며)")
    a2.plot(sw_test.t, sw_test.review, color="black", label="확인필요 비율")
    a2.plot(sw_test.t, sw_test.minP, color="tab:red", label="4클래스 최소 precision")
    a2.set_xlabel("threshold")
    a2.set_ylim(0, 1.02)
    a2.grid(alpha=.3)
    a2.legend(fontsize=8)
    a2.set_title("threshold별 확인필요 비율")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def safe(name):
    return re.sub(r"[^\w.-]+", "_", str(name))


# ───────────────────────── 메인 ─────────────────────────

def evaluate_model(name, gt, pm, a, out):
    classes, recycle = a.classes, a.recycle
    rows, P, det = build_rows(gt, pm, classes)
    y = rows["y_true"].to_numpy(dtype=object)
    sp = rows["split"].to_numpy()
    v, te = sp == "val", sp == "test"
    if not v.any() or not te.any():
        sys.exit(f"[오류] {name}: val 또는 test 행이 없음")

    grid = np.round(np.arange(0, 1 + 1e-9, a.grid_step), 6)
    sw_val = sweep(y[v], P[v], det[v], classes, recycle, grid)
    sw_test = sweep(y[te], P[te], det[te], classes, recycle, grid)
    sw_val.to_csv(out / f"sweep_{safe(name)}_val.csv", index=False, encoding="utf-8-sig")
    sw_test.to_csv(out / f"sweep_{safe(name)}_test.csv", index=False, encoding="utf-8-sig")

    targets = [a.target] + [x for x in a.extra_targets if x != a.target]
    ops, met = {}, {}
    for tg in targets:
        ops[tg], met[tg] = pick_threshold(sw_val, tg)
    t = ops[a.target]

    # test에서 운영점 지표
    yt, Pt, dt = y[te], P[te], det[te]
    pred = decide(Pt, dt, classes, t)
    m = class_metrics(yt, pred, recycle)
    r = rates(yt, pred)
    rt = rows[te].reset_index(drop=True)
    cm = confusion(yt, pred, classes)
    cm.to_csv(out / f"confusion_{safe(name)}.csv", encoding="utf-8-sig")
    plot_confusion(cm, out / f"confusion_{safe(name)}.png",
                   f"{name} · test · t={t:.3f}", a.highlight)
    plot_pr(sw_test, ops, recycle, out / f"pr_{safe(name)}.png", f"{name} · test")

    hard = subgroup_row(yt, pred, rt.hard_case.to_numpy(), recycle)
    levels = sorted(x for x in rt.overlap.unique() if x)
    by_overlap = {lv: subgroup_row(yt, pred, (rt.overlap == lv).to_numpy(), recycle) for lv in levels}

    other_ops = {}
    for tg in targets:
        pr_ = decide(Pt, dt, classes, ops[tg])
        mm = class_metrics(yt, pr_, recycle)
        other_ops[tg] = dict(t=ops[tg], met_on_val=met[tg], review=rates(yt, pr_)["review"],
                             minP=min_p(mm), **{f"P_{c}": mm[c]["P"] for c in recycle},
                             **{f"R_{c}": mm[c]["R"] for c in recycle})

    return dict(model=name, t=t, met_on_val=met[a.target], target=a.target,
                classes=m, rates=r, hard=hard, by_overlap=by_overlap,
                macro_f05=f_beta(m), minP=min_p(m), ops=other_ops,
                n_val=int(v.sum()), n_test=int(te.sum()))


def write_report(results, a, out):
    L = ["# 분류 단계 평가 결과", "",
         f"- 운영점 기준: val에서 재활용 4클래스 precision이 모두 ≥ {a.target:.0%}인 threshold 중 "
         "확인필요 비율이 가장 낮은 값 → test에 그대로 적용",
         "- 오차 범위: ±1.96·√(p(1−p)/n), 괄호 안은 Wilson 95% 구간",
         f"- 확인필요 비율 분모 = 분류기가 판정한 물체 수(미검출 제외, 오검출 포함)", ""]

    # 모델 비교표
    hdr = ["모델", "threshold", "확인필요", "미검출"] + [f"{c} P/R" for c in a.recycle] + ["Macro F0.5"]
    rows = []
    for R in results:
        rows.append([R["model"], f"{R['t']:.3f}" + ("" if R["met_on_val"] else " ⚠"),
                     pct(R["rates"]["review"]), pct(R["rates"]["miss"])] +
                    [f"{pct(R['classes'][c]['P'])} / {pct(R['classes'][c]['R'])}" for c in a.recycle] +
                    [pct(R["macro_f05"])])
    L += ["## 모델 비교 (test)", "", md_table(hdr, rows), ""]
    if any(not R["met_on_val"] for R in results):
        L += [f"⚠ val에서 기준(≥ {a.target:.0%})을 만족하는 threshold가 없어 최소 precision이 가장 높은 지점을 썼다.", ""]

    for R in results:
        r = R["rates"]
        L += [f"## {R['model']}", "",
              f"운영점: threshold = {R['t']:.3f} · 테스트 물체 수 = {r['n_real']}"
              f" (오검출 {r['n_bg']}개 별도) · 확인필요 비율 = {pct(r['review'])}"
              f" · 미검출 = {pct(r['miss'])}", ""]
        rows = []
        for c in a.recycle:
            v = R["classes"][c]
            wp, wr = wilson(v["P"], v["n_pred"]), wilson(v["R"], v["n_true"])
            rows.append([c, v["n_true"], v["n_pred"],
                         f"{pct(v['P'])} ±{pct(wald(v['P'], v['n_pred']))} ({pct(wp[0])}–{pct(wp[1])})",
                         f"{pct(v['R'])} ±{pct(wald(v['R'], v['n_true']))} ({pct(wr[0])}–{pct(wr[1])})", ""])
        h = R["hard"]
        rows.append(["투명·검정 물체 (별도 집계)", h["n_true"], h["n_pred"], pct(h["P"]), pct(h["R"]),
                     f"확인필요 {pct(h['review'])}"])
        L += [md_table(["클래스", "실제 개수", "예측 개수", "Precision", "Recall", "비고"], rows), ""]

        if R["by_overlap"]:
            rows = [[lv, g["n"], pct(g["P"]), pct(g["R"]), pct(g["review"])]
                    for lv, g in R["by_overlap"].items()]
            L += ["겹침 단계별 (재활용 4클래스 묶음)", "",
                  md_table(["단계", "물체 수", "Precision", "Recall", "확인필요"], rows), ""]

        rows = [[f"P≥{tg:.0%}", f"{o['t']:.3f}" + ("" if o["met_on_val"] else " ⚠"),
                 pct(o["review"]), pct(o["minP"])] for tg, o in R["ops"].items()]
        L += ["운영점 비교 (val에서 고른 t → test 값)", "",
              md_table(["기준", "threshold", "확인필요", "최소 precision"], rows), "",
              f"![PR](pr_{safe(R['model'])}.png)", "",
              f"![혼동행렬](confusion_{safe(R['model'])}.png)", ""]

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")

    comp = []
    for R in results:
        row = dict(model=R["model"], threshold=R["t"], met_on_val=R["met_on_val"],
                   review=R["rates"]["review"], miss=R["rates"]["miss"],
                   minP=R["minP"], macro_f05=R["macro_f05"])
        for c in a.recycle:
            row[f"P_{c}"], row[f"R_{c}"] = R["classes"][c]["P"], R["classes"][c]["R"]
        comp.append(row)
    pd.DataFrame(comp).to_csv(out / "compare.csv", index=False, encoding="utf-8-sig")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (float, np.floating)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    (out / "summary.json").write_text(json.dumps(clean({R["model"]: R for R in results}),
                                                 ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="분류 단계 평가 (모델 무관)")
    ap.add_argument("--gt", required=True, help="정답 CSV (object_id, scene_id, split, y_true[, hard_case, overlap])")
    ap.add_argument("--preds", nargs="+", required=True, help="표준 예측 CSV (여러 개 가능)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--classes", default="페트,캔,유리,종이컵,기타", help="모델 출력 클래스 (확률 컬럼 순서)")
    ap.add_argument("--recycle", default="페트,캔,유리,종이컵", help="precision/recall을 보는 클래스")
    ap.add_argument("--target", type=float, default=0.90, help="운영점 기준: 4클래스 최소 precision")
    ap.add_argument("--extra-targets", default="0.95", help="비교용 운영점 (쉼표로 구분)")
    ap.add_argument("--highlight", default="유리:페트", help="혼동행렬 강조 칸 '실제:예측' (쉼표로 여러 개)")
    ap.add_argument("--grid-step", type=float, default=0.005)
    a = ap.parse_args()
    a.classes = [c.strip() for c in a.classes.split(",")]
    a.recycle = [c.strip() for c in a.recycle.split(",")]
    a.extra_targets = [float(x) for x in a.extra_targets.split(",") if x.strip()]
    a.highlight = [tuple(x.split(":")) for x in a.highlight.split(",") if ":" in x]
    if set(a.recycle) - set(a.classes):
        sys.exit("[오류] --recycle 클래스가 --classes에 없음")

    setup_font()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt, preds = load_inputs(a.gt, a.preds, a.classes)
    results = [evaluate_model(name, gt, pm, a, out) for name, pm in preds.groupby("model", sort=False)]
    write_report(results, a, out)
    for R in results:
        print(f"{R['model']:>12}  t={R['t']:.3f}  확인필요={pct(R['rates']['review'])}  "
              f"최소P={pct(R['minP'])}  F0.5={pct(R['macro_f05'])}"
              + ("" if R["met_on_val"] else "  ⚠ val 기준 미달"))
    print(f"→ {out / 'report.md'}")


if __name__ == "__main__":
    main()
