"""Separability of teacher / student latents with vs without latent dynamics prediction.

Input: latent_collect.py outputs (all encoders evaluated on identical observations,
behaviour policies Ours and LPGP pooled). Pre-specified primary outcome: separability of
the TEACHER latent w.r.t. 8 terrain families, measured by the cosine silhouette score and
by linear-probe balanced accuracy, comparing Ours (with latent prediction) and LPGP
(without). Secondary: 11 terrain classes, locomotion mode (rolling vs wheel lifted),
student latents, Davies-Bouldin / Fisher ratio / kNN accuracy, and linear predictability
of future latents given actions ("dynamic consistency").

Samples: alive steps t = 0.5 s .. 10 s every 0.1 s; for non-flat terrains only steps with
local relief >= 2 cm under the base (the robot is on terrain features, not on the flat
spawn platform or patch borders).

Run from the repo root (ephemeral env, the project venv has no scikit-learn):
  uv run --no-project --python 3.13 --with scikit-learn --with numpy --with scipy \
      python scripts/eval/lipm_diagnostics/latent_analysis.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("logs/lipm_eval/full")
DATA = OUT / "diagnostics" / "latent"
RESULTS = OUT / "results"
FAMILY = {
    "flat": "flat", "random_rough": "rough", "hf_pyramid_slope": "slope",
    "hf_pyramid_slope_inv": "slope", "pyramid_stair": "stairs", "pyramid_stair_inv": "stairs",
    "random_stairs": "stairs", "discrete_obstacles": "discrete obstacles",
    "random_spread": "boxes", "stepping_stones": "stepping stones", "tilted_grid": "tilted grid",
}
FAMILIES = ("flat", "rough", "slope", "stairs", "discrete obstacles", "boxes",
            "stepping stones", "tilted grid")
ENCODERS = ("Ours", "RGGP", "LPGP")
STEPS = np.arange(25, 500, 5)
RNG = np.random.default_rng(0)
N_BOOT = 30


def load():
    rows = {k: [] for k in ("terrain", "group", "t", "mode", "zt", "zs")}
    future = []  # (zt sequence, actions, valid) for predictability, per file
    for path in sorted(DATA.glob("*.npz")):
        d = np.load(path)
        terrain, behaviour = str(d["terrain"]), str(d["behaviour"])
        assert tuple(d["models"]) == ENCODERS
        valid, relief = d["valid"], d["relief"]
        keep = np.zeros_like(valid)
        keep[STEPS] = True
        keep &= valid
        if terrain != "flat":
            keep &= relief >= 0.02
        t_idx, env_idx = np.nonzero(keep)
        rows["terrain"].append(np.full(len(t_idx), terrain))
        rows["group"].append(np.array([f"{terrain}|{behaviour}|{e}" for e in env_idx]))
        rows["t"].append(t_idx)
        airborne = ~d["wheel_contact"].all(-1)
        rows["mode"].append(np.where(airborne[t_idx, env_idx], "lifted", "rolling"))
        rows["zt"].append(d["z_teacher"][t_idx, env_idx].astype(np.float32))
        rows["zs"].append(d["z_student"][t_idx, env_idx].astype(np.float32))
        future.append((terrain, behaviour, d["z_teacher"].astype(np.float32), d["actions"], valid))
    data = {k: np.concatenate(v) for k, v in rows.items()}
    data["family"] = np.array([FAMILY[t] for t in data["terrain"]])
    return data, future


def balanced_index(labels, groups, per_class, rng):
    """Up to per_class samples per label, drawn after resampling groups (bootstrap)."""
    index = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        member_groups = np.unique(groups[members])
        if rng is not None:
            chosen = rng.choice(member_groups, size=len(member_groups), replace=True)
            weights = {g: c for g, c in zip(*np.unique(chosen, return_counts=True))}
            p = np.array([weights.get(g, 0) for g in groups[members]], dtype=float)
            p /= p.sum()
            index.append(rng.choice(members, size=min(per_class, len(members)), replace=True, p=p))
        else:
            index.append(RNG.choice(members, size=min(per_class, len(members)), replace=False))
    return np.concatenate(index)


def fisher_ratio(z, labels):
    mean = z.mean(0)
    between = within = 0.0
    for label in np.unique(labels):
        zc = z[labels == label]
        between += len(zc) * np.sum((zc.mean(0) - mean) ** 2)
        within += np.sum((zc - zc.mean(0)) ** 2)
    return between / within


def cluster_scores(latents, labels, groups, per_class=600):
    """Bootstrap over trajectories; identical resamples for every encoder (paired)."""
    boot = {e: {"silhouette": [], "davies_bouldin": [], "fisher": []} for e in latents}
    rng = np.random.default_rng(1)
    for _ in range(N_BOOT):
        idx = balanced_index(labels, groups, per_class, rng)
        for e, z in latents.items():
            boot[e]["silhouette"].append(silhouette_score(z[idx], labels[idx], metric="cosine"))
            boot[e]["davies_bouldin"].append(davies_bouldin_score(z[idx], labels[idx]))
            boot[e]["fisher"].append(fisher_ratio(z[idx], labels[idx]))
    return {e: {k: np.array(v) for k, v in m.items()} for e, m in boot.items()}


def probe_scores(latents, labels, groups, per_class=4000):
    idx = balanced_index(labels, groups, per_class, None)
    folds = list(GroupKFold(n_splits=5).split(idx, labels[idx], groups[idx]))
    scores = {e: {"linear_probe": [], "knn": []} for e in latents}
    for e, z in latents.items():
        x, y = z[idx], labels[idx]
        for train, test in folds:
            linear = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
            linear.fit(x[train], y[train])
            scores[e]["linear_probe"].append(balanced_accuracy_score(y[test], linear.predict(x[test])))
            knn = KNeighborsClassifier(n_neighbors=15, metric="cosine").fit(x[train], y[train])
            scores[e]["knn"].append(balanced_accuracy_score(y[test], knn.predict(x[test])))
    return {e: {k: np.array(v) for k, v in m.items()} for e, m in scores.items()}


def summarise(values):
    return {"mean": float(np.mean(values)), "ci95": [float(np.percentile(values, 2.5)),
                                                     float(np.percentile(values, 97.5))],
            "values": [float(v) for v in values]}


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)),
                                                   float(np.percentile(d, 97.5))],
            "frac_positive": float(np.mean(d > 0))}


def predictability(future, which, horizons=(1, 5, 10)):
    """Ridge R^2 of z_{t+h} from [z_t, a_t .. a_{t+h-1}], grouped 5-fold CV."""
    out = {}
    for h in horizons:
        xs, ys, gs = [], [], []
        for terrain, behaviour, zt, actions, valid in future:
            t_idx = np.arange(25, 500 - h, 5)
            for e in range(zt.shape[1]):
                ok = t_idx[valid[t_idx + h, e]]
                if len(ok) == 0:
                    continue
                act = np.concatenate([actions[ok + k, e] for k in range(h)], axis=-1)
                xs.append(np.concatenate([zt[ok, e, which], act], axis=-1))
                ys.append(zt[ok + h, e, which])
                gs.append(np.full(len(ok), f"{terrain}|{behaviour}|{e}"))
        x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)
        sub = RNG.choice(len(x), size=min(len(x), 60000), replace=False)
        x, y, g = x[sub], y[sub], g[sub]
        r2 = []
        for train, test in GroupKFold(n_splits=5).split(x, y, g):
            model = Ridge(alpha=1.0).fit(x[train], y[train])
            residual = np.sum((y[test] - model.predict(x[test])) ** 2)
            total = np.sum((y[test] - y[test].mean(0)) ** 2)
            r2.append(1.0 - residual / total)
        out[h] = np.array(r2)
    return out


def main():
    data, future = load()
    report = {"n_samples": int(len(data["family"])),
              "samples_per_family": {f: int(np.sum(data["family"] == f)) for f in FAMILIES}}
    views = {
        "teacher_families": ("zt", data["family"]),
        "teacher_classes": ("zt", data["terrain"]),
        "teacher_mode": ("zt", data["mode"]),
        "student_families": ("zs", data["family"]),
    }
    for view, (key, labels) in views.items():
        latents = {e: data[key][:, j] for j, e in enumerate(ENCODERS)}
        clusters = cluster_scores(latents, labels, data["group"])
        probes = probe_scores(latents, labels, data["group"])
        entry = {e: {**{k: summarise(v) for k, v in clusters[e].items()},
                     **{k: summarise(v) for k, v in probes[e].items()}} for e in ENCODERS}
        entry["Ours_minus_LPGP"] = {
            **{k: paired(clusters["Ours"][k], clusters["LPGP"][k]) for k in clusters["Ours"]},
            **{k: paired(probes["Ours"][k], probes["LPGP"][k]) for k in probes["Ours"]},
        }
        report[view] = entry
        print(view, {e: {k: round(entry[e][k]["mean"], 4) for k in
                         ("silhouette", "linear_probe", "knn", "davies_bouldin", "fisher")}
                     for e in ENCODERS}, flush=True)
    report["teacher_predictability_r2"] = {
        e: {str(h): summarise(v) for h, v in predictability(future, j).items()}
        for j, e in enumerate(ENCODERS)
    }
    print("predictability", {e: {h: round(v["mean"], 4) for h, v in m.items()}
                             for e, m in report["teacher_predictability_r2"].items()}, flush=True)

    # t-SNE on identical balanced samples (teacher latents), for the figure.
    idx = balanced_index(data["family"], data["group"], 300, None)
    embeddings = {}
    for j, e in enumerate(ENCODERS):
        embeddings[e] = TSNE(n_components=2, perplexity=30, init="pca", metric="cosine",
                             random_state=0).fit_transform(data["zt"][idx, j])
    np.savez_compressed(RESULTS / "latent_tsne.npz", family=data["family"][idx],
                        terrain=data["terrain"][idx], **{f"tsne_{e}": v for e, v in embeddings.items()})
    (RESULTS / "latent_separability.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
