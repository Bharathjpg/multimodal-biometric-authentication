import cv2
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import time

# ── Settings ──────────────────────────────────────────────────────────────────
ATT_PATH = "datasets/att_faces"

# ── Load AT&T dataset ─────────────────────────────────────────────────────────
def load_att():
    images, labels = [], []
    for sid in range(1, 41):
        folder = os.path.join(ATT_PATH, f"s{sid}")
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".pgm"):
                img = cv2.imread(os.path.join(folder, fname),
                                 cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (92, 112))
                    img = cv2.equalizeHist(img)
                    images.append(img)
                    labels.append(sid - 1)
    return images, labels

# ── Run 5-fold cross validation ───────────────────────────────────────────────
def run_experiment(images, labels, make_recogniser, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, lats = [], [], []

    print(f"\nRunning 5-fold CV for: {name}")

    for fold, (tr_idx, te_idx) in enumerate(skf.split(images, labels)):
        X_tr = [images[i] for i in tr_idx]
        y_tr = [labels[i] for i in tr_idx]

        rec = make_recogniser()
        rec.train(X_tr, np.array(y_tr, dtype=np.int32))

        correct    = 0
        fold_lats  = []
        TP = FP = FN = 0

        for i in te_idx:
            t0         = time.perf_counter()
            pred, conf = rec.predict(images[i])
            fold_lats.append(time.perf_counter() - t0)

            if pred == labels[i]:
                correct += 1
                TP += 1
            else:
                FP += 1

        acc  = correct / len(te_idx)
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec2 = TP / (TP + FN + correct) if (TP + FN + correct) > 0 else 0
        f1   = 2 * prec * acc / (prec + acc) if (prec + acc) > 0 else 0

        accs.append(acc)
        f1s.append(f1)
        lats.append(np.mean(fold_lats))

        print(f"  Fold {fold+1}: acc={acc*100:.1f}%")

    print(f"\n  {name} FINAL RESULTS:")
    print(f"  Accuracy : {np.mean(accs)*100:.1f}%  "
          f"(std {np.std(accs)*100:.1f}%)")
    print(f"  F1 Score : {np.mean(f1s):.3f}")
    print(f"  Latency  : {np.mean(lats)*1000:.1f} ms")

    return {
        "acc" : round(np.mean(accs)*100, 1),
        "std" : round(np.std(accs)*100,  1),
        "f1"  : round(np.mean(f1s),      3),
        "lat" : round(np.mean(lats)*1000,1),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading AT&T dataset...")
    images, labels = load_att()
    print(f"Loaded {len(images)} images from {len(set(labels))} subjects.")

    results = {}

    # Eigenfaces
    results["Eigenfaces"] = run_experiment(
        images, labels,
        lambda: cv2.face.EigenFaceRecognizer_create(num_components=150),
        name="Eigenfaces"
    )

    # Fisherfaces
    results["Fisherfaces"] = run_experiment(
        images, labels,
        lambda: cv2.face.FisherFaceRecognizer_create(num_components=0),
        name="Fisherfaces"
    )

    # LBPH
    results["LBPH"] = run_experiment(
        images, labels,
        lambda: cv2.face.LBPHFaceRecognizer_create(
            radius=3, neighbors=8, grid_x=4, grid_y=4),
        name="LBPH"
    )

    # Summary table
    print("\n" + "="*55)
    print("SUMMARY TABLE — AT&T/ORL DATASET (40 subjects)")
    print("="*55)
    print(f"{'Algorithm':<14} {'Acc%':>6} {'Std':>5} "
          f"{'F1':>6} {'Lat(ms)':>8}")
    print("-"*55)
    for name, r in results.items():
        print(f"{name:<14} {r['acc']:>6} {r['std']:>5} "
              f"{r['f1']:>6} {r['lat']:>8}")
    print("="*55)