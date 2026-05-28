import cv2
import os
import numpy as np

# ── Settings ──────────────────────────────────────────────────────────────────
YALE_PATH = "datasets/yale_b"

TRAIN_CONDITIONS = ["normal", "happy", "sad", "sleepy",
                    "surprised", "wink", "glasses", "noglasses"]

TEST_CONDITIONS  = ["centerlight", "leftlight", "rightlight"]

# ── Load Yale dataset ─────────────────────────────────────────────────────────
def load_image(subject, condition):
    # Files have no extension — try direct path first, then with extensions
    for ext in ["", ".pgm", ".jpg", ".png", ".gif"]:
        path = os.path.join(YALE_PATH, f"{subject}.{condition}{ext}")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (92, 112))
                img = cv2.equalizeHist(img)
                return img
    return None

def load_yale():
    train_imgs, train_lbls = [], []
    test_imgs,  test_lbls  = [], []

    for sid in range(1, 16):
        subject = f"subject{sid:02d}"
        label   = sid - 1

        for condition in TRAIN_CONDITIONS:
            img = load_image(subject, condition)
            if img is not None:
                train_imgs.append(img)
                train_lbls.append(label)

        for condition in TEST_CONDITIONS:
            img = load_image(subject, condition)
            if img is not None:
                test_imgs.append(img)
                test_lbls.append(label)

    return train_imgs, train_lbls, test_imgs, test_lbls

# ── Run experiment ────────────────────────────────────────────────────────────
def run_experiment(train_imgs, train_lbls, test_imgs, test_lbls,
                   make_recogniser, name):

    rec = make_recogniser()
    rec.train(train_imgs, np.array(train_lbls, dtype=np.int32))

    correct = 0
    condition_results = {c: {"correct": 0, "total": 0}
                         for c in TEST_CONDITIONS}

    for i, (img, true_label) in enumerate(zip(test_imgs, test_lbls)):
        pred, conf = rec.predict(img)
        condition  = TEST_CONDITIONS[i % len(TEST_CONDITIONS)]

        condition_results[condition]["total"] += 1
        if pred == true_label:
            correct += 1
            condition_results[condition]["correct"] += 1

    overall_acc = correct / len(test_imgs) * 100

    print(f"\n  {name} RESULTS:")
    print(f"  Overall accuracy : {overall_acc:.1f}%")
    print(f"  Per lighting condition:")
    for condition, r in condition_results.items():
        if r["total"] > 0:
            acc = r["correct"] / r["total"] * 100
            print(f"    {condition:<14}: {acc:.1f}%  "
                  f"({r['correct']}/{r['total']})")

    return overall_acc, condition_results

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading Yale dataset...")
    train_imgs, train_lbls, test_imgs, test_lbls = load_yale()
    print(f"Train: {len(train_imgs)} images  |  "
          f"Test: {len(test_imgs)} images  |  "
          f"Subjects: 15")

    if len(train_imgs) == 0:
        print("ERROR: No images found.")
        exit()

    print("\nTraining on: normal, happy, sad, sleepy, "
          "surprised, wink, glasses, noglasses")
    print("Testing on : centerlight, leftlight, rightlight")

    all_results = {}

    acc, cond = run_experiment(
        train_imgs, train_lbls, test_imgs, test_lbls,
        lambda: cv2.face.EigenFaceRecognizer_create(num_components=14),
        name="Eigenfaces"
    )
    all_results["Eigenfaces"] = (acc, cond)

    acc, cond = run_experiment(
        train_imgs, train_lbls, test_imgs, test_lbls,
        lambda: cv2.face.FisherFaceRecognizer_create(num_components=0),
        name="Fisherfaces"
    )
    all_results["Fisherfaces"] = (acc, cond)

    acc, cond = run_experiment(
        train_imgs, train_lbls, test_imgs, test_lbls,
        lambda: cv2.face.LBPHFaceRecognizer_create(
            radius=3, neighbors=8, grid_x=4, grid_y=4),
        name="LBPH"
    )
    all_results["LBPH"] = (acc, cond)

    # Summary table
    print("\n" + "="*58)
    print("SUMMARY — YALE ILLUMINATION ROBUSTNESS TEST")
    print("="*58)
    print(f"{'Algorithm':<14} {'Center':>8} {'Left':>8} "
          f"{'Right':>8} {'Overall':>8}")
    print("-"*58)
    for name, (overall, cond) in all_results.items():
        c = cond["centerlight"]["correct"]/max(cond["centerlight"]["total"],1)*100
        l = cond["leftlight"]["correct"]/max(cond["leftlight"]["total"],1)*100
        r = cond["rightlight"]["correct"]/max(cond["rightlight"]["total"],1)*100
        print(f"{name:<14} {c:>7.1f}% {l:>7.1f}% "
              f"{r:>7.1f}% {overall:>7.1f}%")
    print("="*58)
    print("\nLower accuracy under leftlight/rightlight = less robust")
    print("LBPH should show smallest drop across conditions")