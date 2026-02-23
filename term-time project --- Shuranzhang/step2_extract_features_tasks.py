from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
LABEL_XLSX = ROOT / "dataset_label.xlsx"
MODEL_PATH = ROOT / "face_landmarker.task"

OUT_FEATURES = ROOT / "features.csv"
OUT_MERGED = ROOT / "dataset_with_features.csv"

# ---- landmark indices ----
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
UPPER_LIP, LOWER_LIP = 13, 14
LEFT_BROW_MID, RIGHT_BROW_MID = 70, 300
LEFT_EYE_UP, RIGHT_EYE_UP = 159, 386


def find_image_path(emotion_value: str, image_id: str) -> Path | None:
    emotion_to_folder = {
        "anger": "Anger",
        "fear": "Fear",
        "happy": "Happy",
        "neutral": "Nuetual",  # your folder spelling
        "sad": "Sad",
        "shocked": "Shocked",
    }
    folder = emotion_to_folder.get(str(emotion_value).strip().lower(), str(emotion_value).strip())
    d = DATASET_DIR / folder
    if not d.exists():
        return None

    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = d / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def lm_xy(lms, idx, w, h):
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def compute_features(landmarks, w, h):
    p_le = lm_xy(landmarks, LEFT_EYE_OUTER, w, h)
    p_re = lm_xy(landmarks, RIGHT_EYE_OUTER, w, h)
    scale = float(np.linalg.norm(p_re - p_le) + 1e-6)

    le_top = lm_xy(landmarks, LEFT_EYE_TOP, w, h)
    le_bot = lm_xy(landmarks, LEFT_EYE_BOTTOM, w, h)
    re_top = lm_xy(landmarks, RIGHT_EYE_TOP, w, h)
    re_bot = lm_xy(landmarks, RIGHT_EYE_BOTTOM, w, h)
    eye_open = (np.linalg.norm(le_top - le_bot) + np.linalg.norm(re_top - re_bot)) / 2.0 / scale

    ml = lm_xy(landmarks, MOUTH_LEFT, w, h)
    mr = lm_xy(landmarks, MOUTH_RIGHT, w, h)
    up = lm_xy(landmarks, UPPER_LIP, w, h)
    low = lm_xy(landmarks, LOWER_LIP, w, h)
    mouth_open = float(np.linalg.norm(up - low) / scale)
    mouth_width = float(np.linalg.norm(ml - mr) / scale)

    mouth_center = (ml + mr) / 2.0
    smile = float(((mouth_center[1] - ml[1]) + (mouth_center[1] - mr[1])) / 2.0 / scale)

    lb = lm_xy(landmarks, LEFT_BROW_MID, w, h)
    rb = lm_xy(landmarks, RIGHT_BROW_MID, w, h)
    le_up = lm_xy(landmarks, LEFT_EYE_UP, w, h)
    re_up = lm_xy(landmarks, RIGHT_EYE_UP, w, h)
    brow_raise = float((((le_up[1] - lb[1]) + (re_up[1] - rb[1])) / 2.0) / scale)

    activity = float(0.4 * eye_open + 0.4 * mouth_open + 0.2 * abs(brow_raise))

    return {
        "eye_open": float(eye_open),
        "mouth_open": mouth_open,
        "mouth_width": mouth_width,
        "smile": smile,
        "brow_raise": brow_raise,
        "activity": activity,
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}\n请把 face_landmarker.task 放在 Reading Week 根目录。")

    labels = pd.read_excel(LABEL_XLSX)
    labels.columns = [c.strip() for c in labels.columns]

    # build image_path
    paths, miss = [], 0
    for _, r in labels.iterrows():
        p = find_image_path(r["emotion"], str(r["image_id"]).strip())
        if p is None:
            paths.append("")
            miss += 1
        else:
            paths.append(str(p))
    labels["image_path"] = paths
    print(f"Images matched: {len(labels)-miss}/{len(labels)}")

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    rows = []
    for i, r in labels.iterrows():
        image_id = r["image_id"]
        path = r["image_path"]

        if not path:
            rows.append({"image_id": image_id, "has_face": 0})
            continue

        bgr = cv2.imread(path)
        if bgr is None:
            rows.append({"image_id": image_id, "has_face": 0})
            continue

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            rows.append({"image_id": image_id, "has_face": 0})
        else:
            feats = compute_features(result.face_landmarks[0], w, h)
            rows.append({"image_id": image_id, "has_face": 1, **feats})

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(labels)}")

    feats_df = pd.DataFrame(rows)
    feats_df.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_FEATURES)

    merged = labels.merge(feats_df, on="image_id", how="left")
    merged.to_csv(OUT_MERGED, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_MERGED)
    print("Face detected:", int(merged["has_face"].fillna(0).sum()), "/", len(merged))


if __name__ == "__main__":
    main()