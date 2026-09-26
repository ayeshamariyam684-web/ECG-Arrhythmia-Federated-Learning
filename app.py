import warnings
warnings.filterwarnings('ignore')

import os
import torch
import timm
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms


# INIT

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL

model = timm.create_model('convnext_base', pretrained=False, num_classes=5)
model.load_state_dict(torch.load("models/fl_final_model.pth", map_location=device))
model.to(device)
model.eval()


# LOAD ECG CSV

ecg_df = pd.read_csv("mitbih_train.csv", header=None)


# TRANSFORM

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# CLASS INFO

CLASS_INFO = {
    1: {
        "name": "Normal Beat",
        "desc": "Normal sinus rhythm",
        "risk": "Low",
        "recommendation": "Maintain healthy lifestyle, regular checkups",
        "tips": "Exercise regularly, reduce stress, balanced diet"
    },
    3: {
        "name": "Supraventricular Ectopic Beat",
        "desc": "Premature atrial contraction",
        "risk": "Medium",
        "recommendation": "Consult cardiologist if frequent",
        "tips": "Avoid caffeine, monitor heart rate"
    },
    4: {
        "name": "Ventricular Ectopic Beat",
        "desc": "Premature ventricular contraction",
        "risk": "High",
        "recommendation": "Medical evaluation recommended",
        "tips": "Reduce stress, avoid stimulants"
    },
    0: {
        "name": "Fusion Beat",
        "desc": "Combination of normal and abnormal beat",
        "risk": "Medium",
        "recommendation": "Clinical observation needed",
        "tips": "Regular ECG monitoring"
    },
    2: {
        "name": "Unknown Beat",
        "desc": "Unclassified heartbeat",
        "risk": "High",
        "recommendation": "Further diagnostic tests required",
        "tips": "Consult specialist immediately"
    }
}

# ECG SIGNAL FETCH

def get_ecg_signals(class_id, num_samples=5):
    data = ecg_df[ecg_df.iloc[:, -1] == class_id]
    samples = data.sample(num_samples)

    signals = []
    for _, row in samples.iterrows():
        signals.append(row[:-1].values.astype(float).tolist())

    return signals


# QRS DETECTION (simple peak)

def detect_qrs(signal):
    signal = np.array(signal)
    peaks = []

    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.5:
            peaks.append(i)

    return peaks


# BPM CALCULATION

def compute_bpm(peaks, fs=125):
    if len(peaks) < 2:
        return 0

    intervals = np.diff(peaks) / fs
    avg_rr = np.mean(intervals)

    bpm = 60 / avg_rr
    return int(bpm)


# ROUTE

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            img_path = filepath

            img = Image.open(filepath).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

            info = CLASS_INFO[pred]

            signals = get_ecg_signals(pred)

            processed_signals = []
            for sig in signals:
                peaks = detect_qrs(sig)
                bpm = compute_bpm(peaks)

                processed_signals.append({
                    "signal": sig,
                    "peaks": peaks,
                    "bpm": bpm
                })

            result = {
                    "class": info["name"],
                    "desc": info["desc"],
                    "risk": info["risk"],
                    "recommendation": info["recommendation"],
                    "tips": info["tips"],
                    "confidence": round(confidence * 100, 2),
                    "signals": processed_signals
                }

    return render_template("index.html", result=result, img_path=img_path)


if __name__ == "__main__":
    app.run()