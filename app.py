import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import sqlite3
import logging
from flask import Flask, render_template, request, redirect, session
from tensorflow.keras.models import load_model
from joblib import load

# 🔇 disable logs
logging.getLogger('werkzeug').disabled = True
logging.disable(logging.CRITICAL)

app = Flask(__name__)
app.secret_key = "secret123"

svm = rf = cnn = None

def load_models():
    global svm, rf, cnn

    if svm is None and os.path.exists("svm_model.joblib"):
        svm = load("svm_model.joblib")

    if rf is None and os.path.exists("rf_model.joblib"):
        rf = load("rf_model.joblib")

    if cnn is None and os.path.exists("cnn_model.h5"):
        cnn = load_model("cnn_model.h5")

labels = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    try:
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cur.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect("/")
        return "Invalid login ❌"
    except:
        return "Error"

@app.route("/")
def home():
    if not session.get("user"):
        return redirect("/login")
    return render_template("index.html")

@app.route("/signup", methods=["POST"])
def signup():
    try:
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username,password) VALUES (?,?)", (username, password))
        conn.commit()
        conn.close()

        return redirect("/login")
    except:
        return "Error"

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_models()

        if svm is None or rf is None or cnn is None:
            return "Model not loaded"

        file = request.files.get("file")

        if file is None or file.filename == "":
            return "No file uploaded"
        print("FILES:", request.files)

        os.makedirs("static", exist_ok=True)
        path = os.path.join("static", "test.jpg")
        file.save(path)

        img_color = cv2.imread(path)
        img_gray = cv2.imread(path, 0)

        if img_color is None or img_gray is None:
            return "Image processing failed. Try another image."

        img_color = cv2.resize(img_color, (100, 100))
        img_gray = cv2.resize(img_gray, (100, 100))

        flat = img_gray.flatten().reshape(1, -1)

        svm_pred = svm.predict(flat)[0]
        rf_pred = rf.predict(flat)[0]

        cnn_img = img_color.reshape(1, 100, 100, 3) / 255.0
        probs = cnn.predict(cnn_img, verbose=0)[0]

        cnn_pred = labels[np.argmax(probs)]
        confidence = round(float(np.max(probs)) * 100, 2)

        return render_template(
            "result.html",
            svm=svm_pred,
            rf=rf_pred,
            cnn=cnn_pred,
            confidence=confidence,
            img_path=path
        )

    except Exception as e:
        import traceback
    return f"<pre>{traceback.format_exc()}</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)