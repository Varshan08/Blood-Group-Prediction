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

    if cnn is None and os.path.exists("cnn_model_fixed.h5"):
        cnn = load_model("cnn_model_fixed.h5", compile=False)

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
        print("STEP 1: entering predict")

        file = request.files.get("file")
        print("STEP 2: file =", file)

        if file is None or file.filename == "":
            return "No file uploaded"

        os.makedirs("static", exist_ok=True)
        path = os.path.join("static", "test.jpg")
        file.save(path)

        print("STEP 3: file saved at", path)

        # FORCE CHECK
        if not os.path.exists(path):
            return "File not saved properly"

        from PIL import Image
        img = Image.open(path).convert("RGB")
        img = np.array(img)

        print("STEP 4: image loaded")

        img_color = cv2.resize(img, (100, 100))
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        print("STEP 5: image processed")

        load_models()

        print("STEP 6: models loaded")

        if svm is None or rf is None or cnn is None:
            return "Model not loaded"

        flat = img_gray.flatten().reshape(1, -1)

        svm_pred = svm.predict(flat)[0]
        rf_pred = rf.predict(flat)[0]

        cnn_img = img_color.reshape(1, 100, 100, 3) / 255.0
        probs = cnn.predict(cnn_img, verbose=0)[0]

        cnn_pred = labels[np.argmax(probs)]
        confidence = round(float(np.max(probs)) * 100, 2)

        print("STEP 7: prediction done")

        return f"SVM: {svm_pred}, RF: {rf_pred}, CNN: {cnn_pred}, Confidence: {confidence}%"

    except Exception as e:
        import traceback
        print("ERROR OCCURRED:")
        print(traceback.format_exc())
        return f"<pre>{traceback.format_exc()}</pre>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)