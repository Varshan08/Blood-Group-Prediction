import os
import cv2
import numpy as np
import pickle
import gdown
import zipfile
import sqlite3
import tensorflow as tf
from flask import Flask, render_template, request, redirect, session

app = Flask(__name__)
app.secret_key = "secret123"

# -------- DOWNLOAD MODELS IF NOT PRESENT --------
if not os.path.exists("cnn_saved_model"):
    print("Downloading CNN model...")
    gdown.download(
        "https://drive.google.com/uc?id=1lSTNLC8KNF47YFegzIKKlcGIa8PyZQe4",
        "cnn_saved_model.zip",
        quiet=False
    )
    print("Extracting CNN model...")
    with zipfile.ZipFile("cnn_saved_model.zip", "r") as z:
        z.extractall(".")
    os.remove("cnn_saved_model.zip")

if not os.path.exists("rf_model.pkl"):
    print("Downloading RF model...")
    gdown.download(
        "https://drive.google.com/uc?id=11YlCO1H3_egYLviJ1WGBbbBy1Uv5qaqT",
        "rf_model.pkl",
        quiet=False
    )

# -------- LOAD MODELS --------
rf = pickle.load(open("rf_model.pkl", "rb"))
cnn = tf.keras.layers.TFSMLayer("cnn_saved_model", call_endpoint="serving_default")

# -------- WARM UP CNN --------
print("Warming up CNN model...")
dummy = tf.constant(np.zeros((1, 100, 100, 3), dtype="float32"))
_ = cnn(dummy, training=False)
print("CNN ready!")

labels = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# -------- LOGIN PAGE --------
@app.route("/login")
def login_page():
    return render_template("login.html")

# -------- LOGIN --------
@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cur.fetchone()

    print("LOGIN INPUT:", username, password)
    print("DB RESULT:", user)
    conn.close()

    if user:
        session["user"] = username
        return redirect("/")
    else:
        return "Invalid login ❌"

@app.route("/")
def home():
    if not session.get("user"):
        return redirect("/login")
    return render_template("index.html")

# -------- SIGNUP --------
@app.route("/signup", methods=["POST"])
def signup():
    username = request.form["username"]
    password = request.form["password"]

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO users (username,password) VALUES (?,?)", (username, password))
    conn.commit()
    conn.close()
    return redirect("/login")

# -------- LOGOUT --------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

# -------- PREDICT --------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        os.makedirs("static", exist_ok=True)
        path = os.path.join("static", "test.jpg")
        file.save(path)

        # -------- IMAGE PROCESSING --------
        img_color = cv2.imread(path)
        img_gray = cv2.imread(path, 0)

        img_color = cv2.resize(img_color, (100, 100))
        img_gray = cv2.resize(img_gray, (100, 100))

        # -------- RF --------
        flat = img_gray.flatten().reshape(1, -1)
        rf_pred = rf.predict(flat)[0]

        # -------- CNN --------
        cnn_img = tf.constant(
            img_color.reshape(1, 100, 100, 3).astype("float32") / 255.0
        )
        output = cnn(cnn_img, training=False)
        probs = list(output.values())[0].numpy()[0]
        cnn_pred = labels[np.argmax(probs)]
        confidence = round(float(max(probs)) * 100, 2)

        return render_template(
            "result.html",
            rf=rf_pred,
            cnn=cnn_pred,
            confidence=confidence,
            img_path=path
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
