# **VibeSync**

```md
# 🎧 VibeSync — Mood Detection from Audio Features

A simple, lightweight project that predicts the mood of a song using an AI model built with PyTorch and served through FastAPI.  
Frontend uses TailwindCSS + vanilla HTML/JS.

---

## 🚀 Features

- FastAPI backend for prediction
- PyTorch model (`.pth` file)
- Data preprocessing + training scripts
- Simple HTML + Tailwind frontend
- Works on Windows, macOS, and Linux

---

## 📁 Project Structure
```

VibeSync/
│
├── backend/
│ ├── models/
│ │ └── mood_nn_model.pth
│ ├── scripts/
│ ├── src/
│ │ ├── main.py
│ │ ├── predict_mood.py
│ │ ├── train_model.py
│ │ └── data_preprocessing.py
│ ├── requirements.txt
│
└── frontend/
├── public/
│ └── index.html
├── src/
│ ├── input.css
│ ├── output.css
│ ├── images/
│ └── script.js
├── package.json

````

---

## 🧰 Backend Requirements

The backend uses:

- fastapi
- uvicorn
- pydantic
- pandas
- numpy
- scikit-learn
- joblib
- torch
- torchvision
- imblearn

All of these are already listed inside **`backend/requirements.txt`**.

---

## 📦 1. Install Dependencies

Make sure your virtual environment is active, then:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🤖 2. Run the FastAPI Backend

Inside `backend/`:

```bash
uvicorn backend.main:app --reload
```

👉 Server will start
---

# 🎨 3. Install Frontend (Tailwind)

Go to frontend:

```bash
cd frontend
npm install
```

### Start Tailwind CLI (watch mode)

```bash
npm run build
```

This generates or updates:

```
src/output.css
```

Open `frontend/public/index.html` in browser to view UI.

---

## 📝 License

Free to use for personal or academic projects.

---

## ❤️ Author

VibeSync by Carl Madrigal & Erickson Dave Geroleo
Made with AI, music, and caffeine.

```

```
````
