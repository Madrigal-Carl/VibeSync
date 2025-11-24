# 🎧 VibeSync — Mood Detection from Audio Features

A simple, lightweight project that predicts the mood of a song using an AI model built with PyTorch and served through FastAPI.  
Frontend uses TailwindCSS + vanilla HTML/JS.

---

## 🚀 Features

- FastAPI backend for prediction
- PyTorch model (`.pth` file)
- Data preprocessing + training scripts
- Simple HTML + Tailwind frontend

---

## 📁 Project Structure

```

VibeSync/
│
├── backend/
│   ├── models/
│   │   └── mood_nn_model.pth
│   ├── scripts/
│   ├── src/
│   │   ├── main.py
│   │   ├── predict_mood.py
│   │   ├── train_model.py
│   │   └── data_preprocessing.py
│   ├── requirements.txt
│
└── frontend/
├── public/
│   └── index.html
├── src/
│   ├── input.css
│   ├── output.css
│   ├── images/
│   └── script.js
├── package.json

```

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

All dependencies are listed inside **`backend/requirements.txt`**.

---

## 🛠 0. Install Python and pip

Make sure Python 3.10+ is installed:

- **Windows:** [Download Python](https://www.python.org/downloads/windows/) and select "Add Python to PATH" during installation.

Check installation:

```bash
python --version
pip --version
```

---

## 🛠 1. Set Up Virtual Environment (Recommended)

Inside the `backend/` folder:

```bash
python -m venv venv       # Create virtual environment

# Activate environment:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

---

## 📦 2. Install Backend Dependencies

Make sure your virtual environment is active, then run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🤖 3. Run the FastAPI Backend

Inside `backend/`:

```bash
uvicorn src.main:app --reload
```

The server will start at:
`http://127.0.0.1:8000`

You can access the API docs at:
`http://127.0.0.1:8000/docs`

---

# 🎨 4. Install Frontend (Tailwind)

Go to the frontend folder:

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

Open `frontend/public/index.html` in a browser to view the UI.

---

## 📝 License

Free to use for personal or academic projects.

---

## ❤️ Author

VibeSync by Carl Madrigal & Erickson Dave Geroleo
Made with AI, music, and caffeine.

```

This is a complete, ready-to-use `README.md` file.

If you want, I can also **add a full “How to Use the API” section with example requests** so it’s fully functional for someone cloning the project.

Do you want me to add that?
```

```

```
