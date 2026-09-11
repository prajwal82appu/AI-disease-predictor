# 🩺 AI Disease Predictor

<p align="center">
  <img src="https://img.shields.io/badge/AI-Disease%20Prediction-blueviolet?style=for-the-badge&logo=python" alt="AI Disease Prediction">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Machine%20Learning-Powered-orange?style=for-the-badge" alt="Machine Learning">
</p>

<p align="center">
  <b>An AI-powered web application designed to predict possible diseases from user-provided symptoms.</b>
</p>

<p align="center">
  Built with ❤️ using Python, Machine Learning, and Flask.
</p>

---

## 📌 About The Project

**AI Disease Predictor** is a machine-learning-based healthcare project that provides a simple web interface for disease prediction.

The system takes relevant health/symptom information from the user, processes the input through a trained machine-learning model, and presents a predicted result through a web interface.

The project was developed as an academic **Major Project** to demonstrate the practical application of:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 🐍 Python
* 🌐 Web Development
* 📊 Data-driven prediction

> ⚠️ **Disclaimer:** This project is intended for educational and demonstration purposes only. It should not be considered a replacement for professional medical advice, diagnosis, or treatment.

---

## ✨ Key Features

### 🧠 AI-Based Prediction

Uses a machine-learning model to analyze user-provided information and generate a prediction.

### 🌐 Web-Based Interface

Provides a user-friendly web interface through Flask templates.

### 🩺 Symptom-Based Analysis

Users can provide relevant symptoms/information to obtain a predicted disease.

### 📋 Patient Data Handling

The project includes patient-related data storage functionality.

### 🗂️ Manual Entries

Supports manually maintained disease-related information through JSON data.

### 🎨 Simple & Interactive UI

The application includes HTML templates and static resources for presenting the system through a browser.

### 📚 Academic Project Documentation

A complete **Major Project Report** is included in the repository.

---

## 🛠️ Technology Stack

| Technology              | Purpose                   |
| ----------------------- | ------------------------- |
| 🐍 **Python**           | Core programming language |
| 🤖 **Machine Learning** | Disease prediction        |
| 🌐 **Flask**            | Web application framework |
| HTML                    | Frontend structure        |
| CSS                     | Interface styling         |
| JavaScript              | Client-side interaction   |
| JSON                    | Data storage              |
| Git & GitHub            | Version control           |

---

## 🏗️ Project Architecture

```text
                ┌──────────────────────┐
                │       User           │
                │  Symptoms / Inputs   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Flask Web App     │
                │       app.py         │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Prediction Model   │
                │       model.py       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Prediction Result   │
                │  & Health Information│
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │      Web UI          │
                │   Result Display     │
                └──────────────────────┘
```

---

## 📂 Project Structure

```text
AI-disease-predictor/
│
├── 📁 .snapshots/
│
├── 📁 __pycache__/
│
├── 📁 static/
│   └── 📁 samples/
│
├── 📁 templates/
│
├── 📄 app.py
│
├── 📄 model.py
│
├── 📄 manual_entries.json
│
├── 📄 patients.json
│
├── 📄 AI Disease Prediction System - Major Project Report.docx
│
└── 📄 README.md
```

---

## 🚀 Getting Started

Follow these steps to run the project locally.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/prajwal82appu/AI-disease-predictor.git
```

### 2️⃣ Open the Project

```bash
cd AI-disease-predictor
```

### 3️⃣ Create a Virtual Environment

It is recommended to use a virtual environment.

```bash
python -m venv venv
```

### 4️⃣ Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 5️⃣ Install Dependencies

If the project contains a `requirements.txt` file, install the dependencies using:

```bash
pip install -r requirements.txt
```

If dependencies are not listed yet, install the required packages used by the project.

### 6️⃣ Run the Application

```bash
python app.py
```

The Flask application should start locally.

Open the local address shown in your terminal in a web browser.

---

## 🔄 How It Works

The general workflow of the application is:

```text
User Input
    ↓
Input Processing
    ↓
Machine Learning Model
    ↓
Disease Prediction
    ↓
Result Display
```

### Step 1 — User Input

The user provides the required symptoms or health-related information through the web interface.

### Step 2 — Data Processing

The application processes the submitted information into a format that can be handled by the prediction model.

### Step 3 — Machine Learning

The processed input is passed to the model implemented in `model.py`.

### Step 4 — Prediction

The model generates a predicted result based on the information provided.

### Step 5 — Result

The prediction is returned to the Flask application and displayed through the web interface.

---

## 🧩 Main Components

### `app.py`

The main Flask application responsible for:

* Starting the web server
* Handling web routes
* Receiving user input
* Connecting the interface with the prediction logic
* Returning results to the frontend

### `model.py`

Contains the machine-learning/prediction logic used by the application.

### `templates/`

Contains the HTML templates responsible for the web interface.

### `static/`

Contains static resources used by the application, including sample resources.

### `manual_entries.json`

Stores manually maintained application/disease-related information.

### `patients.json`

Stores patient-related application data.

### `AI Disease Prediction System - Major Project Report.docx`

Contains the project's academic documentation and major-project report.

---

## 🎯 Objectives

The main objectives of this project are:

* To demonstrate the use of Machine Learning in healthcare applications.
* To develop a simple web-based disease prediction system.
* To process user-provided symptom information.
* To integrate a machine-learning model with a web application.
* To provide an easy-to-use interface for educational experimentation.
* To demonstrate how AI can assist with preliminary health-risk awareness.

---

## 🌟 Why This Project?

Healthcare is an important area where Artificial Intelligence and Machine Learning can support data-driven applications.

This project demonstrates how a machine-learning model can be integrated into a web application to process health-related inputs and generate predictions.

It also provides a practical example of combining:

**Machine Learning + Python + Flask + Web Development**

into a single application.

---

## 🔮 Future Enhancements

Some possible improvements for future versions include:

* 📈 Improved model accuracy through better datasets
* 🧠 Support for multiple machine-learning algorithms
* 📊 Prediction confidence visualization
* 👤 User authentication and secure profiles
* 🗃️ Database integration instead of JSON storage
* 📱 Fully responsive mobile interface
* 📑 Downloadable prediction reports
* 📊 Patient history and analytics dashboard
* 🔐 Improved privacy and security
* ☁️ Cloud deployment
* 🧪 Automated model evaluation and testing

---

## ⚠️ Important Medical Disclaimer

This application is a **student/educational machine-learning project**.

The predictions generated by this system are **not medical diagnoses** and should not be used to make healthcare decisions.

For any health concern, users should consult a qualified healthcare professional.

---

## 🤝 Contributing

Contributions and suggestions are welcome.

### Steps to contribute

```bash
# Fork the repository

# Clone your fork
git clone https://github.com/prajwal82appu/AI-disease-predictor.git

# Create a new branch
git checkout -b feature/your-feature

# Make your changes

# Commit your changes
git add .
git commit -m "Add new feature"

# Push the branch
git push origin feature/your-feature
```

Then open a **Pull Request** on GitHub.

---

## 📜 License

Please refer to the repository for the applicable licensing information.

If you intend to publish this project as an open-source project, adding a dedicated `LICENSE` file is recommended.

---

## 👨‍💻 Author

### **Prajwal**

🔗 GitHub:
https://github.com/prajwal82appu

🔗 Project Repository:
https://github.com/prajwal82appu/AI-disease-predictor

---

## ⭐ Support

If you find this project useful or interesting:

⭐ Star the repository
🍴 Fork the project
🐛 Report issues
💡 Suggest improvements
🤝 Contribute to the project

---

<p align="center">
  <b>🩺 AI Disease Predictor</b>
  <br>
  <i>Exploring the power of Artificial Intelligence in healthcare.</i>
</p>

<p align="center">
  Made with ❤️ using Python & Machine Learning
</p>
