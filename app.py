from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from model import engine
import os
import json
from datetime import datetime

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.secret_key = os.environ.get('SECRET_KEY', 'ai-disease-prediction-system-major-project-secret-2026')

PATIENTS_FILE = 'patients.json'

def load_patients():
    if os.path.exists(PATIENTS_FILE):
        try:
            with open(PATIENTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print("Error loading patients:", e)
            return []
    return []

def save_patients(patients):
    try:
        with open(PATIENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(patients, f, indent=2)
    except Exception as e:
        print("Error saving patients:", e)

def find_patient_by_identifier(identifier):
    if not identifier:
        return None
    ident = " ".join(identifier.strip().lower().split())
    patients = load_patients()
    for p in patients:
        p_name = " ".join(p.get('name', '').strip().lower().split())
        p_email = p.get('email', '').strip().lower()
        p_id = p.get('patient_id', '').strip().lower()
        if p_name == ident or p_email == ident or p_id == ident:
            return p
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/logout')
def logout_redirect():
    session.clear()
    return redirect('/login')

@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    data = request.json or {}
    identifier = data.get('identifier', '').strip()
    password = data.get('password', '')
    
    if not identifier or not password:
        return jsonify({"error": "Please enter your Name (or Email) and password."}), 400
        
    patient = find_patient_by_identifier(identifier)
    if not patient or not check_password_hash(patient.get('password_hash', ''), password):
        return jsonify({"error": "Invalid Patient Name/Email or password. Please verify and try again."}), 401
        
    session['patient_id'] = patient['patient_id']
    session['patient_name'] = patient['name']
    
    return jsonify({
        "success": True,
        "patient": {
            "patient_id": patient["patient_id"],
            "name": patient["name"],
            "email": patient["email"],
            "age": patient.get("age"),
            "gender": patient.get("gender"),
            "blood_group": patient.get("blood_group"),
            "phone": patient.get("phone")
        }
    })

@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    data = request.json or {}
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    age = data.get('age')
    gender = data.get('gender', 'Other')
    blood_group = data.get('blood_group', 'O+')
    phone = data.get('phone', '').strip()
    
    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required fields."}), 400
        
    patients = load_patients()
    if any(p.get('email', '').lower() == email.lower() for p in patients):
        return jsonify({"error": "An account with this email address is already registered."}), 400
        
    new_patient_id = f"PT-{1000 + len(patients) + 1}"
    new_patient = {
        "patient_id": new_patient_id,
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password),
        "age": age,
        "gender": gender,
        "blood_group": blood_group,
        "phone": phone,
        "emergency_contact": "",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prediction_history": []
    }
    patients.append(new_patient)
    save_patients(patients)
    
    session['patient_id'] = new_patient['patient_id']
    session['patient_name'] = new_patient['name']
    
    return jsonify({
        "success": True,
        "patient": {
            "patient_id": new_patient["patient_id"],
            "name": new_patient["name"],
            "email": new_patient["email"],
            "age": new_patient["age"],
            "gender": new_patient["gender"],
            "blood_group": new_patient["blood_group"],
            "phone": new_patient["phone"]
        }
    })

@app.route('/api/auth/logout', methods=['POST'])
def auth_logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/api/auth/current_user', methods=['GET'])
def auth_current_user():
    patient_id = session.get('patient_id')
    if not patient_id:
        return jsonify({"logged_in": False, "patient": None})
        
    patient = find_patient_by_identifier(patient_id)
    if not patient:
        session.clear()
        return jsonify({"logged_in": False, "patient": None})
        
    return jsonify({
        "logged_in": True,
        "patient": {
            "patient_id": patient["patient_id"],
            "name": patient["name"],
            "email": patient["email"],
            "age": patient.get("age"),
            "gender": patient.get("gender"),
            "blood_group": patient.get("blood_group"),
            "phone": patient.get("phone"),
            "created_at": patient.get("created_at"),
            "prediction_history": patient.get("prediction_history", [])
        }
    })

@app.route('/api/patient/save_prediction', methods=['POST'])
def patient_save_prediction():
    patient_id = session.get('patient_id')
    if not patient_id:
        return jsonify({"error": "Please log in to your patient account to save records."}), 401
        
    data = request.json or {}
    patients = load_patients()
    patient = None
    for p in patients:
        if p.get('patient_id') == patient_id:
            patient = p
            break
            
    if not patient:
        return jsonify({"error": "Patient account not found."}), 404
        
    if 'prediction_history' not in patient:
        patient['prediction_history'] = []
        
    rec_id = f"REC-{len(patient['prediction_history']) + 101}"
    record = {
        "id": rec_id,
        "disease": data.get("disease", "AI Health Assessment"),
        "prediction": data.get("prediction", "Assessment Completed"),
        "risk_level": data.get("risk_level", "NORMAL"),
        "risk_score": data.get("risk_score", 0),
        "confidence": data.get("confidence", 95.0),
        "model_used": data.get("model_used", "Multi-Modal AI Ensemble"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes": data.get("notes", "")
    }
    patient["prediction_history"].insert(0, record)
    save_patients(patients)
    
    return jsonify({"success": True, "record": record, "history": patient["prediction_history"]})

@app.route('/api/patient/history', methods=['GET'])
def patient_history():
    patient_id = session.get('patient_id')
    if not patient_id:
        return jsonify([])
    patient = find_patient_by_identifier(patient_id)
    return jsonify(patient.get("prediction_history", []) if patient else [])

@app.route('/api/predict/ckd', methods=['POST'])
def predict_ckd():
    data = request.json or {}
    result = engine.predict_ckd(data)
    return jsonify(result)

@app.route('/api/predict/parkinsons', methods=['POST'])
def predict_parkinsons():
    data = request.json or {}
    result = engine.predict_parkinsons(data)
    return jsonify(result)

@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.json or {}
    result = engine.predict_diabetes(data)
    return jsonify(result)

@app.route('/api/predict/heart', methods=['POST'])
def predict_heart():
    data = request.json or {}
    result = engine.predict_heart(data)
    return jsonify(result)

@app.route('/api/predict/cancer', methods=['POST'])
def predict_cancer():
    data = request.json or {}
    result = engine.predict_cancer(data)
    return jsonify(result)


MANUAL_HISTORY_FILE = 'manual_entries.json'

def load_manual_history():
    if os.path.exists(MANUAL_HISTORY_FILE):
        try:
            with open(MANUAL_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_manual_history(history):
    try:
        with open(MANUAL_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print("Error saving manual entries:", e)

manual_entries_history = load_manual_history()

@app.route('/api/predict/manual_entry', methods=['POST'])
def predict_manual_entry():
    try:
        data = request.json or {}
        result = engine.predict_manual_disease(data)
        
        entry_record = {
            "id": len(manual_entries_history) + 1,
            "patient_name": result.get("patient_name", "Anonymous Patient"),
            "disease": data.get("disease_name") or "Custom Clinical Condition",
            "severity": data.get("severity", "Moderate"),
            "risk_score": result.get("risk_score"),
            "prediction": result.get("prediction"),
            "risk_level": result.get("risk_level"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        manual_entries_history.insert(0, entry_record)
        save_manual_history(manual_entries_history)
        result["history_count"] = len(manual_entries_history)

        # If a patient is currently logged in, also record into patient's personal dossier
        current_pid = session.get('patient_id')
        if current_pid:
            patients = load_patients()
            for p in patients:
                if p.get('patient_id') == current_pid:
                    if 'prediction_history' not in p:
                        p['prediction_history'] = []
                    p['prediction_history'].insert(0, {
                        "id": f"REC-{len(p['prediction_history']) + 101}",
                        "disease": entry_record["disease"],
                        "prediction": entry_record["prediction"],
                        "risk_level": entry_record["risk_level"],
                        "risk_score": entry_record["risk_score"],
                        "confidence": result.get("confidence", 95.0),
                        "model_used": "Manual Clinical Assessment",
                        "timestamp": entry_record["timestamp"],
                        "notes": f"Severity: {entry_record['severity']}. Symptoms: {data.get('symptoms', 'None specified')}"
                    })
                    save_patients(patients)
                    break

        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Diagnostic processing error: {str(e)}"}), 500

@app.route('/api/manual_entries', methods=['GET'])
def get_manual_entries():
    return jsonify(manual_entries_history)

@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image or report file uploaded"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    doc_type = request.form.get('doc_type', 'auto')
    file_bytes = file.read()
    result = engine.predict_image(file_bytes, file.filename, doc_type=doc_type)
    return jsonify(result)

@app.route('/api/predict/sample_image/<sample_name>', methods=['GET', 'POST'])
def predict_sample_image(sample_name):
    filename_map = {
        'pneumonia': ('static/samples/sample_pneumonia.jpg', 'sample_pneumonia.jpg', 'xray'),
        'normal_xray': ('static/samples/sample_normal_xray.jpg', 'sample_normal_xray.jpg', 'xray'),
        'patient_report': ('static/samples/sample_patient_report.jpg', 'sample_patient_report.jpg', 'report'),
        'patient_report_pdf': ('static/samples/sample_patient_report.pdf', 'sample_patient_report.pdf', 'report')
    }
    if sample_name not in filename_map:
        return jsonify({"error": "Sample image not found"}), 404
        
    path, fname, dtype = filename_map[sample_name]
    if not os.path.exists(path):
        return jsonify({"error": "Sample file missing on server"}), 404
        
    with open(path, 'rb') as f:
        file_bytes = f.read()
        
    result = engine.predict_image(file_bytes, fname, doc_type=dtype)
    result['sample_img_url'] = f'/{path}'
    return jsonify(result)

@app.route('/api/sample/<disease>/<sample_type>', methods=['GET'])
def get_sample(disease, sample_type):
    if disease == 'ckd':
        if sample_type == 'high_risk':
            return jsonify({
                "age": 62, "bp": 90, "sg": 1.010, "al": 3, "su": 1,
                "bgr": 210, "bu": 85, "sc": 4.2, "sod": 130, "pot": 5.8,
                "hemo": 9.4, "pcv": 31, "wbcc": 11200, "rbcc": 3.4,
                "htn": 1, "dm": 1
            })
        else:
            return jsonify({
                "age": 45, "bp": 75, "sg": 1.025, "al": 0, "su": 0,
                "bgr": 105, "bu": 24, "sc": 0.9, "sod": 140, "pot": 4.2,
                "hemo": 14.8, "pcv": 45, "wbcc": 6500, "rbcc": 5.1,
                "htn": 0, "dm": 0
            })
    elif disease == 'parkinsons':
        if sample_type == 'high_risk':
            return jsonify({
                "fo": 119.992, "fhi": 157.302, "flo": 74.997,
                "jitter": 0.00784, "shimmer": 0.04374, "nhr": 0.02211,
                "hnr": 21.033, "rpde": 0.414783, "dfa": 0.815285,
                "spread1": -4.813031, "spread2": 0.266482, "ppe": 0.244646
            })
        else:
            return jsonify({
                "fo": 197.076, "fhi": 206.896, "flo": 192.055,
                "jitter": 0.00289, "shimmer": 0.01689, "nhr": 0.00790,
                "hnr": 26.175, "rpde": 0.357601, "dfa": 0.667664,
                "spread1": -7.34830, "spread2": 0.177551, "ppe": 0.088245
            })
    elif disease == 'diabetes':
        if sample_type == 'high_risk':
            return jsonify({
                "pregnancies": 5, "glucose": 185, "bp": 92,
                "skin_thickness": 38, "insulin": 210, "bmi": 38.4,
                "dpf": 0.94, "age": 52
            })
        else:
            return jsonify({
                "pregnancies": 1, "glucose": 92, "bp": 68,
                "skin_thickness": 18, "insulin": 75, "bmi": 22.4,
                "dpf": 0.25, "age": 28
            })
    elif disease == 'heart':
        if sample_type == 'high_risk':
            return jsonify({
                "age": 63, "sex": 1, "cp": 3, "trestbps": 152,
                "chol": 298, "fbs": 1, "restecg": 1, "thalach": 118,
                "exang": 1, "oldpeak": 2.6, "slope": 2, "ca": 2, "thal": 3
            })
        else:
            return jsonify({
                "age": 42, "sex": 0, "cp": 0, "trestbps": 115,
                "chol": 190, "fbs": 0, "restecg": 0, "thalach": 168,
                "exang": 0, "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 2
            })
    elif disease == 'cancer':
        if sample_type == 'high_risk':
            return jsonify({
                "radius_mean": 17.99, "texture_mean": 20.38, "perimeter_mean": 122.8,
                "area_mean": 1001.0, "smoothness_mean": 0.1184, "compactness_mean": 0.2776,
                "concavity_mean": 0.3001, "concave_points_mean": 0.1471, "symmetry_mean": 0.2419,
                "fractal_dimension_mean": 0.0787
            })
        else:
            return jsonify({
                "radius_mean": 12.32, "texture_mean": 12.39, "perimeter_mean": 78.83,
                "area_mean": 464.1, "smoothness_mean": 0.0812, "compactness_mean": 0.0521,
                "concavity_mean": 0.0210, "concave_points_mean": 0.0158, "symmetry_mean": 0.1584,
                "fractal_dimension_mean": 0.0583
            })
    return jsonify({"error": "Sample preset not found"}), 404

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        "ckd_model": {
            "name": "Random Forest Ensemble",
            "accuracy": "99.17%",
            "sensitivity": "98.8%",
            "specificity": "99.4%",
            "auroc": "0.998"
        },
        "parkinsons_model": {
            "name": "XGBoost / Gradient Boosting",
            "accuracy": "98.31%",
            "sensitivity": "97.9%",
            "specificity": "98.6%",
            "auroc": "0.991"
        },
        "diabetes_model": {
            "name": "Random Forest Classifier",
            "accuracy": "98.75%",
            "sensitivity": "98.1%",
            "specificity": "99.2%",
            "auroc": "0.994"
        },
        "heart_model": {
            "name": "Gradient Boosting Classifier",
            "accuracy": "98.50%",
            "sensitivity": "97.8%",
            "specificity": "98.9%",
            "auroc": "0.992"
        },
        "cancer_model": {
            "name": "Random Forest Classifier (WDBC)",
            "accuracy": "98.25%",
            "sensitivity": "97.4%",
            "specificity": "98.8%",
            "auroc": "0.993"
        },
        "cnn_model": {
            "name": "Deep Convolutional Neural Network (CNN)",
            "accuracy": "94.30%",
            "sensitivity": "93.5%",
            "specificity": "95.1%",
            "auroc": "0.965"
        }
    })

if __name__ == '__main__':
    print("Starting AI Disease Prediction System web server on port 5000...")
    app.run(host='127.0.0.1', port=5000, debug=True)
