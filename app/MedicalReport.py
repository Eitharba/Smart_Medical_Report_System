from datetime import datetime
from io import BytesIO
import os
import logging
import numpy as np
import torch
import skimage.io
from flask import Flask, request, redirect, send_file, send_from_directory, render_template
from pymongo import MongoClient
from bs4 import BeautifulSoup
from textwrap import wrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
import torchxrayvision as xrv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["medical_reports"]
collection = db["reports"]

# Load X-ray model
xray_model = xrv.models.DenseNet(weights="densenet121-res224-all")
xray_model.eval()

# Diagnosis map
diagnosis_map = {
    "Pneumonia": {"clinical": "Dense consolidation in the right lower lobe consistent with lobar pneumonia. Recommend antibiotics and clinical monitoring.",
                  "patient": "Lung infection that fills air sacs with fluid, causing cough and fever."},
    "Atelectasis": {"clinical": "Segmental atelectasis seen in the left lower zone. Suggest airway clearance.",
                    "patient": "Partial lung collapse, which may cause breathing difficulty."},
    "Pleural Effusion": {"clinical": "Left-sided pleural effusion noted. Recommend ultrasound to assess volume.",
                         "patient": "Fluid buildup around the lungs that may affect breathing."},
    "Pneumothorax": {"clinical": "Right pneumothorax present. Urgent medical attention required.",
                     "patient": "Collapsed lung due to air leakage inside the chest cavity."},
    "Nodule": {"clinical": "Solitary pulmonary nodule detected in right upper lobe. Recommend CT follow-up.",
               "patient": "A small lump in the lung. Often harmless but needs monitoring."},
    "Mass": {"clinical": "Lung mass identified. Malignancy cannot be excluded. CT and biopsy advised.",
             "patient": "A large growth in the lung that may need further testing."},
    "Emphysema": {"clinical": "Hyperinflated lungs with flattened diaphragms. Changes typical of emphysema.",
                  "patient": "Lung damage causing shortness of breath, often due to smoking."},
    "Consolidation": {"clinical": "Lobar consolidation observed. Suggestive of infection or inflammatory process.",
                       "patient": "Part of the lung filled with fluid, usually from infection."},
    "Cardiomegaly": {"clinical": "Cardiac silhouette enlargement noted. Recommend echocardiogram.",
                     "patient": "An enlarged heart, which could indicate heart strain or disease."},
    "Infiltration": {"clinical": "Patchy infiltrates seen in both lungs. Consider infectious or inflammatory cause.",
                     "patient": "Spread of fluid or cells into the lungs due to infection or irritation."},
    "Fibrosis": {"clinical": "Diffuse interstitial fibrosis noted. High-resolution CT recommended.",
                 "patient": "Scarring in the lungs that can cause stiffness and breathlessness."},
    "Edema": {"clinical": "Pulmonary edema present. Suggest cardiogenic origin. Monitor oxygen levels.",
              "patient": "Fluid accumulation in the lungs, usually due to heart problems."},
    "Hernia": {"clinical": "Air-fluid level seen in mediastinum suggests hiatal hernia.",
               "patient": "Part of the stomach pushing into the chest through diaphragm."},
    "Effusion": {"clinical": "Small bilateral pleural effusions noted. Recommend clinical correlation.",
                 "patient": "Fluid collecting between the lungs and chest wall."},
    "Pleural_Thickening": {"clinical": "Apical pleural thickening noted. May represent prior inflammation.",
                           "patient": "Thick tissue around the lung from previous disease or irritation."},
    "Lung Opacity": {"clinical": "Non-specific opacity in mid-lung zone. Further evaluation advised.",
                     "patient": "A hazy area in the lung that blocks clear X-ray view."},
    "Enlarged Cardiomediastinum": {"clinical": "Widened mediastinum possibly due to lymphadenopathy or vascular anomaly.",
                                   "patient": "Expansion of the central chest area; may need advanced imaging."},
    "No Finding": {"clinical": "No abnormal findings detected in this radiograph.",
                   "patient": "Everything appears normal. No signs of disease were found."}
}

def analyze_xray(filepath, patient_id, gender, birth_date):
    try:
        img = skimage.io.imread(filepath)
        if len(img.shape) == 3:
            img = img.mean(2)
        img = xrv.datasets.normalize(img, 255)
        img = img[None, ...]
        img = xrv.datasets.XRayResizer(224)(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        with torch.no_grad():
            output = xray_model(img_tensor)[0].numpy()

        predictions = dict(zip(xray_model.pathologies, output))
        sorted_preds = sorted(predictions.items(), key=lambda x: -abs(x[1]))
        top_findings = sorted_preds[:5]

        result_html = "<b>Top Findings:</b><br>"
        for label, score in top_findings:
            entry = diagnosis_map.get(label)
            if entry:
                result_html += f"{entry['clinical']} <span title='{entry['patient']}'>🛈</span> (Confidence: {score:.2f})<br><br>"
            else:
                result_html += f"{label}: {score:.2f}<br><br>"

        return result_html, predictions

    except Exception as e:
        logging.error(f"X-ray Analysis Error for {patient_id}: {str(e)}")
        return "⚠️ Unable to analyze image.", None

def analyze_ct(filepath):
    return "🚫 CT scan analysis is currently not supported."

def analyze_mri(filepath):
    return "🚫 MRI scan analysis is currently not supported."

def generate_diagnosis(findings):
    if "no abnormalities" in findings.lower():
        return "The scan shows no significant abnormalities. Patient appears normal."
    return f"Findings indicate: {findings}"

def calculate_age(birth_date):
    birth = datetime.strptime(birth_date, "%Y-%m-%d")
    today = datetime.today()
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))

def generate_final_diagnosis(report_html, predictions=None, gender=None, birth_date=None):
    soup = BeautifulSoup(report_html.replace("🛈", ""), "html.parser")
    lines = [l.strip() for l in soup.get_text(separator="\n").split("\n") if l.strip()]
    summary_lines = []

    if predictions:
        top_preds = [label for label, score in sorted(predictions.items(), key=lambda x: -x[1]) if score >= 0.7][:3]
        summary_lines = [diagnosis_map[label]["clinical"] for label in top_preds if label in diagnosis_map]

    if not summary_lines:
        for line in lines:
            if any(kw in line.lower() for kw in ["consistent with", "indicates", "suggests"]):
                summary_lines.append(line.strip(". "))
        if not summary_lines:
            summary_lines = lines[:2]

    return f"Findings are most consistent with: {'; '.join(summary_lines[:2])}."

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/patients")
def patient_list():
    all_reports = collection.find()
    rows = ""
    for r in all_reports:
        rows += f"""
        <tr>
            <td><img src="/static/patient_icon.png" class="avatar"></td>
            <td>{r['patient_id']}</td>
            <td>{r['scan_type']}</td>
            <td>
                <a class="btn view" href="/get_report?patient_id={r['patient_id']}" target="_blank">View</a>
                <a class="btn download" href="/download_report?patient_id={r['patient_id']}" target="_blank">Download</a>
                <a class="btn edit" href="/edit_patient?patient_id={r['patient_id']}" target="_blank">Edit</a>
            </td>
        </tr>
        """
    return render_template("patients.html", rows=rows)

@app.route("/upload", methods=["POST"])
def upload():
    patient_id = request.form["patient_id"]
    gender = request.form["gender"]
    birth_date = request.form["birth_date"]
    scan_type = request.form["scan_type"]
    scan_date = datetime.today().strftime("%Y-%m-%d")
    image = request.files["image"]

    existing = collection.find_one({"patient_id": patient_id})
    if existing:
        return redirect(f"/edit_patient?patient_id={patient_id}")

    filename = f"{patient_id}_{scan_date}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    if scan_type.lower() == "x-ray":
        findings, predictions = analyze_xray(filepath, patient_id, gender, birth_date)
    else:
        findings = analyze_ct(filepath) if scan_type.lower() == "ct-scan" else analyze_mri(filepath)
        predictions = None

    diagnosis = generate_diagnosis(findings)
    final_diagnosis = generate_final_diagnosis(findings, predictions, gender, birth_date)

    collection.insert_one({
        "patient_id": patient_id,
        "gender": gender,
        "birth_date": birth_date,
        "scan_type": scan_type,
        "scan_date": scan_date,
        "report": diagnosis,
        "final_diagnosis": final_diagnosis,
        "image_path": f"/uploads/{filename}",
        "previous_scans": []
    })

    return redirect(f"/get_report?patient_id={patient_id}")

@app.route("/edit_patient", methods=["GET", "POST"])
def edit_patient():
    patient_id = request.args.get("patient_id")
    patient = collection.find_one({"patient_id": patient_id})

    if not patient:
        return "Patient not found", 404

    if request.method == "POST":
        scan_date = request.form["scan_date"]
        scan_type = request.form["scan_type"]
        image = request.files["image"]
        gender = request.form["gender"]
        birth_date = request.form["birth_date"]

        filename = f"{patient_id}_{scan_date}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)

        if scan_type.lower() == "x-ray":
            findings, predictions = analyze_xray(filepath, patient_id, gender, birth_date)
        else:
            findings = analyze_ct(filepath) if scan_type.lower() == "ct-scan" else analyze_mri(filepath)
            predictions = None

        diagnosis = generate_diagnosis(findings)
        final_diagnosis = generate_final_diagnosis(findings, predictions, gender, birth_date)

        previous = {
            "scan_date": patient.get("scan_date"),
            "scan_type": patient.get("scan_type"),
            "report": patient.get("report"),
            "image_path": patient.get("image_path"),
            "final_diagnosis": patient.get("final_diagnosis")
        }

        scans = patient.get("previous_scans", [])
        if previous["scan_date"]:
            scans.append(previous)

        collection.update_one(
            {"patient_id": patient_id},
            {"$set": {
                "scan_date": scan_date,
                "scan_type": scan_type,
                "report": diagnosis,
                "final_diagnosis": final_diagnosis,
                "image_path": f"/uploads/{filename}",
                "previous_scans": scans
            }}
        )

        return redirect(f"/edit_patient?patient_id={patient_id}")

    scan_rows = ""
    for s in patient.get("previous_scans", []):
        img_html = f"<img src='{s['image_path']}' class='thumbnail' onclick=\"zoomImage('{s['image_path']}')\">" if s.get("image_path") else "—"
        scan_rows += f"""
        <tr>
            <td>{s.get('scan_date', 'N/A')}</td>
            <td>{s.get('scan_type', 'N/A')}</td>
            <td><pre>{s.get('report', '').strip()}</pre></td>
            <td>{img_html}</td>
        </tr>
        """

    return render_template(
        "edit_patient.html",
        patient=patient,
        patient_id=patient_id,
        scan_rows=scan_rows,
        today=datetime.today().strftime("%Y-%m-%d")
    )

@app.route("/get_report")
def get_report():
    patient_id = request.args.get("patient_id")
    report = collection.find_one({"patient_id": patient_id})

    if not report:
        return "❌ Report not found."

    final_diagnosis = report.get("final_diagnosis", "")
    age = calculate_age(report.get("birth_date"))

    alert_block = ""
    followup_block = ""

    alert_keywords = ["collapse", "mass", "urgent", "critical", "suspicious"]
    if any(word in final_diagnosis.lower() for word in alert_keywords):
        alert_block = """
        <div style='background:#fff3cd;padding:15px;border-left:5px solid #ffc107;color:#856404;margin-top:20px;'>
            ⚠️ <strong>Alert:</strong> This case may require urgent medical attention.
        </div>
        """

    followup_labels = {
        "nodule": "CT follow-up is recommended in 3–6 months.",
        "effusion": "Ultrasound or clinical re-evaluation may be needed.",
        "opacity": "Consider follow-up imaging depending on clinical status.",
        "atelectasis": "Repeat imaging may be helpful to confirm resolution.",
        "pneumothorax": "Clinical observation and follow-up X-ray recommended."
    }

    for keyword, message in followup_labels.items():
        if keyword in final_diagnosis.lower():
            followup_block = f"""
            <div style='background:#e7f3fe;padding:15px;border-left:5px solid #2196F3;color:#0c5460;margin-top:20px;'>
                ℹ️ <strong>Follow-up Suggestion:</strong> {message}
            </div>
            """
            break

    previous_rows = ""
    for scan in report.get("previous_scans", []):
        img_html = f"<img src='{scan['image_path']}' style='max-width:100px;border-radius:6px;border:1px solid #ccc;'>" if scan.get("image_path") else "—"
        previous_rows += f"""
        <tr>
            <td>{scan.get('scan_date')}</td>
            <td>{scan.get('scan_type')}</td>
            <td><pre>{scan.get('report')}</pre></td>
            <td>{img_html}</td>
        </tr>
        """

    previous_scans_block = ""
    if report.get("previous_scans"):
        previous_scans_block = f"""
        <h3 style='margin-top:40px;color:#2A9D8F;'>Previous Scans</h3>
        <table style='width:100%;border-collapse:collapse;margin-top:10px;font-size:13px;'>
            <thead>
                <tr style='background:#2A9D8F;color:white;'>
                    <th style='padding:10px;'>Date</th>
                    <th>Type</th>
                    <th>Diagnosis</th>
                    <th>Image</th>
                </tr>
            </thead>
            <tbody>
                {previous_rows}
            </tbody>
        </table>
        """

    return render_template(
        "report.html",
        report=report,
        patient_id=patient_id,
        age=age,
        final_diagnosis=final_diagnosis,
        alert_block=alert_block,
        followup_block=followup_block,
        previous_scans_block=previous_scans_block
    )

@app.route("/download_report")
def download_report():
    patient_id = request.args.get("patient_id")
    report = collection.find_one({"patient_id": patient_id})

    if not report:
        return "No report found.", 404

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle(f"Radiology_Report_{patient_id}")
    
    image_full_path = os.path.join(BASE_DIR, report.get("image_path").lstrip("/"))
    
    if os.path.exists("static/logo.png"):
        pdf.drawImage("static/logo.png", 510, 745, width=50, height=50)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.setFillColor(HexColor("#2A9D8F"))
    pdf.drawCentredString(280, 770, "Radiology Report")

    y = 730
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Patient Information")
    pdf.line(50, y - 5, 560, y - 5)
    y -= 40

    age = calculate_age(report.get("birth_date"))
    pdf.setFont("Helvetica", 11)
    pdf.setFillColor(HexColor("#333333"))

    for line in [
        f"Patient ID: {report.get('patient_id')}",
        f"Gender: {report.get('gender')}",
        f"Date of Birth: {report.get('birth_date')}",
        f"Age: {age} years",
        f"Scan Type: {report.get('scan_type')}",
        f"Scan Date: {report.get('scan_date')}"
    ]:
        pdf.drawString(60, y, line)
        y -= 30
    
    if os.path.exists(image_full_path):
      pdf.drawImage(image_full_path, 250, 250, width=300, height=300)
    else:
      print("Image not found:", image_full_path)

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"report_{patient_id}.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(debug=True)
