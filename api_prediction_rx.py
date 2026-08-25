# ============================================================
# API de classification de radiographies thoraciques
# Modele : EfficientNetB0 Fine-tune
# AUC-ROC : 0.870
# Auteur : Octavien YAMESSE
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import io
import tensorflow as tf

# ============================================================
# CHARGEMENT DU MODELE AU DEMARRAGE
# ============================================================
try:
    modele = tf.keras.models.load_model("modele_rx_pneumonie_finetuned.keras")
    print("Modele charge avec succes !")
except Exception as e:
    modele = None
    print(f"AVERTISSEMENT : modele introuvable — {e}")

# Seuil optimise via courbe ROC (Youden Index)
SEUIL_OPTIMAL = 0.635

app = FastAPI(
    title="API Classification RX Thorax — Pneumonie",
    description="Classifie une radiographie thoracique : NORMAL ou PNEUMONIE",
    version="1.0.0"
)

@app.get("/")
def accueil():
    return {
        "message": "API Classification RX Thorax",
        "version": "1.0.0",
        "modele": "EfficientNetB0 Fine-tune",
        "classes": ["NORMAL", "PNEUMONIE"]
    }

@app.get("/health")
def health():
    return {
        "statut": "OK" if modele is not None else "DEGRADE",
        "modele": "EfficientNetB0",
        "auc_roc": 0.870,
        "seuil_decision": SEUIL_OPTIMAL
    }

@app.post("/predict")
async def predire_pneumonie(file: UploadFile = File(...)):
    # Verification du type de fichier
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Format invalide : {file.content_type}. Acceptes : JPEG, PNG"
        )

    if modele is None:
        raise HTTPException(
            status_code=503,
            detail="Modele non disponible"
        )

    # Lecture et preprocessing de l'image
    contenu = await file.read()
    image = Image.open(io.BytesIO(contenu)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Inference
    proba = float(modele.predict(image_array, verbose=0)[0][0])

    # Classification selon seuil optimise
    if proba >= SEUIL_OPTIMAL:
        classe = "PNEUMONIE"
        niveau_risque = "ELEVE" if proba >= 0.8 else "MODERE"
    else:
        classe = "NORMAL"
        niveau_risque = "FAIBLE"

    return {
        "fichier": file.filename,
        "classe_predite": classe,
        "probabilite_pneumonie": round(proba, 3),
        "probabilite_normal": round(1 - proba, 3),
        "niveau_risque": niveau_risque,
        "seuil_utilise": SEUIL_OPTIMAL,
        "interpretation": f"Probabilite de pneumonie : {round(proba * 100, 1)}%"
    }
