# ============================================================
# API Classification RX Thorax - Pneumonie
# Modele : EfficientNetB0 Fine-tune - AUC 0.870
# Auteur : Octavien YAMESSE
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import cv2

try:
    modele = tf.keras.models.load_model(
        "modele_rx_pneumonie_finetuned.keras"
    )
    print("Modele charge avec succes !")
except Exception as e:
    modele = None
    print(f"AVERTISSEMENT : modele introuvable - {e}")

SEUIL_OPTIMAL = 0.50

app = FastAPI(
    title="API Classification RX Thorax - Pneumonie",
    description="Classifie une radiographie thoracique : NORMAL ou PNEUMONIE",
    version="1.1.0"
)

def preprocess_medical_image(image_bytes):
    """
    Preprocessing ameliore avec CLAHE.
    CLAHE = Contrast Limited Adaptive Histogram Equalization.
    Normalise le contraste des images provenant de sources
    differentes (differents hopitaux, differents scanners).
    Resout en partie le probleme de distribution shift.
    """
    # 1. Decoder l'image en niveaux de gris
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image impossible a decoder")

    # 2. Appliquer CLAHE pour normaliser le contraste medical
    # clipLimit=2.0 : limite la sur-amplification du bruit
    # tileGridSize=(8,8) : divise l'image en 64 zones
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # 3. Convertir en RGB (3 canaux pour EfficientNetB0)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    # 4. Redimensionner a 224x224
    img_resized = cv2.resize(img_rgb, (224, 224))

    # 5. Normaliser entre 0 et 1
    img_array = img_resized.astype(np.float32) / 255.0

    return np.expand_dims(img_array, axis=0)


@app.get("/")
def accueil():
    return {
        "message": "API Classification RX Thorax",
        "version": "1.1.0",
        "modele": "EfficientNetB0 Fine-tune",
        "nouveaute": "Preprocessing CLAHE + Zone incertitude",
        "classes": ["NORMAL", "PNEUMONIE", "INCERTAIN"]
    }

@app.get("/health")
def health():
    return {
        "statut": "OK" if modele is not None else "DEGRADE",
        "modele": "EfficientNetB0",
        "auc_roc": 0.870,
        "seuil_decision": SEUIL_OPTIMAL,
        "preprocessing": "CLAHE actif"
    }

@app.post("/predict")
async def predire_pneumonie(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Format invalide. Acceptes : JPEG, PNG"
        )

    if modele is None:
        raise HTTPException(
            status_code=503,
            detail="Modele non disponible"
        )

    # Lire l'image
    contenu = await file.read()

    # Preprocessing CLAHE
    try:
        image_array = preprocess_medical_image(contenu)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur preprocessing : {str(e)}"
        )

    # Inference
    proba = float(modele.predict(image_array, verbose=0)[0][0])

    # Zone d'incertitude (proposition Gemini)
    # Si la probabilite est entre 35% et 65%
    # le modele n'est pas certain
    # → souvent du au distribution shift
    if 0.35 <= proba <= 0.65:
        classe = "INCERTAIN"
        niveau_risque = "INCERTAIN"
        avertissement = (
            "La qualite ou la provenance de cette radiographie "
            "ne permet pas au modele de statuer avec certitude. "
            "Ce resultat peut indiquer une image hors distribution "
            "(source differente du dataset d'entrainement). "
            "Une analyse radiologique humaine est indispensable."
        )
    elif proba > 0.65:
        classe = "PNEUMONIE"
        niveau_risque = "ELEVE" if proba >= 0.8 else "MODERE"
        avertissement = None
    else:
        classe = "NORMAL"
        niveau_risque = "FAIBLE"
        avertissement = None

    reponse = {
        "fichier": file.filename,
        "classe_predite": classe,
        "probabilite_pneumonie": round(proba, 3),
        "probabilite_normal": round(1 - proba, 3),
        "niveau_risque": niveau_risque,
        "seuil_utilise": SEUIL_OPTIMAL,
        "zone_incertitude": "35% - 65%",
        "interpretation": f"Probabilite de pneumonie : {round(proba * 100, 1)}%",
        "preprocessing": "CLAHE applique",
    }

    if avertissement:
        reponse["avertissement"] = avertissement

    return reponse
