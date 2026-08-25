
# Classification de radiographies thoraciques  Detection de pneumonie

## Contexte

Projet de Computer Vision applique a l'imagerie medicale.

Classification binaire de radiographies thoraciques :

NORMAL vs PNEUMONIE.

## Dataset

Chest X-Ray Images (Pneumonia) — Kaggle

5 863 images RX thorax :

- Train : 5 216 images (NORMAL : 1 341 / PNEUMONIA : 3 875)

- Test  :   624 images (NORMAL :   234 / PNEUMONIA :   390)

Desequilibre de classes : 74% PNEUMONIE vs 26% NORMAL

→ gere par class weights et seuil de decision optimise

## Architecture du modele

Transfer Learning avec EfficientNetB0 pre-entraine sur ImageNet :

- Base : EfficientNetB0 (gele puis debloque pour fine-tuning)

- GlobalAveragePooling2D

- BatchNormalization

- Dense(128, relu)

- Dropout(0.3)

- Dense(1, sigmoid)

Total params    : 4 218 788

Trainable (base): 166 657 (avant fine-tuning)

## Resultats — Progression en 3 etapes

| Etape | Technique | AUC-ROC | Accuracy |

|-------|-----------|---------|----------|

| 1 | Transfer Learning (baseline) | 0.805 | 73% |

| 2 | + Class weights | 0.809 | 74% |

| 3 | + Fine-tuning (30 dernieres couches) | 0.870 | 78% |

## Resultats finaux (Fine-tuning)

| Classe | Precision | Recall | F1-score |

|--------|-----------|--------|----------|

| NORMAL | 0.65 | 0.91 | 0.75 |

| PNEUMONIA | 0.93 | 0.70 | 0.80 |

| **Macro avg** | **0.79** | **0.80** | **0.78** |

AUC-ROC : 0.870

Seuil de decision optimise : 0.635

## Points cles

- NORMAL recall : 91% → sur 100 patients sains, 91 correctement identifies

- PNEUMONIA precision : 93% → sur 100 predits malades, 93 le sont vraiment

- Seuil optimise via courbe ROC (Youden Index)

## Limitations

Ce projet est un prototype a visee pedagogique et technique.

Les performances obtenues ne constituent pas une validation clinique.

Pour un usage medical reel, une validation rigoureuse sur des

cohortes reelles et une certification seraient necessaires.

## Stack technique

- Python 3.11

- TensorFlow / Keras

- EfficientNetB0 (Transfer Learning)

- scikit-learn

- Google Colab (GPU T4)

- Git + GitHub

## Lancer le projet

1. Ouvrir rx_pneumonie_cv.ipynb dans Google Colab

2. Activer le GPU : Execution > Modifier le type d'execution > T4

3. Executer toutes les cellules dans l'ordre

