# Promy : OCR par fine-tuning PaddleOCR REC

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/licence-MIT-green)

Pipeline de reconnaissance optique de caractères spécialisé pour l'extraction automatique de données sur images.
Le module de reconnaissance (REC) de PaddleOCR, architecture CRNN (MobileNetV3 + BiLSTM + CTC), est fine-tuné sur un corpus Kaggle de 1 413 factures annotées.
Le pipeline est démontrable via une application Docker exposant une API FastAPI et une interface Streamlit.

Projet réalisé dans le cadre de la certification RNCP38616 Alyra, bloc 05 : conception d'un service IA en production.

## Table des matières

1. [Ce que fait le projet](#ce-que-fait-le-projet)
2. [Architecture du pipeline](#architecture-du-pipeline)
3. [Structure du dépôt](#structure-du-dépôt)
4. [Prérequis](#prérequis)
5. [Récupérer le dataset](#récupérer-le-dataset)
6. [Lancer le démonstrateur Docker](#lancer-le-démonstrateur-docker)
7. [Lire les notebooks](#lire-les-notebooks)
8. [Ré-entraîner le modèle](#ré-entraîner-le-modèle-optionnel)
9. [Licences et dataset](#licences-et-dataset)

## Ce que fait le projet

Le démonstrateur reçoit une image (JPG ou PNG, jusqu'à 10 Mo) et renvoie une sortie structurée ligne par ligne contenant le texte reconnu, la confiance associée par ligne, et les métadonnées de prétraitement (deskew, taille originale, taille après preprocessing).

En interne, le pipeline enchaîne :

1. **Prétraitement image** (`deployment/preprocessing.py`) : conversion en niveaux de gris via espace LAB, CLAHE pour rehausser le contraste, correction de skew, debruitage léger, normalisation de résolution.
2. **Détection de texte (DET)** : RapidOCR (DBNet ONNX) pour localiser chaque zone de texte sur la facture complète.
3. **Reconnaissance de caractères (REC)** : PaddleOCR CRNN fine-tuné sur factures, appliqué à chaque crop de ligne de texte.
4. **Agrégation** : tri des lignes par confiance, export JSON / CSV.

Le REC fine-tuné atteint un CER proxy de 0,19 % sur la validation interne (split 75/25 anti-leakage par facture), pour une latence d'inférence de 3,4 ms par crop (batch=12) sur GPU L4.

## Architecture du pipeline

```
Image brute (JPG/PNG)
        |
        v
+---------------------------+
| Preprocessing             |  grayscale LAB, CLAHE, deskew, denoise, resize
| deployment/preprocessing  |
+---------------------------+
        |
        v
+---------------------------+
| DET : RapidOCR (DBNet)    |  bounding boxes texte
| ONNX embarqué             |
+---------------------------+
        |
        v (crops ligne par ligne)
+---------------------------+
| REC : PaddleOCR CRNN      |  MobileNetV3 + BiLSTM + CTC
| fine-tuné sur factures    |  ~8 M paramètres
| deployment/models/rec_infer
+---------------------------+
        |
        v
Sortie structurée (JSON / CSV)
    - lignes
    - confiances
    - métadonnées preprocessing
```

Seul le REC est fine-tuné. La détection est déléguée à RapidOCR non réentraînée : c'est une limite assumée du périmètre, documentée dans les notebooks et le livret de certification.

## Structure du dépôt

```
Promy/
├── deployment/                     # Démonstrateur Docker (livrable BC05)
│   ├── api/                        # API FastAPI, routes /ocr et /health
│   ├── front/                      # App Streamlit
│   ├── models/rec_infer/           # Modèle CRNN fine-tuné (inference)
│   ├── preprocessing.py              
│   ├── tests/                      # Tests pytest API + vendor
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── pyproject.toml
│
├── notebooks/                      # Notebooks du projet DL
│   ├── NB1_EDA.ipynb                       
│   ├── NB2_Preprocessing.ipynb             
│   ├── NB3_Fine-tuning_DETRapidOCR_RECPaddleOCR.ipynb
│   ├── NB_Comparatif.ipynb                 # Arbitrage TrOCR vs PaddleOCR
│   ├── NB_DET_Benchmark.ipynb              # Benchmark de la détection
│   ├── NB_experiment_TrOCR.ipynb           # Expérimentation TrOCR
│   ├── preprocessing.py
│   └── outputs/                            # Outputs de runs de notebooks
│
├── models/                         # Modèle + métriques du run final
│   └── PaddleOCR_Invoice_v2/
│       ├── rec_infer/              # Modèle exporté
│       ├── latency_benchmark.json
│       └── README.md
│
├── workspace_paddleocr_invoice/    # Artefacts d'entraînement (voir README local)
│   ├── export/rec_infer/           # Modèle exporté (inference.json, pdiparams, yml)
│   ├── runs/
│   │   ├── rec/
│   │   │   ├── config.yml          # Config PaddleOCR du fine-tuning
│   │   │   └── train.log           # Log d'entraînement (40 epochs)
│   │   └── metrics/                # CSV + plots des métriques par epoch
│   ├── prepared_data/
│   │   └── prepared_data/
│   │       ├── pseudo_quality_train.csv
│   │       └── pseudo_quality_val.csv
│   ├── testsAB_outputs/            # Résultats des tests A/B hors corpus
│   └── inference_raw_text.{csv,json,txt}
│
├── Promy_raw/                      # Données brutes (voir README local)
│   └── datasets/                   # Dataset Kaggle + images de test
│
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Prérequis

Pour le démonstrateur Docker uniquement :
- Docker 24+ et Docker Compose v2

Pour ouvrir et exécuter les notebooks localement :
- Python 3.12
- [uv](https://docs.astral.sh/uv/) pour la gestion d'environnement
- Optionnel : GPU NVIDIA avec CUDA pour ré-entraîner le REC

## Récupérer le dataset

Le dataset principal est **High Quality Invoice Images for OCR** (Kaggle) : 1 413 factures annotées pour l'entraînement (`batch_1`) + 300 factures non annotées pour validation qualitative (`batch_2`).

1. Télécharger le dataset depuis Kaggle :

   https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr

2. Décompresser l'archive dans :

   ```
   Promy_raw/datasets/High-Quality Invoice Images for OCR/
   ```

   La structure attendue après extraction :
   ```
   Promy_raw/datasets/High-Quality Invoice Images for OCR/
   ├── batch_1/
   │   ├── *.csv            (annotations documentaires)
   │   └── images...
   └── batch_2/
       └── images...
   ```

3. Alternative via `kagglehub` (requiert une clé Kaggle configurée) : dans NB3, passer `ALLOW_KAGGLEHUB_FALLBACK = True` (cellule 3).

Les deux images hors corpus utilisées pour les tests A/B sont présentes dans `Promy_raw/datasets/` et exploitables directement.

## Lancer le démonstrateur Docker

Le démonstrateur expose une API FastAPI (port 8000) et une interface Streamlit (port 8501).

```bash
cd deployment
docker compose up -d --build
```

Une fois les services lancés :

- **Interface utilisateur** : http://localhost:8501
  Dépose une facture, ajuste le seuil de confiance, visualise le tableau et télécharge le CSV.

- **API FastAPI** : http://localhost:8000
  - `GET /health` : ping de santé
  - `POST /ocr` : multipart file upload, renvoie `{lines, confidences, mean_confidence, n_segments, preprocessing}`
  - Docs Swagger : http://localhost:8000/docs

## Lire les notebooks

Ordre de lecture recommandé :

1. `NB1_EDA.ipynb` : compréhension du dataset, inventaire des annotations, biais
2. `NB2_Preprocessing.ipynb` : pipeline preprocessing image, justification des étapes
3. `NB3_Fine-tuning_DETRapidOCR_RECPaddleOCR.ipynb` : pseudo-labelling, split anti-leakage, fine-tuning REC, export, tests A/B
4. `NB_Comparatif.ipynb` : arbitrage chiffré TrOCR vs PaddleOCR
5. `NB_DET_Benchmark.ipynb` : benchmark détection
6. `NB_experiment_TrOCR.ipynb` : archive narrative (expérimentation TrOCR écartée)

## Ré-entraîner le modèle (optionnel)

Le fine-tuning complet requiert le clone de PaddleOCR et les modèles pré-entraînés.

1. Cloner PaddleOCR dans le workspace :

   ```bash
   cd workspace_paddleocr_invoice
   git clone https://github.com/PaddlePaddle/PaddleOCR.git .
   ```

   Le dossier `runs/rec/config.yml` conservé dans le dépôt contient la configuration utilisée pour le fine-tuning et peut servir de référence.

2. Télécharger les poids pré-entraînés référencés dans la config (voir `pretrain_models/` du clone PaddleOCR).

3. Dans `NB3`, passer :
   - `FORCE_REBUILD_PREPARED_DATA = True` (cellule 5) pour regénérer les pseudo-labels
   - `RUN_REC_TRAINING = True` (cellule 5) pour déclencher l'entraînement

4. L'entraînement écrit ses checkpoints dans `workspace_paddleocr_invoice/runs/rec/` et exporte le meilleur modèle dans `workspace_paddleocr_invoice/export/rec_infer/`.

Voir `workspace_paddleocr_invoice/README.md` pour les détails.

## Licences et dataset

- **Code** : sous licence MIT, sauf mention contraire dans les fichiers sources.
- **Dataset Kaggle** : la licence du dataset High Quality Invoice Images for OCR est celle fixée par son auteur sur Kaggle. Le dataset n'est pas redistribué ici.
- **PaddleOCR** : Apache 2.0, voir https://github.com/PaddlePaddle/PaddleOCR
- **RapidOCR** : Apache 2.0, voir https://github.com/RapidAI/RapidOCR
- **Images de test hors-corpus** présentes dans `Promy_raw/datasets/` : scans internes anonymisés à usage strictement pédagogique.

---

## Auteur

**Valentin Valluet**

- GitHub : [github.com/V-Vaal](https://github.com/V-Vaal)
- LinkedIn : [linkedin.com/in/valentin-valluet](https://linkedin.com/in/valentin-valluet)
- X : [@val2_x](https://x.com/val2_x)
