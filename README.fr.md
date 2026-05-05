# Promy : OCR par fine-tuning PaddleOCR REC

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/licence-MIT-green)

**[English version](README.md)**

Pipeline de reconnaissance optique de caractères spécialisé pour l'extraction automatique de données sur images de factures. Le module de reconnaissance (REC) de PaddleOCR, architecture CRNN (MobileNetV3 + BiLSTM + CTC), est fine-tuné sur un corpus Kaggle de 1 413 factures annotées. Le pipeline est démontrable via une application Docker exposant une API FastAPI et une interface Streamlit.

Projet réalisé dans le cadre de la certification RNCP38616 Alyra, bloc 05 : conception d'un service IA en production.

## Table des matières

1. [Ce que fait le projet](#ce-que-fait-le-projet)
2. [Architecture du pipeline](#architecture-du-pipeline)
3. [Dataset](#dataset)
4. [Prétraitement](#prétraitement)
5. [Détection](#détection)
6. [Modèle de reconnaissance](#modèle-de-reconnaissance)
7. [Comparaison de modèles](#comparaison-de-modèles)
8. [Métriques d'évaluation](#métriques-dévaluation)
9. [Résultats clés](#résultats-clés)
10. [API et démonstrateur](#api-et-démonstrateur)
11. [Structure du dépôt](#structure-du-dépôt)
12. [Prérequis et lancement](#prérequis-et-lancement)
13. [Lire les notebooks](#lire-les-notebooks)
14. [Ré-entraîner le modèle](#ré-entraîner-le-modèle-optionnel)
15. [Limitations](#limitations)
16. [Améliorations futures](#améliorations-futures)
17. [Stack technique](#stack-technique)
18. [Licences et dataset](#licences-et-dataset)
19. [Auteur](#auteur)

## Ce que fait le projet

Le démonstrateur reçoit une image (JPG ou PNG, jusqu'à 10 Mo) et renvoie une sortie structurée ligne par ligne contenant le texte reconnu, la confiance associée par ligne, et les métadonnées de prétraitement (deskew, taille originale, taille après preprocessing).

En interne, le pipeline enchaîne :

1. **Prétraitement image** (`deployment/preprocessing.py`) : conversion en niveaux de gris via espace LAB, CLAHE pour rehausser le contraste, correction de skew, débruitage léger, normalisation de résolution.
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

Seul le REC est fine-tuné. La détection est déléguée à RapidOCR non réentraînée : c'est une limite assumée du périmètre, documentée dans les notebooks.

## Dataset

**High Quality Invoice Images for OCR** (Kaggle, Osama Hosam Abdellatif)

- 1 413 factures annotées (batch_1, entraînement)
- 300 factures non annotées (batch_2, validation qualitative)
- 2 images hors corpus pour les tests A/B

Lien : https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr

Le dataset n'est pas redistribué dans ce dépôt.

## Prétraitement

Le module de prétraitement (`deployment/preprocessing.py`, présent aussi dans `notebooks/preprocessing.py`) applique les étapes suivantes :

1. Conversion en niveaux de gris via espace colorimétrique LAB
2. Rehaussement de contraste par CLAHE
3. Correction de skew (deskew)
4. Débruitage léger
5. Normalisation de résolution

Ce module est partagé entre l'environnement notebook et l'API déployée.

## Détection

La détection de texte utilise **RapidOCR** avec DBNet au format ONNX. Il localise les zones de texte sur l'image complète de la facture et produit des bounding boxes transmises au module de reconnaissance.

RapidOCR est utilisé tel quel, sans fine-tuning. Un benchmark des alternatives de détection est documenté dans `NB_DET_Benchmark.ipynb`.

## Modèle de reconnaissance

Le modèle de reconnaissance est l'architecture **CRNN** de PaddleOCR :

- Backbone : MobileNetV3
- Modélisation séquentielle : BiLSTM
- Décodeur : CTC
- Taille approximative : 8 M paramètres

Il est fine-tuné sur des crops de factures générés par pseudo-labelling à partir des annotations batch_1. L'entraînement utilise un split 75/25 anti-leakage par facture. La durée est de 40 epochs ; le meilleur checkpoint est sélectionné à l'epoch 34 sur `val_norm_edit_dis`.

## Comparaison de modèles

Le notebook `NB_Comparatif.ipynb` documente une comparaison quantitative entre :

- **TrOCR** (Microsoft, Transformer)
- **PaddleOCR CRNN** (fine-tuné sur factures)

PaddleOCR CRNN a été retenu pour sa latence d'inférence plus faible, son empreinte mémoire réduite et son adéquation au contexte d'un démonstrateur. L'expérimentation TrOCR est conservée dans `NB_experiment_TrOCR.ipynb`.

## Métriques d'évaluation

Le fine-tuning utilise `norm_edit_dis` (distance d'édition normalisée) comme métrique d'entraînement, équivalent à 1 - CER au niveau caractère. Les métriques par epoch sont disponibles dans :

- `workspace_paddleocr_invoice/runs/metrics/rec_epoch_metrics.csv`
- `workspace_paddleocr_invoice/runs/metrics/rec_epoch_metrics.png`

## Résultats clés

Résultats sur le jeu de validation interne (split 75/25 anti-leakage par facture) :

- **CER proxy** : 0,19 % sur la validation
- **Latence d'inférence** : 3,4 ms par crop (batch=12) sur GPU L4

Ces chiffres correspondent à un benchmark contrôlé sur le corpus d'entraînement. Les performances peuvent varier sur des formats de factures significativement différents.

## API et démonstrateur

Le package de déploiement expose :

**FastAPI** (port 8000) :
- `GET /health` : ping de santé
- `POST /ocr` : multipart file upload, renvoie `{lines, confidences, mean_confidence, n_segments, preprocessing}`
- Docs Swagger : http://localhost:8000/docs

**Streamlit** (port 8501) : interface web pour déposer une facture, ajuster le seuil de confiance, visualiser le tableau et télécharger le CSV.

Les deux services sont packagés ensemble avec Docker Compose.

## Structure du dépôt

```
Promy/
├── deployment/                     # Démonstrateur Docker (API + interface)
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
│   ├── NB_Comparatif.ipynb         # Arbitrage TrOCR vs PaddleOCR
│   ├── NB_DET_Benchmark.ipynb      # Benchmark de la détection
│   ├── NB_experiment_TrOCR.ipynb   # Expérimentation TrOCR (archive)
│   ├── preprocessing.py
│   └── outputs/
│
├── models/                         # Modèle + métriques du run final
│   └── PaddleOCR_Invoice_v2/
│       ├── rec_infer/
│       ├── latency_benchmark.json
│       └── README.md
│
├── workspace_paddleocr_invoice/    # Artefacts d'entraînement (voir README local)
│   ├── export/rec_infer/
│   ├── runs/
│   │   ├── rec/
│   │   │   ├── config.yml          # Config PaddleOCR du fine-tuning
│   │   │   └── train.log           # Log d'entraînement (40 epochs)
│   │   └── metrics/
│   ├── prepared_data/
│   ├── testsAB_outputs/
│   └── README.md
│
├── Promy_raw/                      # Données brutes (voir README local)
│   └── datasets/
│
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
└── README.fr.md
```

## Prérequis et lancement

### Prérequis

Pour le démonstrateur Docker uniquement :
- Docker 24+ et Docker Compose v2

Pour ouvrir et exécuter les notebooks localement :
- Python 3.12
- [uv](https://docs.astral.sh/uv/) pour la gestion d'environnement
- Optionnel : GPU NVIDIA avec CUDA pour ré-entraîner le REC

### Récupérer le dataset

1. Télécharger depuis Kaggle :

   https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr

2. Décompresser dans :

   ```
   Promy_raw/datasets/High-Quality Invoice Images for OCR/
   ├── batch_1/
   │   ├── *.csv
   │   └── images...
   └── batch_2/
       └── images...
   ```

3. Alternative via `kagglehub` (requiert une clé Kaggle configurée) : dans NB3, passer `ALLOW_KAGGLEHUB_FALLBACK = True` (cellule 3).

Les deux images hors corpus utilisées pour les tests A/B sont présentes dans `Promy_raw/datasets/` et exploitables directement.

### Lancer le démonstrateur Docker

```bash
cd deployment
docker compose up -d --build
```

Une fois les services lancés :
- Interface Streamlit : http://localhost:8501
- Docs FastAPI : http://localhost:8000/docs

## Lire les notebooks

Les notebooks sont principalement rédigés en français afin de documenter finement la démarche et les choix techniques. Chaque notebook inclut un résumé anglais en tête pour permettre aux lecteurs non francophones de comprendre le rôle de chaque étape.

Ordre de lecture recommandé :

1. `NB1_EDA.ipynb` : compréhension du dataset, inventaire des annotations, biais
2. `NB2_Preprocessing.ipynb` : pipeline preprocessing image, justification des étapes
3. `NB3_Fine-tuning_DETRapidOCR_RECPaddleOCR.ipynb` : pseudo-labelling, split anti-leakage, fine-tuning REC, export, tests A/B
4. `NB_Comparatif.ipynb` : arbitrage chiffré TrOCR vs PaddleOCR
5. `NB_DET_Benchmark.ipynb` : benchmark détection
6. `NB_experiment_TrOCR.ipynb` : archive narrative (expérimentation TrOCR écartée)

## Ré-entraîner le modèle (optionnel)

1. Cloner PaddleOCR dans le workspace :

   ```bash
   cd workspace_paddleocr_invoice
   git clone https://github.com/PaddlePaddle/PaddleOCR.git .
   ```

2. Télécharger les poids pré-entraînés référencés dans `runs/rec/config.yml` (section `Global.pretrained_model`).

3. Dans NB3, passer :
   - `FORCE_REBUILD_PREPARED_DATA = True` (cellule 5) pour regénérer les pseudo-labels
   - `RUN_REC_TRAINING = True` (cellule 5) pour déclencher l'entraînement

4. Les checkpoints sont écrits dans `runs/rec/` et le meilleur modèle exporté dans `export/rec_infer/`.

Voir `workspace_paddleocr_invoice/README.md` pour les détails.

## Limitations

- **La détection n'est pas fine-tunée.** RapidOCR est utilisé tel quel. Les performances sur des layouts atypiques dépendent du modèle DBNet pré-entraîné.
- **Espaces entre mots.** Le modèle CRNN peut manquer des espaces dans certaines configurations.
- **Couverture du français.** Le modèle de base et le corpus d'entraînement sont à dominante anglaise. Les performances sur des factures en français ne sont pas complètement caractérisées.
- **Pas d'extraction structurée.** Le pipeline produit des lignes de texte brutes. Il n'extrait pas automatiquement des champs structurés (montants, dates, fournisseurs).
- **Diversité de templates.** Les résultats peuvent se dégrader sur des formats de factures très différents du corpus d'entraînement.
- **Périmètre démonstrateur.** Le déploiement Docker est un démonstrateur, pas un système de production.

## Améliorations futures

- Fine-tuner le module de détection sur des layouts de factures
- Ajouter une couche d'extraction de champs structurés (KIE)
- Élargir la couverture du français dans le corpus d'entraînement
- Benchmarker sur un panel plus large de templates de factures
- Mettre en place un pipeline CI pour les tests de régression automatisés

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.12 |
| Reconnaissance OCR | PaddleOCR (fine-tuning CRNN) |
| Détection OCR | RapidOCR (DBNet ONNX) |
| Preprocessing image | OpenCV, NumPy, Pillow |
| Expérimentation modèles | PyTorch, Hugging Face Transformers (TrOCR) |
| API | FastAPI |
| Interface | Streamlit |
| Déploiement | Docker, Docker Compose |
| Environnement | uv |

## Licences et dataset

- **Code** : sous licence MIT, sauf mention contraire dans les fichiers sources.
- **Dataset Kaggle** : la licence du dataset High Quality Invoice Images for OCR est celle fixée par son auteur sur Kaggle. Le dataset n'est pas redistribué ici.
- **PaddleOCR** : Apache 2.0 - https://github.com/PaddlePaddle/PaddleOCR
- **RapidOCR** : Apache 2.0 - https://github.com/RapidAI/RapidOCR
- **Images de test hors-corpus** présentes dans `Promy_raw/datasets/` : scans internes anonymisés à usage strictement pédagogique.

## Auteur

**Valentin Valluet**

- GitHub : [github.com/V-Vaal](https://github.com/V-Vaal)
- LinkedIn : [linkedin.com/in/valentin-valluet](https://linkedin.com/in/valentin-valluet)
- X : [@val2_x](https://x.com/val2_x)
