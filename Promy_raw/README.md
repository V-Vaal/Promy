# Promy_raw

Dossier des données brutes du projet. Le dataset principal n'est pas versionné, il doit être téléchargé séparément depuis Kaggle.

## Structure attendue

```
Promy_raw/
└── datasets/
    ├── High-Quality Invoice Images for OCR/   (à télécharger, non versionné)
    │   ├── batch_1/                           (1 413 factures + CSV d'annotations)
    │   └── batch_2/                           (300 factures non annotées)
    ├── invoiceextra.png                       (image de test hors corpus, versionnée)
    └── promy_catalog_v1/                      (images de test catalogue, versionnées)
```

## Télécharger le dataset Kaggle

Le projet utilise le dataset **High Quality Invoice Images for OCR** :

https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr

1. Télécharger l'archive depuis Kaggle (compte Kaggle requis).
2. Extraire dans `Promy_raw/datasets/High-Quality Invoice Images for OCR/`.
3. Vérifier la présence des sous-dossiers `batch_1/` et `batch_2/` à la racine extraite.

Alternative : depuis NB3, passer `ALLOW_KAGGLEHUB_FALLBACK = True` (cellule 3) pour déclencher le téléchargement automatique via `kagglehub`, à condition d'avoir configuré une clé Kaggle locale.

## Images de test hors corpus

Les fichiers `invoiceextra.png` et `promy_catalog_v1/` sont versionnés. Ce sont des images hors du dataset Kaggle utilisées pour les tests qualitatifs A/B dans `NB3`. Scans internes anonymisés, à usage strictement pédagogique.
