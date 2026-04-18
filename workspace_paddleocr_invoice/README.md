# workspace_paddleocr_invoice

Dossier de travail utilisé par `NB3` pour le fine-tuning du module REC de PaddleOCR.

## Contenu versionné

```
workspace_paddleocr_invoice/
├── export/
│   └── rec_infer/                          # Modèle CRNN fine-tuné (format inférence)
│       ├── inference.json
│       ├── inference.pdiparams
│       └── inference.yml
├── runs/
│   ├── rec/
│   │   ├── config.yml                      # Config PaddleOCR du fine-tuning final
│   │   └── train.log                       # Log complet (40 epochs)
│   └── metrics/
│       ├── rec_epoch_metrics.csv           # Métriques par epoch (loss, acc, norm_edit_dis)
│       ├── rec_epoch_metrics.png           # Courbes d'apprentissage
│       ├── rec_epoch_metrics_live.png      # Variante live du plot
│       └── rec_train.log
├── prepared_data/
│   └── prepared_data/
│       ├── pseudo_quality_train.csv        # Qualité du pseudo-labelling train
│       └── pseudo_quality_val.csv          # Qualité du pseudo-labelling val
├── testsAB_outputs/                        # Résultats des tests A/B hors corpus
│   ├── avril2025auchan-local*.csv/json/txt
│   ├── batch3-0501.csv/json/txt
│   └── hors_corpus_local.csv/json/txt
├── inference_raw_text.csv                  # Dump d'une inférence unitaire
├── inference_raw_text.json
└── inference_raw_text.txt
```

Le dictionnaire `en_dict.txt` utilisé par le REC est bundlé côté déploiement (`deployment/models/rec_infer/en_dict.txt`)

## Contenu exclu par `.gitignore`

- `runs/rec/*.pdopt`, `*.pdparams`, `*.states`, `runs/rec/best_model/` : checkpoints d'entraînement, régénérables.
- `prepared_data/batch2_unlabeled_images.txt` et `prepared_data/prepared_data/batch2_unlabeled_images.txt` : listes d'images régénérables.
- `prepared_data/prepared_data/rec_images/` : crops REC générés au pseudo-labelling, régénérables.

Si vous aviez un clone PaddleOCR complet (`paddleocr/`, `ppocr/`, `ppstructure/`, `tools/`, `tests/`, `configs/`, `pretrain_models/`) dans ce dossier, il n'est pas versionné. Pour ré-entraîner, re-cloner PaddleOCR comme indiqué ci-dessous.

## Reproduire le fine-tuning

1. Cloner PaddleOCR dans ce dossier :

   ```bash
   cd workspace_paddleocr_invoice
   git clone https://github.com/PaddlePaddle/PaddleOCR.git .
   ```

2. Télécharger les poids pré-entraînés nécessaires (chemin référencé dans `runs/rec/config.yml`, section `Global.pretrained_model`).

3. Placer le dataset Kaggle dans `../Promy_raw/datasets/High-Quality Invoice Images for OCR/` (voir `../Promy_raw/README.md`).

4. Ouvrir `../notebooks/NB3_Fine-tuning_DETRapidOCR_RECPaddleOCR.ipynb` et passer :
   - `FORCE_REBUILD_PREPARED_DATA = True` pour regénérer pseudo-labels + crops
   - `RUN_REC_TRAINING = True` pour déclencher l'entraînement

5. Les checkpoints seront écrits dans `runs/rec/` et le meilleur modèle exporté dans `export/rec_infer/`.

## Pourquoi garder `config.yml`, `train.log` et `metrics/` ?

- `config.yml` documente tous les hyperparamètres du run final (architecture, optimiseur, LR schedule, augmentations)
- `train.log` trace l'évolution complète sur 40 epochs, la convergence et la sélection du meilleur checkpoint (`epoch 34` sur `val_norm_edit_dis`).
- `metrics/` contient les CSV et plots de métriques par epoch.
