# PaddleOCR_Invoice_v2

Modèle REC PaddleOCR fine-tuné issu du notebook NB3.

Contenu :
- `rec_infer/inference.json` : graphe d'inférence exporté (format PIR)
- `rec_infer/inference.pdiparams` : poids du modèle
- `rec_infer/inference.yml` : configuration d'inférence PaddleOCR
- `rec_infer/en_dict.txt` : dictionnaire de caractères utilisé pour le REC

Provenance :
- meilleur checkpoint sélectionné depuis `runs/rec`
- export standard PaddleOCR depuis NB3
- packaging réalisé automatiquement dans le notebook
