# Résultats de quantization (validation, 1817 images, CPU PC / ONNX Runtime)

| Modèle | Taille | mAP50 | mAP50-95 | Précision | Rappel | Latence |
|---|---|---|---|---|---|---|
| FP32 (.pt) — **meilleur modèle** | 5,4 Mo | 0,912 | 0,621 | 0,877 | 0,790 | n/a |
| FP32 (ONNX) | 9,8 Mo | 0,916 | 0,593 | 0,892 | 0,810 | 47 ms |
| INT8 dynamique (ONNX) | 2,8 Mo | 0,912 | 0,576 | 0,895 | 0,808 | 503 ms |
| INT8 statique (ONNX) | 3,0 Mo | 0,000 | 0,000 | 0 | 0 | 124 ms |
| FP16 (ONNX) | n/a | échec de conversion | | | | |
| QAT (.pt) | 5,8 Mo | 0,798 | 0,403 | 0,812 | 0,690 | n/a |
| QAT (ONNX) | 10,0 Mo | 0,794 | 0,402 | 0,806 | 0,689 | 74 ms |

La latence PC est indicative ; la mesure de référence se fait sur le Pixel (bouton CPU/GPU de l'app).
Le notebook `notebook_mobile.ipynb` (section « Quantization Benchmark ») régénère ces fichiers.
