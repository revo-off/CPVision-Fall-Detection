"""
À exécuter dans Google Colab (gratuit), cellule par cellule.

1) Cellule d'installation :
       !pip install -q ultralytics
2) Téléversez best_nonqat.pt dans le panneau « Fichiers » de Colab.
3) Cellule d'export + vérification : le code ci-dessous.
4) Téléchargez le fichier *_float32.tflite, renommez-le  model.tflite
   et placez-le dans  app/src/main/assets/
"""
import glob
from ultralytics import YOLO

# --- Export FP32 (FP16 n'est pas pris en charge par le format LiteRT actuel) ---
model = YOLO("best_nonqat.pt")
path = model.export(format="litert", imgsz=640, quantize=None)
print("Export terminé :", path)

# Variante INT8 (plus petite, à comparer) : décommentez et fournissez le data.yaml
# (selon la version d'Ultralytics l'option s'appelle int8=True ou quantize=8)
# model.export(format="litert", imgsz=640, quantize=8, data="data.yaml")

# --- Vérification : formes d'entrée/sortie attendues ---
import tensorflow as tf
f = sorted(glob.glob("best_nonqat_saved_model/*float32.tflite"))[0]
it = tf.lite.Interpreter(model_path=f)
it.allocate_tensors()
i = it.get_input_details()[0]
print("ENTREE :", i["shape"], i["dtype"])            # attendu : [1 640 640 3] float32
for o in it.get_output_details():
    print("SORTIE :", o["shape"], o["dtype"])          # attendu : [1 300 6]
print("Fichier à copier :", f)
