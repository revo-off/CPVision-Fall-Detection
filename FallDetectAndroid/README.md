# Fall Detect — YOLO26n sur Android (Pixel 7 Pro)

Application caméra temps réel : détection `fall` / `non-fall` avec votre `best_nonqat.pt`,
exécutée localement via LiteRT (TFLite) en GPU ou CPU, avec alerte (bandeau + vibration + bip).

## 1. Exporter le modèle (Google Colab, gratuit)
1. Ouvrez Colab, `!pip install -q ultralytics`, téléversez `best_nonqat.pt`.
2. Exécutez `tools/export_model.py` (affiche les formes d'entrée/sortie).
   Attendu : entrée `[1 640 640 3]` float32, sortie `[1 300 6]`.
3. Téléchargez `best_nonqat_saved_model/best_nonqat_float32.tflite`.
4. Renommez-le **`model.tflite`** et placez-le dans **`app/src/main/assets/`**.

## 2. Ouvrir et lancer
1. Installez **Android Studio** (gratuit).
2. `File > Open` → dossier `FallDetectAndroid`. Acceptez l'installation du SDK 34 si proposé.
   Attendez la fin de « Gradle sync » (internet requis).
3. Sur le Pixel 7 Pro : `Paramètres > À propos du téléphone` → touchez 7 fois « Numéro de build »,
   puis `Paramètres > Système > Options pour les développeurs` → activez **Débogage USB**.
4. Branchez en USB, acceptez l'empreinte RSA sur le téléphone.
5. Sélectionnez le Pixel en haut d'Android Studio et cliquez sur ▶ **Run**.

## 3. Utilisation
- Autorisez la caméra. Le coin haut-gauche affiche backend, latence d'inférence, fps.
- Curseur : seuil de confiance. Bouton : bascule GPU / CPU (pour votre tableau de benchmark).
- Alerte : `fall` vu sur ≥ 5 des 8 dernières images (modifiable dans `FallAlarm.kt`).

## 4. Vérifier le format de sortie (important, 1 minute)
Terminal d'Android Studio ou onglet Logcat, filtre `FallDetect` :
`adb logcat -s FallDetect`
Vous verrez « 1re ligne brute de sortie : x1, y1, x2, y2, score, classe ».
Si les boîtes sont décalées ou incohérentes, passez `BOX_FORMAT` à `BoxFormat.XYWH`
dans `YoloDetector.kt` (les coordonnées normalisées 0..1 ou en pixels sont détectées automatiquement).

## 5. Problèmes courants
- **« Modèle introuvable »** : `model.tflite` absent de `app/src/main/assets/`.
- **« Sortie inattendue … »** : le modèle n'est pas end-to-end ; réexportez avec les options par défaut.
- **Gradle ne démarre pas (wrapper)** : `File > Settings > Build Tools > Gradle`, choisissez
  « Gradle from gradle-wrapper.properties » ; si une mise à jour d'AGP est proposée, acceptez-la.
- **GPU en repli CPU** : normal si certaines opérations ne sont pas supportées ; comparez les deux.
- **« Didn't find op for builtin opcode »** : mettez à jour Google Play services sur le téléphone.

## 6. Benchmark pour le rapport
Pour chaque combinaison (FP32/FP16/INT8 × 320/416/640 × CPU/GPU) relevez la latence affichée
après ~30 s de stabilisation, la taille du fichier, et le mAP via `YOLO(model.tflite).val(...)` sur Colab.

Licence : Ultralytics est sous AGPL-3.0 (projet scolaire non commercial : publiez le dépôt si vous distribuez l'app).
