from gen_common import md, code, save, CONFIG_CELL, KAGGLE_CELL, IMAGES_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 04. OV : détection ouverte, segmentation et tracking

**OV** (*Open Vocabulary*) désigne des modèles de détection qui cherchent un objet à partir
d'un **mot** plutôt que d'une liste figée de catégories. Ce notebook montre trois briques
visuelles utiles au champ :
- **bounding box** : repérer un objet avec un mot-clé (« a leaf », « fruit »…) ;
- **segmentation** : isoler l'objet dans l'image ;
- **tracking** : suivre cet objet sur une petite séquence d'images.

Le modèle utilisé (**OWL-ViT base**, ~150M paramètres) est volontairement **petit et rapide**,
pour tourner confortablement sur le CPU gratuit de Google Colab. Le code reste minimal : le but
est d'apprendre à **utiliser** le modèle et à observer l'effet du **prompt** (le mot-clé donné
au détecteur), pas d'écrire beaucoup de code.
"""))

cells.append(code(CONFIG_CELL))

cells.append(code('''\
pip_install("transformers>=4.56", "accelerate", "pillow", "opencv-contrib-python")
import numpy as np
import cv2
from PIL import ImageDraw
print("✅ opencv", cv2.__version__)
'''))

cells.append(code(KAGGLE_CELL))
cells.append(code(IMAGES_CELL))

cells.append(md("""\
## 1. Charger des images agricoles (Kaggle)

On réutilise le jeu **Plant Disease Recognition** (Kaggle, `Healthy` / `Powdery` / `Rust`) déjà
vu au notebook 02, et on garde la première photo pour la démonstration pas à pas.
"""))

cells.append(code('''\
N_EXEMPLES = 3 if MODE_DEMO else 10
echantillon = echantillon_images_plantes(N_EXEMPLES)
image, vrai_label = echantillon[0]
print(f"{len(echantillon)} images chargées. Photo de démonstration : {vrai_label}")
image.thumbnail((480, 480))
image
'''))

cells.append(md("""\
## 2. Trouver un objet avec du texte (détection ouverte)

On donne un mot, et le modèle cherche dans l'image l'objet correspondant — sans avoir été
entraîné spécifiquement sur ce mot. On garde toujours la **meilleure** boîte trouvée, même si
sa confiance est faible (utile sur une photo dense, où « leaf » désigne une zone plutôt qu'un
objet unique) : on affiche alors un avertissement plutôt que de planter.
"""))

cells.append(code('''\
from transformers import pipeline

MODELE_DETECTION = "google/owlvit-base-patch32"
detecteur = pipeline("zero-shot-object-detection", model=MODELE_DETECTION, dtype="auto")

def detecter(image, texte_cible, seuil_alerte=0.05):
    resultats = detecteur(image, candidate_labels=[texte_cible], threshold=0.0)
    if not resultats:
        print(f"⚠️ Aucune détection pour « {texte_cible} ».")
        return None
    meilleure = resultats[0]
    if meilleure["score"] < seuil_alerte:
        print(f"⚠️ Confiance faible ({meilleure['score']:.2f}) pour « {texte_cible} » "
              "— résultat à vérifier.")
    return meilleure

prediction = detecter(image, "a leaf")
print(prediction)
'''))

cells.append(code('''\
def dessiner_box(image, detection, couleur="red"):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    box = detection["box"]
    draw.rectangle([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], outline=couleur, width=4)
    draw.text((box["xmin"], max(0, box["ymin"] - 18)), f"{detection['label']} {detection['score']:.2f}", fill=couleur)
    return img

dessiner_box(image, prediction)
'''))

cells.append(md("""\
## 3. Segmenter l'objet détecté

La segmentation isole les pixels de l'objet. Pour rester simple, on prend la boîte détectée
comme point de départ et on applique `grabCut` (OpenCV).
"""))

cells.append(code('''\
def segmenter(image, detection):
    arr = np.array(image)
    mask = np.zeros(arr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    box = detection["box"]
    h, w = arr.shape[:2]
    # Rogner la boîte dans l'image en gardant une marge de fond : grabCut plante
    # (cv2.error) si le rectangle couvre tout le cadre (aucun pixel de fond) — ce qui
    # arrive avec une feuille qui remplit l'image. On garde donc >= 1 px de marge.
    x0 = max(1, int(box["xmin"]))
    y0 = max(1, int(box["ymin"]))
    x1 = min(w - 1, int(box["xmax"]))
    y1 = min(h - 1, int(box["ymax"]))
    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    try:
        cv2.grabCut(arr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        print("⚠️ Segmentation impossible pour cette boîte — image d'origine renvoyée.")
        return image
    masque = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    retour = arr * masque[:, :, np.newaxis]
    return Image.fromarray(retour)

image_segmente = segmenter(image, prediction)
image_segmente
'''))

cells.append(md("""\
## 4. Suivre l'objet dans le temps (tracking)

Le tracking consiste à suivre la boîte d'un objet d'une image à la suivante. Pour un notebook
pédagogique, on fabrique une mini-séquence à partir de la même image et on lance un tracker
OpenCV.
"""))

cells.append(code('''\
def sequence_depuis_image(image, nb_images=5):
    base = np.array(image)
    frames = []
    for i in range(nb_images):
        canvas = np.full_like(base, 245)
        decalage = 18 * i
        h, w = base.shape[:2]
        y0 = min(20 + decalage, max(0, h - 20))
        x0 = min(20 + decalage, max(0, w - 20))
        x1 = min(x0 + w - 80, w)
        y1 = min(y0 + h - 80, h)
        crop = base[:y1 - y0, :x1 - x0]
        canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
        frames.append(canvas)
    return frames

frames = sequence_depuis_image(image)
box = prediction["box"]
init_box = (
    int(box["xmin"]),
    int(box["ymin"]),
    int(box["xmax"] - box["xmin"]),
    int(box["ymax"] - box["ymin"]),
)

tracker = cv2.TrackerCSRT_create() if hasattr(cv2, "TrackerCSRT_create") else cv2.legacy.TrackerCSRT_create()
# On initialise le tracker sur l'image d'origine (où `init_box` est exacte), puis on le
# laisse suivre l'objet à mesure qu'il se déplace dans la séquence fabriquée.
tracker.init(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), init_box)

for i, frame in enumerate(frames, start=1):
    ok, boite = tracker.update(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Image {i} : suivi {ok} | boîte = {tuple(round(v, 1) for v in boite)}")
'''))

cells.append(md("""\
## 5. Détecter sur tout l'échantillon Kaggle

On applique la détection « a leaf » à chacune des photos chargées à l'étape 1, et on observe
la confiance obtenue selon la catégorie réelle de la photo.
"""))

cells.append(code('''\
for img, label in echantillon:
    resultat = detecter(img, "a leaf", seuil_alerte=1.0)  # seuil à 1.0 : pas d'avertissement répété
    score = resultat["score"] if resultat else 0.0
    print(f"Vrai : {label:10s} | confiance détection 'a leaf' = {score:.2f}")
'''))

cells.append(md("""\
---
# 🏋️ Exercices

Ces exercices ne visent pas à *écrire du code*, mais à **observer l'effet** du prompt donné
au détecteur.
"""))

cells.append(md("""\
### 🏋️ Exercice 1 — Le prompt engineering change la confiance

Comparez la confiance obtenue pour `"leaf"`, `"a leaf"` et `"a green plant leaf"` sur la même
image. La formulation du mot-clé change-t-elle vraiment le résultat ?
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 1
"""))

cells.append(code('''\
for texte in ["leaf", "a leaf", "a green plant leaf"]:
    resultat = detecter(image, texte, seuil_alerte=1.0)
    score = resultat["score"] if resultat else 0.0
    print(f"{texte:22s} → confiance = {score:.3f}")
'''))

cells.append(md("""\
### 🏋️ Exercice 2 — Comparer plusieurs mots-clés

Comparez les scores de confiance obtenus pour `"a leaf"`, `"a plant"` et `"a fruit"` sur la
même image.
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 2
"""))

cells.append(code('''\
for mot in ["a leaf", "a plant", "a fruit"]:
    resultat = detecter(image, mot, seuil_alerte=1.0)
    score = resultat["score"] if resultat else 0.0
    print(f"{mot:10s} → confiance = {score:.2f}")
'''))

cells.append(md("""\
## ✅ Récapitulatif

- La **détection ouverte (OV)** permet de chercher un objet avec un mot, sans réentraînement.
- La **formulation du mot-clé** (prompt engineering) change nettement la confiance obtenue.
- La **segmentation** transforme une boîte en masque de pixels, le **tracking** suit l'objet
  d'image en image.
- Un modèle **petit (~150M paramètres)** suffit pour ce trio, et reste rapide sur Colab.

**➡️ Notebook suivant : `05_LLM_quantization.ipynb`** — faire tourner un grand modèle de
langage (~9 Md paramètres) grâce à la quantification, en local ou via une API cloud (Groq).
"""))

save(cells, "../notebooks/04_OV_detection_ouverte.ipynb")
