from gen_common import md, code, save, CONFIG_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 05. Détection ouverte, segmentation et tracking

Ce notebook montre trois briques visuelles utiles au champ :
- **bounding box** : repérer un objet avec un mot-clé ;
- **segmentation** : isoler l'objet dans l'image ;
- **tracking** : suivre cet objet sur une petite séquence d'images.

L'idée est simple : on part d'une image agricole, on cherche un objet avec un **texte**,
puis on affine le résultat pour aller vers une aide à l'analyse.
"""))

cells.append(code(CONFIG_CELL))

cells.append(code('''\
pip_install("transformers", "torch", "pillow", "opencv-contrib-python")
import io, urllib.request
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw
print("✅ torch", torch.__version__, "| GPU:", torch.cuda.is_available())
'''))

cells.append(md("""\
## 1. Charger une image agricole

On réutilise une photo simple de feuille. Si le réseau ne répond pas, le notebook crée
une image de secours pour rester exécutable.
"""))

cells.append(code('''\
URL_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Leaf_curl_on_peach.jpg/960px-Leaf_curl_on_peach.jpg"
ENTETES = {"User-Agent": "Mozilla/5.0 (AtelierIA-Agricole; educatif)"}

def image_de_secours():
    img = Image.new("RGB", (512, 384), (236, 240, 222))
    d = ImageDraw.Draw(img)
    d.ellipse([100, 40, 420, 340], fill=(74, 146, 70))
    d.line([260, 50, 260, 330], fill=(42, 96, 40), width=5)
    for box in [(180, 120, 220, 160), (300, 170, 340, 210), (230, 250, 270, 290)]:
        d.ellipse(box, fill=(135, 90, 45))
    return img

def charger_image(url=URL_IMAGE):
    try:
        req = urllib.request.Request(url, headers=ENTETES)
        with urllib.request.urlopen(req, timeout=20) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception as e:
        print("⚠️ Image de secours utilisée :", e)
        return image_de_secours()

image = charger_image()
image
'''))

cells.append(md("""\
## 2. Trouver un objet avec du texte

On utilise un modèle de **détection ouverte** : on donne un mot, et le modèle cherche
dans l'image l'objet correspondant.
"""))

cells.append(code('''\
from transformers import pipeline

MODELE_DETECTION = "google/owlvit-base-patch32"
detecteur = pipeline("zero-shot-object-detection", model=MODELE_DETECTION)

def detecter(image, texte_cible):
    resultats = detecteur(image, candidate_labels=[texte_cible])
    return resultats[0] if resultats else None

prediction = detecter(image, "leaf")
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
comme point de départ et on applique `grabCut`.
"""))

cells.append(code('''\
def segmenter(image, detection):
    arr = np.array(image)
    mask = np.zeros(arr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    box = detection["box"]
    rect = (
        int(box["xmin"]),
        int(box["ymin"]),
        int(box["xmax"] - box["xmin"]),
        int(box["ymax"] - box["ymin"]),
    )
    cv2.grabCut(arr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    masque = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    retour = arr * masque[:, :, np.newaxis]
    return Image.fromarray(retour)

image_segmente = segmenter(image, prediction)
image_segmente
'''))

cells.append(md("""\
## 4. Suivre l'objet dans le temps

Le tracking consiste à suivre la boîte d'un objet d'une image à la suivante.
Pour un notebook pédagogique, on fabrique une mini-séquence et on lance un tracker OpenCV.
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
        crop = base[:y1-y0, :x1-x0]
        canvas[y0:y0+crop.shape[0], x0:x0+crop.shape[1]] = crop
        frames.append(canvas)
    return frames

frames = sequence_depuis_image(image)
premiere = frames[0]
box = prediction["box"]
init_box = (
    int(box["xmin"]),
    int(box["ymin"]),
    int(box["xmax"] - box["xmin"]),
    int(box["ymax"] - box["ymin"]),
)

if hasattr(cv2, "TrackerCSRT_create"):
    tracker = cv2.TrackerCSRT_create()
else:
    tracker = cv2.legacy.TrackerCSRT_create()
tracker.init(cv2.cvtColor(premiere, cv2.COLOR_RGB2BGR), init_box)

suivi = []
for frame in frames:
    ok, boite = tracker.update(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    suivi.append((ok, boite))

for i, (ok, boite) in enumerate(suivi, start=1):
    print(f"Image {i} : suivi {ok} | boîte = {tuple(round(v, 1) for v in boite)}")
'''))

cells.append(md("""\
## ✅ Récapitulatif

- La **détection ouverte** permet de chercher un objet avec un mot ou une phrase.
- La **segmentation** transforme une boîte en masque de pixels.
- Le **tracking** suit l'objet d'image en image.

Ce trio aide à passer de la simple photo à une analyse plus précise sur le terrain.
"""))

save(cells, "../05_detection_ouverte.ipynb")