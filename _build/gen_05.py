"""05_OV_video_tracking.ipynb — Prompt -> boîte, segmentation et suivi dans une VIDÉO.

Modèles (100 % Hugging Face, aucun service payant, aucune boîte dessinée à la main):
  - Grounding DINO (détection ouverte): un TEXTE -> des boîtes englobantes.
  - SAM 2 vidéo (Segment Anything 2): les boîtes -> masques de segmentation
    PROPAGÉS et SUIVIS (tracking) d'une image à l'autre de la vidéo, avec un
    identifiant stable par cible.
Ensemble = « Grounded SAM 2 »: on écrit un mot, le modèle repère, découpe et suit
la cible dans toute la vidéo — automatiquement.
"""
from gen_common import md, code, save, CONFIG_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 05. Prompt → boîte, segmentation & suivi dans une VIDÉO

Les notebooks précédents travaillaient sur du **texte** ou une **image fixe**. Ici on passe à la
**vidéo**: on écrit un **mot** (le *prompt*) et deux modèles font le reste, **tout seuls** —
on ne dessine aucune boîte à la main.

1. **Détection ouverte** — un texte (« cow », « a cow »…) → des **boîtes englobantes**
   (*bounding boxes*). Modèle: **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`).
2. **Segmentation + suivi** — chaque boîte → un **masque** de pixels, **propagé et suivi**
   (*tracking*) sur toutes les images, avec un **identifiant stable** par cible.
   Modèle: **SAM 2** vidéo (`facebook/sam2.1-hiera-tiny`).

Cette combinaison s'appelle **« Grounded SAM 2 »**. Tout est produit **automatiquement** par les
modèles à partir du **prompt**: le seul geste humain est d'écrire le mot recherché.

> 🐄 **Cas agricole**: repérer, isoler et **suivre le bétail** dans une vidéo de pâturage
> (comptage de troupeau, surveillance, analyse de comportement). On utilise une courte vidéo
> libre de droits (Pexels) montrant des vaches qui broutent.
"""))

cells.append(code(CONFIG_CELL))

cells.append(code('''\
# SAM 2 vidéo est arrivé dans transformers >= 4.57 -> on force la mise à niveau si besoin.
pip_install("transformers>=4.57", "accelerate", "opencv-contrib-python", "pillow")
# SAM 2 s'appuie sur torchvision (déjà présent sur Colab). On ne l'installe que s'il manque,
# pour ne pas risquer de changer la version de torch.
try:
    import torchvision  # noqa: F401
except Exception:
    pip_install("torchvision")

import numpy as np, cv2, torch
from PIL import Image, ImageDraw
print("✅ torch", torch.__version__, "| opencv", cv2.__version__)

# GPU si disponible (Colab T4), sinon CPU. Les modèles choisis sont petits -> OK sur CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Appareil de calcul:", DEVICE)
'''))

cells.append(md("""\
## 1. Récupérer une courte vidéo de pâturage

On télécharge une vidéo **libre de droits** (Pexels). Si le réseau est indisponible, une petite
vidéo de secours (une forme qui se déplace) est fabriquée pour que le notebook aille **jusqu'au
bout** partout.
"""))

cells.append(code('''\
import urllib.request

URL_VIDEO = "https://www.pexels.com/download/video/8187697/"  # vaches au pâturage (Pexels)
CHEMIN_VIDEO = "vaches.mp4"

def telecharger_video(url, chemin):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(chemin, "wb") as f:
            f.write(r.read())
        print(f"✅ Vidéo téléchargée: {chemin}")
        return True
    except Exception as e:
        print(f"⚠️ Téléchargement impossible ({e}) → vidéo de secours synthétique.")
        return False

def video_de_secours(chemin, n=40, taille=(640, 360)):
    """Fabrique une vidéo simple (disque rouge qui traverse un pré vert) si Pexels est injoignable."""
    out = cv2.VideoWriter(chemin, cv2.VideoWriter_fourcc(*"mp4v"), 25, taille)
    w, h = taille
    for i in range(n):
        img = np.full((h, w, 3), (60, 140, 70), np.uint8)   # fond vert (BGR)
        x = int(40 + (w - 80) * i / n)
        cv2.circle(img, (x, h // 2), 40, (40, 40, 200), -1)  # disque rouge
        out.write(img)
    out.release()

if not telecharger_video(URL_VIDEO, CHEMIN_VIDEO):
    video_de_secours(CHEMIN_VIDEO)
'''))

cells.append(md("""\
## 2. Échantillonner quelques images de la vidéo

Une vidéo 4K de 20 s, c'est ~500 images: beaucoup trop pour un CPU. On garde **quelques images
réparties** dans le temps et on les **réduit** (largeur fixe). En mode démo (sans GPU), on en
prend encore moins pour rester rapide.
"""))

cells.append(code('''\
N_IMAGES = 8 if MODE_DEMO else 24      # nombre d'images échantillonnées dans la vidéo
LARGEUR = 480 if MODE_DEMO else 640    # largeur cible (les images sont redimensionnées)

def echantillonner_video(chemin, n, largeur):
    cap = cv2.VideoCapture(chemin)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, max(0, total - 1), n).astype(int)
    images = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w = f.shape[:2]
        images.append(cv2.resize(f, (largeur, int(round(h * largeur / w)))))
    cap.release()
    return np.stack(images)

frames = echantillonner_video(CHEMIN_VIDEO, N_IMAGES, LARGEUR)
print(f"{len(frames)} images de {frames.shape[2]}×{frames.shape[1]} px prêtes.")
Image.fromarray(frames[0])   # aperçu de la 1re image
'''))

cells.append(md("""\
## 3. Le prompt → des boîtes (détection ouverte)

On donne un **mot** au détecteur **Grounding DINO**. Il renvoie les **boîtes** des objets
correspondants sur la **première image**, sans avoir été entraîné sur ce mot précis. Convention
du modèle: un prompt en minuscules terminé par un point (`"cow."`); on peut chaîner plusieurs
cibles (`"cow. sheep."`). On se contente ici de **lire** les boîtes trouvées (coordonnées +
confiance): l'affichage visuel viendra plus loin, une fois le suivi calculé.
"""))

cells.append(code('''\
from transformers import AutoProcessor, GroundingDinoForObjectDetection

ID_DETECTEUR = "IDEA-Research/grounding-dino-tiny"
proc_dino = AutoProcessor.from_pretrained(ID_DETECTEUR)
modele_dino = GroundingDinoForObjectDetection.from_pretrained(ID_DETECTEUR).to(DEVICE)

def detecter(image_rgb, prompt, seuil=0.35, max_objets=4):
    """image_rgb: np.uint8 HxWx3. Renvoie une liste de boîtes [x0,y0,x1,y1] (au plus max_objets)."""
    image = Image.fromarray(image_rgb)
    entrees = proc_dino(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        sorties = modele_dino(**entrees)
    res = proc_dino.post_process_grounded_object_detection(
        sorties, entrees.input_ids, threshold=seuil, text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )[0]
    boites = [[float(v) for v in b] for b in res["boxes"][:max_objets]]
    scores = [round(float(s), 2) for s in res["scores"][:max_objets]]
    print(f"Prompt « {prompt} » → {len(boites)} boîte(s) ; confiance {scores}")
    return boites

PROMPT = "cow."
boites = detecter(frames[0], PROMPT)

if not boites:
    # Filet de sécurité (ex.: vidéo de secours) : on sème une boîte centrale pour que la
    # démonstration de suivi fonctionne quand même. Le vrai cas d'usage reste le prompt ci-dessus.
    h, w = frames[0].shape[:2]
    boites = [[w * 0.35, h * 0.35, w * 0.65, h * 0.65]]
    print("⚠️ Aucune détection → boîte centrale de secours pour illustrer le suivi.")
'''))

cells.append(md("""\
## 4. Segmenter **et suivre** chaque cible dans la vidéo (SAM 2)

On donne les boîtes de la 1re image à **SAM 2 vidéo** (une boîte = une cible, avec un identifiant
`#1`, `#2`…). Il en déduit un **masque** par cible, puis le **propage** sur toutes les images en
gardant le **même identifiant**: c'est le **suivi** (tracking). Une seule intervention (les boîtes
sur l'image 0), et le modèle fait le reste sur toute la séquence.
"""))

cells.append(code('''\
from transformers import Sam2VideoModel, Sam2VideoProcessor

ID_SAM = "facebook/sam2.1-hiera-tiny"
proc_sam = Sam2VideoProcessor.from_pretrained(ID_SAM)
modele_sam = Sam2VideoModel.from_pretrained(ID_SAM).to(DEVICE).eval()

# 1) ouvrir une "session vidéo" contenant toutes les images
session = proc_sam.init_video_session(video=frames, inference_device=DEVICE, dtype=torch.float32)

# 2) déclarer les cibles sur l'image 0 : une boîte = un objet (identifiants 1..N)
ids_objets = list(range(1, len(boites) + 1))
proc_sam.add_inputs_to_inference_session(
    inference_session=session, frame_idx=0, obj_ids=list(ids_objets),  # copie: SAM 2 vide la liste au fil du suivi
    input_boxes=[boites],   # format [batch][objet][x0,y0,x1,y1]
)

# 3) propager : SAM 2 segmente et suit chaque cible, image par image
masques_par_image = {}
with torch.no_grad():
    for sortie in modele_sam.propagate_in_video_iterator(session, start_frame_idx=0):
        masks = proc_sam.post_process_masks([sortie.pred_masks], [frames.shape[1:3]], binarize=True)[0]
        masques_par_image[sortie.frame_idx] = (sortie.object_ids, masks)
print(f"✅ {len(ids_objets)} cible(s) suivie(s) sur {len(masques_par_image)} images.")
'''))

cells.append(md("""\
## 5. Visualiser le résultat: masques + boîtes + identifiants qui **suivent**

Tout ce qui s'affiche ci-dessous est **produit par les modèles**, pas dessiné à la main:
pour chaque image on superpose le **masque** (couleur = identifiant de la cible), on trace la
**boîte englobante déduite du masque** (elle aussi automatique, et qui suit la cible), et on écrit
l'**identifiant**. On assemble le tout en un **GIF animé** pour voir le suivi.
"""))

cells.append(code('''\
# Couleur stable par identifiant : la même cible garde sa couleur d'un bout à l'autre = suivi.
PALETTE = [(230,60,60),(60,140,230),(60,200,120),(240,180,40),(180,90,220),(240,120,60),(30,200,200)]
def couleur(oid):
    return PALETTE[(oid - 1) % len(PALETTE)]

def masque_2d(m):
    return m.squeeze().cpu().numpy().astype(bool)

def annoter(image_rgb, ids, masks):
    """Superpose masques colorés + boîte (déduite du masque) + identifiant. Renvoie une image PIL."""
    base = image_rgb.astype(np.float32)
    overlay = base.copy()
    for oid, m in zip(ids, masks):
        mm = masque_2d(m)
        if mm.any():
            overlay[mm] = 0.45 * base[mm] + 0.55 * np.array(couleur(oid), np.float32)  # teinte le masque
    img = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for oid, m in zip(ids, masks):
        mm = masque_2d(m)
        if not mm.any():
            continue
        ys, xs = np.where(mm)                                   # boîte englobante = extrêmes du masque
        d.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    outline=couleur(oid), width=2)
        d.text((int(xs.min()) + 2, max(0, int(ys.min()) - 12)), f"#{oid}", fill=couleur(oid))
    return img

images_annotees = [annoter(frames[i], *masques_par_image[i]) for i in sorted(masques_par_image)]
print(f"{len(images_annotees)} images annotées.")
images_annotees[len(images_annotees) // 2]   # aperçu d'une image du milieu
'''))

cells.append(code('''\
from IPython.display import Image as ImageAffichee

def faire_gif(images, chemin="suivi.gif", ms=350):
    images[0].save(chemin, save_all=True, append_images=images[1:], duration=ms, loop=0)
    return chemin

chemin_gif = faire_gif(images_annotees)
print(f"GIF: {chemin_gif} — chaque cible garde sa couleur/identifiant = suivi réussi.")
ImageAffichee(filename=chemin_gif)
'''))

cells.append(md("""\
## 6. Application: compter le bétail suivi

Le suivi attribue un **identifiant unique** par animal. Compter les identifiants effectivement
vus (masque non vide) donne une estimation du **nombre d'animaux** repérés à partir du seul mot
« cow » — utile pour un comptage de troupeau.
"""))

cells.append(code('''\
ids_vus = {oid for ids, masks in masques_par_image.values()
           for oid, m in zip(ids, masks) if masque_2d(m).any()}
print(f"🐄 {len(ids_vus)} cible(s) distincte(s) suivie(s) à partir du prompt « {PROMPT} » : "
      f"{sorted(ids_vus)}")
'''))

cells.append(md("""\
---
# 🏋️ Exercices

Ces exercices ne visent pas à *écrire du code*, mais à **observer l'effet du prompt** sur la
détection, la segmentation et le suivi.
"""))

cells.append(md("""\
### 🏋️ Exercice 1 — Changer la cible avec le prompt

Relancez la détection sur `frames[0]` avec un autre mot (`"grass."`, `"animal."`, `"field."`).
Combien de boîtes obtenez-vous, et avec quelle confiance? Le prompt suffit à changer de cible,
sans réentraînement.
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 1
"""))

cells.append(code('''\
for mot in ["cow.", "grass.", "animal.", "field."]:
    _ = detecter(frames[0], mot)
'''))

cells.append(md("""\
### 🏋️ Exercice 2 — Le seuil de confiance filtre les cibles

Rappelez `detecter(frames[0], "cow.", seuil=...)` avec un seuil bas (`0.3`) puis élevé (`0.7`).
Un seuil élevé garde-t-il moins de boîtes? Comment cela changerait-il le nombre d'animaux suivis?
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 2
"""))

cells.append(code('''\
# max_objets élevé pour que le plafond de 4 ne masque pas l'effet du seuil.
for s in [0.3, 0.5, 0.7]:
    b = detecter(frames[0], "cow.", seuil=s, max_objets=10)
    print(f"  seuil={s} → {len(b)} boîte(s)")
'''))

cells.append(md("""\
## ✅ Récapitulatif

- **Grounding DINO** transforme un **mot** en **boîtes** (détection *ouverte*: pas de liste figée
  de catégories).
- **SAM 2 vidéo** transforme ces boîtes en **masques** et les **suit** image par image, avec un
  **identifiant stable** par cible (segmentation + tracking).
- Tout est **automatique** à partir du **prompt**: on ne dessine aucune boîte à la main; les seules
  boîtes affichées sont **déduites des masques** produits par le modèle.
- Des modèles **petits** (`grounding-dino-tiny`, `sam2.1-hiera-tiny`) suffisent et tournent même
  sur le CPU gratuit de Colab; un GPU T4 accélère et permet plus d'images / de plus gros modèles
  (`sam2.1-hiera-large`).

**🎉 Fin de l'atelier.** Vous avez parcouru: **SLM** (01), **quantification LLM** (02),
**TinyLLM/TinyVLM** (03), **VLM** (04) et enfin la **vision ouverte sur vidéo** (05).
"""))

save(cells, "../notebooks/05_OV_video_tracking.ipynb")
