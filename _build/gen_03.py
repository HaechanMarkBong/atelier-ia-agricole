from gen_common import md, code, save, CONFIG_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 03. VLM : analyser des images de plantes

Un **VLM** (*Vision-Language Model*) comprend à la fois les **images** et le **texte**.
On peut lui montrer une **photo de feuille** et lui demander de la **décrire** ou de
répondre à une **question** à son sujet.

**Usages agricoles :**
- 🍃 Décrire l'état d'une feuille (taches, jaunissement…).
- 🔎 Aider au **pré-diagnostic** d'une maladie à partir d'une photo.
- 🧺 Reconnaître un fruit, estimer une maturité, etc.

On utilise ici des modèles **gratuits** de **HuggingFace** (famille **BLIP**), légers et
capables de tourner sur CPU.
"""))

cells.append(code(CONFIG_CELL))

cells.append(code('''\
# Installation des bibliothèques de vision et de génération d'images
pip_install("transformers", "torch", "pillow", "diffusers", "accelerate", "safetensors")
import torch
from PIL import Image, ImageDraw
print("✅ torch", torch.__version__, "| GPU:", torch.cuda.is_available())
'''))

cells.append(md("""\
## 1. Obtenir une image de plante

On charge une photo de feuille depuis une URL. Si le téléchargement échoue (pas de réseau),
on **génère une image de secours** pour que le notebook fonctionne quand même.
"""))

cells.append(code('''\
import urllib.request, io

URLS_FEUILLE = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Leaf_curl_on_peach.jpg/960px-Leaf_curl_on_peach.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Jowar_leaf_infested_by_pest_or_disease.jpg/960px-Jowar_leaf_infested_by_pest_or_disease.jpg",
]
# Certains serveurs (ex. Wikimedia) exigent un "User-Agent".
ENTETES = {"User-Agent": "Mozilla/5.0 (AtelierIA-Agricole; educatif)"}

def image_de_secours():
    """Crée une feuille verte stylisée avec des taches brunes (si pas de réseau)."""
    img = Image.new("RGB", (320, 320), (235, 240, 220))
    d = ImageDraw.Draw(img)
    d.ellipse([60, 30, 260, 300], fill=(70, 150, 60))          # la feuille
    d.line([160, 40, 160, 290], fill=(40, 100, 35), width=4)    # la nervure
    for (x, y) in [(120, 110), (200, 160), (150, 220), (190, 90)]:
        d.ellipse([x, y, x + 26, y + 26], fill=(120, 80, 40))   # taches brunes
    return img

def charger_image(url=None):
    url = url or URLS_FEUILLE[0]
    try:
        req = urllib.request.Request(url, headers=ENTETES)
        with urllib.request.urlopen(req, timeout=20) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
        print("✅ Image téléchargée :", url)
        return img
    except Exception as e:
        print("⚠️ Téléchargement impossible (", e, ") → image de secours générée.")
        return image_de_secours()

image = charger_image()
image.thumbnail((320, 320))
image
'''))

cells.append(md("""\
## 2. Décrire l'image (image → texte)

On utilise le modèle **BLIP**. Pour un modèle de vision il faut **deux objets** :
- le **processeur** (`processor`) : il transforme l'image (et le texte) en nombres ;
- le **modèle** (`model`) : il génère la réponse.

On définit deux petites fonctions pratiques — `decrire_image(...)` et `repondre_question(...)` —
et un **cache** pour ne télécharger chaque modèle qu'une seule fois.

> 🧩 On charge les modèles via leurs **classes** (`BlipForConditionalGeneration`, …) plutôt
> que par un *pipeline* : c'est **stable** d'une version de `transformers` à l'autre.
"""))

cells.append(code('''\
import gc
from transformers import (BlipProcessor, BlipForConditionalGeneration,
                          BlipForQuestionAnswering)

MODELE_LEGENDE = "Salesforce/blip-image-captioning-base"   # description d'image
MODELE_VQA     = "Salesforce/blip-vqa-base"                # questions/réponses visuelles

_modeles = {}   # cache : {nom: (processeur, modele)}

def _charger(nom, classe):
    if nom not in _modeles:
        print(f"Chargement de {nom} (1re fois : téléchargement)...")
        proc = BlipProcessor.from_pretrained(nom)
        mdl = classe.from_pretrained(nom)
        _modeles[nom] = (proc, mdl)
    return _modeles[nom]

def decrire_image(image, modele=MODELE_LEGENDE):
    """Renvoie une légende (description) de l'image."""
    proc, mdl = _charger(modele, BlipForConditionalGeneration)
    entrees = proc(image, return_tensors="pt")
    sortie = mdl.generate(**entrees, max_new_tokens=40)
    return proc.decode(sortie[0], skip_special_tokens=True)

def repondre_question(image, question, modele=MODELE_VQA):
    """Répond à une question (en anglais) au sujet de l'image."""
    proc, mdl = _charger(modele, BlipForQuestionAnswering)
    entrees = proc(image, question, return_tensors="pt")
    sortie = mdl.generate(**entrees, max_new_tokens=20)
    return proc.decode(sortie[0], skip_special_tokens=True)

print("🖼️ Description générée :", decrire_image(image))
'''))

cells.append(md("""\
## 3. Poser une question sur l'image (VQA)

Avec un modèle de **Visual Question Answering**, on peut **poser une question** précise
sur l'image, par exemple « *La feuille est-elle saine ?* ».
"""))

cells.append(code('''\
for question in ["What color is the leaf?", "Are there spots on the leaf?", "Is the plant healthy?"]:
    print(f"Q: {question:35s} → R: {repondre_question(image, question)}")
'''))

cells.append(md("""\
## 4. Générer une image à partir d'un texte

Cette partie montre l'autre sens du travail avec l'IA visuelle : **texte → image**.
Pour garder le notebook simple et rapide, on utilise ici un **petit pipeline de démonstration**.

> 💡 Le but est de comprendre le principe. Pour un usage réel, on choisira plus tard un modèle
> de génération plus performant.
"""))

cells.append(code('''\
from diffusers import StableDiffusionPipeline

MODELE_GENERATION = "hf-internal-testing/tiny-stable-diffusion-pipe"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Chargement du générateur d'image...")
pipe = StableDiffusionPipeline.from_pretrained(MODELE_GENERATION)
pipe = pipe.to(device)
if device == "cpu":
    pipe.enable_attention_slicing()

def generer_image(prompt, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    resultat = pipe(prompt, num_inference_steps=20, guidance_scale=7.0, generator=g)
    return resultat.images[0]

image_generee = generer_image("a healthy tomato plant in a sunny field, watercolor style")
image_generee
'''))

cells.append(md("""\
> 🌍 **Remarque langue.** Ces petits modèles VLM « pensent » surtout en **anglais**.
> Astuce : posez la question en anglais, puis **traduisez la réponse** avec un LLM (notebook 01) !
"""))

cells.append(md("""\
---
# 🏋️ Exercices
"""))

cells.append(md("""\
### 🏋️ Exercice 1 — Changer d'image

Chargez la **deuxième** URL de la liste `URLS_FEUILLE` (rouille du caféier) et
demandez une description au modèle.
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 1
"""))

cells.append(code('''\
image2 = charger_image(URLS_FEUILLE[1])
image2.thumbnail((320, 320))
print("Description :", decrire_image(image2))
image2
'''))

cells.append(md("""\
### 🏋️ Exercice 2 — Changer de modèle

Comparez le modèle de base avec une version **plus grande**,
**`Salesforce/blip-image-captioning-large`**, en passant l'argument `modele=...`
à `decrire_image`. (Lien avec le notebook SLM : plus gros = souvent meilleur, mais plus lourd !)

> 💡 En **mode démo**, on reste sur le petit modèle pour éviter un gros téléchargement.
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 2
"""))

cells.append(code('''\
autre_modele = MODELE_LEGENDE if MODE_DEMO else "Salesforce/blip-image-captioning-large"
print("Modèle de base :", decrire_image(image))
print("Autre modèle   :", decrire_image(image, modele=autre_modele))
'''))

cells.append(md("""\
### 🏋️ Exercice 3 — Changer la question (prompt visuel)

En utilisant `repondre_question(...)`, demandez au modèle **combien** de taches il voit,
puis **ce qui ne va pas** avec la plante. Observez ses réponses.
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 3
"""))

cells.append(code('''\
for q in ["How many spots are on the leaf?", "What is wrong with this plant?"]:
    print(q, "→", repondre_question(image, q))
'''))

cells.append(md("""\
## ✅ Récapitulatif

- Un **VLM** relie **image** et **texte** : description ou **questions/réponses** visuelles.
- On peut aussi faire l'opération inverse : **texte → image**.
- Avec HuggingFace, on charge un **processeur** + un **modèle** (classes `Blip…`), gratuitement.
- Très utile pour le **pré-diagnostic** à partir d'une photo — à confirmer toujours par un expert.

**➡️ Notebook suivant : `02_SLM_petits_modeles.ipynb`** — les petits modèles de langage pour le terrain.
"""))

save(cells, "../03_VLM_vision.ipynb")
