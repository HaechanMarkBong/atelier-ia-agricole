"""Helpers partagés pour générer les notebooks de l'atelier (nbformat v4)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


def save(cells, path):
    nb = new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"provenance": []},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("écrit:", path)


# Cellule de configuration commune insérée en haut de chaque notebook.
# Détecte Colab, expose MODE_DEMO (via variable d'env pour le test automatique)
# et un utilitaire d'installation de paquets silencieux.
CONFIG_CELL = '''\
# === Configuration de l'environnement (exécuter en premier) ===
import os, sys, subprocess

# Sommes-nous dans Google Colab ?
try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# MODE_DEMO = True  -> utilise de tout petits modèles / jeux de données réduits
# (utile pour tester rapidement, ou hors GPU). Mettez False pour l'atelier complet.
# La variable d'environnement ATELIER_DEMO permet de forcer le mode pour les tests.
MODE_DEMO = os.environ.get("ATELIER_DEMO", "0") == "1"

def pip_install(*paquets):
    """Installe des paquets pip de façon silencieuse (Colab ou local)."""
    cmd = [sys.executable, "-m", "pip", "install", "-q", *paquets]
    print("Installation :", " ".join(paquets), "...")
    subprocess.run(cmd, check=False)

print(f"IS_COLAB = {IS_COLAB}")
print(f"MODE_DEMO = {MODE_DEMO}")
print("Python :", sys.version.split()[0])
'''

# Cellule partagée pour charger un jeu de données Kaggle public.
# kagglehub télécharge les jeux de données PUBLICS sans clé API. Si Kaggle est
# injoignable (réseau, quota...), chaque notebook bascule sur un petit échantillon
# de secours codé en dur, pour rester exécutable partout.
KAGGLE_CELL = '''\
pip_install("kagglehub", "pandas")
import kagglehub

def telecharger_dataset_kaggle(reference):
    """Télécharge un jeu de données Kaggle public et renvoie son dossier local.
    Renvoie None si Kaggle est injoignable (un échantillon de secours prend alors le relais)."""
    try:
        dossier = kagglehub.dataset_download(reference)
        print(f"✅ Jeu de données Kaggle prêt : {reference}")
        return dossier
    except Exception as e:
        print(f"⚠️ Kaggle indisponible ({e}) → utilisation d'un échantillon de secours.")
        return None
'''

# Cellule partagée pour échantillonner des photos de plantes malades/saines depuis le jeu
# Kaggle "Plant Disease Recognition" (Healthy / Powdery / Rust). Utilisée par les notebooks
# de vision (VLM, TinyVLM, OV). Retombe sur quelques URLs publiques si Kaggle est injoignable.
IMAGES_CELL = '''\
pip_install("pillow")
import os, random, io, urllib.request
from PIL import Image

REFERENCE_IMAGES_KAGGLE = "rashikrahmanpritom/plant-disease-recognition-dataset"
URLS_SECOURS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Leaf_curl_on_peach.jpg/960px-Leaf_curl_on_peach.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Jowar_leaf_infested_by_pest_or_disease.jpg/960px-Jowar_leaf_infested_by_pest_or_disease.jpg",
]

def echantillon_images_plantes(n=10):
    """Renvoie une liste de (image PIL, label) : n photos du jeu Kaggle Healthy/Powdery/Rust ;
    à défaut, quelques URLs publiques ; et en dernier recours une image synthétique hors-ligne.
    Ainsi la liste renvoyée n'est jamais vide, même sans réseau."""
    dossier = telecharger_dataset_kaggle(REFERENCE_IMAGES_KAGGLE)
    if dossier:
        try:
            base = os.path.join(dossier, "Test", "Test")
            fichiers = []
            for classe in sorted(os.listdir(base)):
                dossier_c = os.path.join(base, classe)
                if not os.path.isdir(dossier_c):
                    continue
                for nom in sorted(os.listdir(dossier_c))[:20]:
                    fichiers.append((os.path.join(dossier_c, nom), classe))
            random.Random(42).shuffle(fichiers)
            images = [(Image.open(chemin).convert("RGB"), label) for chemin, label in fichiers[:n]]
            if images:
                return images
            print("⚠️ Jeu Kaggle vide/inattendu → images de secours.")
        except Exception as e:
            print(f"⚠️ Lecture du jeu Kaggle impossible ({e}) → images de secours.")

    entetes = {"User-Agent": "Mozilla/5.0 (AtelierIA-Agricole; educatif)"}
    images = []
    for url in URLS_SECOURS[:n]:
        try:
            req = urllib.request.Request(url, headers=entetes)
            with urllib.request.urlopen(req, timeout=20) as r:
                images.append((Image.open(io.BytesIO(r.read())).convert("RGB"), "inconnu"))
        except Exception as e:
            print("⚠️ Image ignorée :", e)

    # Dernier recours 100% hors-ligne : si Kaggle ET les URLs échouent, on fabrique au moins
    # une image synthétique pour que le notebook aille jusqu'au bout (pas d'IndexError sur
    # echantillon[0], liste jamais vide).
    if not images:
        from PIL import ImageDraw
        print("⚠️ Aucune image en ligne → image de secours générée localement.")
        for _ in range(max(1, min(n, 3))):
            img = Image.new("RGB", (256, 256), (150, 170, 90))
            ImageDraw.Draw(img).ellipse([40, 40, 216, 216], fill=(70, 110, 50))
            images.append((img, "inconnu"))
    return images
'''
