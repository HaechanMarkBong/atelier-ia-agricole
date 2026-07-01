from gen_common import md, code, save, CONFIG_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 00. Introduction & Installation

Bienvenue dans cet atelier d'introduction à l'**intelligence artificielle pour l'agriculture** !

Ce premier notebook pose les bases :
1. Comprendre les familles de modèles : **LLM**, **VLM**, **SLM**, **tinyML** et détection ouverte.
2. Préparer l'environnement Google Colab.
3. Installer **Ollama** (pour exécuter des modèles de langage gratuitement).
4. Installer les bibliothèques **HuggingFace**.
5. Configurer l'accès aux **jeux de données Kaggle**.

> 💡 **100 % gratuit.** Tous les outils utilisés ici sont libres d'accès.
> Aucun abonnement payant n'est nécessaire.

## 👉 Comment utiliser ce notebook
- Exécutez les cellules **dans l'ordre**, de haut en bas (`Maj + Entrée`).
- Une cellule grise = du code à exécuter. Le texte = des explications.
- Sur Colab : `Exécution` → `Modifier le type d'exécution` → choisissez **GPU** (T4) si disponible (gratuit).
"""))

cells.append(md("""\
## 1. Les familles de modèles en un coup d'œil

| Famille | Nom complet | Entrée → Sortie | Taille typique | Exemple agricole |
|--------|-------------|------------------|----------------|------------------|
| **LLM** | *Large Language Model* | texte → texte | 7–70 milliards de paramètres | Répondre à une question d'agronomie |
| **VLM** | *Vision-Language Model* | image (+texte) → texte / image | 1–10 milliards | Décrire une feuille malade ou générer une image |
| **SLM** | *Small Language Model* | texte → texte | 0,5–3 milliards | Même chose, mais sur un téléphone / hors-ligne |
| **tinyML** | *Tiny Machine Learning* | capteurs/tableaux → prédiction | < 1 Mo (!) | Recommander une culture selon le sol, sur un microcontrôleur |
| **Détection ouverte** | boîte / masque / suivi | image → boîte, masque, trajectoire | modèles légers + OpenCV | Repérer, segmenter et suivre un objet |

**Idée clé :** plus le modèle est gros, plus il est « intelligent »… mais plus il est lent,
coûteux et gourmand en énergie. En agriculture, on travaille souvent **au champ**, sans bonne
connexion : les **petits** modèles (SLM, tinyML) deviennent alors très intéressants.
"""))

cells.append(code(CONFIG_CELL))

cells.append(md("""\
## 2. Vérifier le matériel (GPU)

Le GPU accélère beaucoup les modèles. Sur Colab il est gratuit mais optionnel.
La cellule suivante fonctionne **avec ou sans** GPU.
"""))

cells.append(code('''\
# Vérification du GPU (ne provoque pas d'erreur s'il n'y en a pas)
gpu_dispo = False
try:
    sortie = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                             "--format=csv,noheader"],
                            capture_output=True, text=True)
    if sortie.returncode == 0 and sortie.stdout.strip():
        gpu_dispo = True
        print("✅ GPU détecté :", sortie.stdout.strip())
    else:
        print("ℹ️ Pas de GPU détecté — tout fonctionnera sur le processeur (CPU), en plus lent.")
except FileNotFoundError:
    print("ℹ️ Pas de GPU détecté — tout fonctionnera sur le processeur (CPU), en plus lent.")
'''))

cells.append(md("""\
## 3. Installer Ollama

[**Ollama**](https://ollama.com) permet de télécharger et d'exécuter des modèles de langage
(LLM/SLM) **localement et gratuitement**, avec une seule commande.

Sur Colab, on l'installe avec le script officiel. (En local, s'il est déjà installé, cette
cellule ne casse rien.)
"""))

cells.append(code('''\
import shutil

if shutil.which("ollama") is None:
    if IS_COLAB:
        print("Installation d'Ollama sur Colab...")
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                       shell=True, check=False)
    else:
        print("Ollama n'est pas dans le PATH. En local, installez-le depuis https://ollama.com/download")
else:
    print("✅ Ollama est déjà disponible :", shutil.which("ollama"))
'''))

cells.append(md("""\
### Démarrer le serveur Ollama

Ollama fonctionne comme un petit **serveur** en arrière-plan. On le lance une fois,
puis tous les notebooks pourront lui parler via `http://localhost:11434`.
"""))

cells.append(code('''\
import time, urllib.request

def demarrer_ollama():
    """Démarre le serveur Ollama en arrière-plan s'il ne tourne pas déjà."""
    def serveur_actif():
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False

    if serveur_actif():
        print("✅ Le serveur Ollama tourne déjà.")
        return
    if shutil.which("ollama") is None:
        print("⚠️ Ollama n'est pas installé — étape précédente à vérifier.")
        return
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        if serveur_actif():
            print("✅ Serveur Ollama démarré.")
            return
        time.sleep(1)
    print("⚠️ Le serveur n'a pas répondu à temps (réessayez la cellule).")

demarrer_ollama()
'''))

cells.append(md("""\
### Télécharger et tester un premier modèle

On télécharge un **petit** modèle pour aller vite : `qwen2.5:0.5b` (~400 Mo).
La commande `ollama pull` télécharge le modèle ; on l'interroge ensuite.
"""))

cells.append(code('''\
import json

MODELE_TEST = "qwen2.5:0.5b"   # petit modèle, rapide à télécharger

print(f"Téléchargement de {MODELE_TEST} (peut prendre 1-2 min la première fois)...")
subprocess.run(["ollama", "pull", MODELE_TEST], check=False)

def demander_ollama(modele, prompt):
    """Envoie un prompt au modèle et renvoie la réponse texte."""
    donnees = json.dumps({"model": modele, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=donnees,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]

reponse = demander_ollama(MODELE_TEST, "Donne un conseil court pour cultiver des tomates.")
print("\\n🤖 Réponse du modèle :\\n", reponse)
'''))

cells.append(md("""\
## 4. Installer les bibliothèques HuggingFace

[**HuggingFace**](https://huggingface.co) héberge des milliers de modèles gratuits.
On l'utilisera surtout pour les **VLM** (vision) dans le notebook 03.
"""))

cells.append(code('''\
# transformers = la bibliothèque de modèles ; datasets = pour charger des jeux de données
pip_install("transformers", "datasets", "pillow")
print("✅ Bibliothèques HuggingFace prêtes.")
'''))

cells.append(md("""\
## 5. Configurer Kaggle (jeux de données gratuits)

[**Kaggle**](https://www.kaggle.com) propose des milliers de jeux de données agricoles gratuits.
Pour les télécharger par programme, il faut un **jeton d'API** :

1. Créez un compte gratuit sur kaggle.com.
2. Allez dans **Settings** (profil) → section **API** → **Create New Token**.
3. Un fichier `kaggle.json` est téléchargé.
4. Téléversez-le ci-dessous (sur Colab) ou placez-le dans `~/.kaggle/kaggle.json`.

> 💡 **Pas de jeton ?** Pas de panique : nos notebooks proposent **toujours** une
> source de secours **sans authentification**, donc tout fonctionne même sans compte Kaggle.
"""))

cells.append(code('''\
import os

def configurer_kaggle():
    """Installe le jeton Kaggle. Sur Colab, propose le téléversement de kaggle.json."""
    cible = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(cible):
        print("✅ Jeton Kaggle déjà configuré.")
        return True
    if IS_COLAB and not MODE_DEMO:
        try:
            from google.colab import files
            print("Sélectionnez votre fichier kaggle.json :")
            televerse = files.upload()
            if "kaggle.json" in televerse:
                os.makedirs(os.path.dirname(cible), exist_ok=True)
                with open(cible, "wb") as f:
                    f.write(televerse["kaggle.json"])
                os.chmod(cible, 0o600)
                print("✅ Jeton Kaggle installé.")
                return True
        except Exception as e:
            print("Téléversement ignoré :", e)
    print("ℹ️ Pas de jeton Kaggle — les notebooks utiliseront la source de secours.")
    return False

kaggle_ok = configurer_kaggle()
'''))

cells.append(md("""\
## ✅ Récapitulatif

Vous avez :
- compris les familles **LLM / VLM / SLM / tinyML** et la détection ouverte ;
- installé et testé **Ollama** avec un petit modèle ;
- installé les bibliothèques **HuggingFace** ;
- (optionnellement) configuré **Kaggle**.

**➡️ Passez maintenant au notebook `01_LLM_ollama.ipynb`.**
"""))

save(cells, "../00_introduction_et_installation.ipynb")
