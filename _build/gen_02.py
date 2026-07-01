from gen_common import md, code, save, CONFIG_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 02. SLM : les petits modèles de langage

Un **SLM** (*Small Language Model*) est un modèle de langage **compact** (0,5 à ~3 milliards
de paramètres) au lieu de dizaines de milliards.

**Pourquoi est-ce crucial en agriculture ?**
- 📶 Fonctionne **hors-ligne** (utile au champ, sans Internet).
- 📱 Tourne sur du **matériel modeste** (téléphone, mini-PC, Raspberry Pi).
- ⚡ **Rapide** et **peu gourmand** en énergie.
- 🔒 Données qui **restent locales** (confidentialité).

Le compromis : un SLM est **moins savant** qu'un gros LLM. Dans ce notebook, on **compare**
plusieurs SLM en **taille**, **vitesse** et **qualité**.
"""))

cells.append(code(CONFIG_CELL))

cells.append(code('''\
import os, subprocess, shutil, time, json, urllib.request

try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except Exception:
    IS_COLAB = False

def chemin_ollama():
    candidats = [
        shutil.which("ollama"),
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        "/root/.local/bin/ollama",
    ]
    for candidat in candidats:
        if candidat and os.path.exists(candidat):
            return candidat
    return None

if chemin_ollama() is None and IS_COLAB:
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=False)
    time.sleep(2)

OLLAMA_CMD = chemin_ollama()

def serveur_actif():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False

if not serveur_actif():
    if OLLAMA_CMD:
        subprocess.Popen([OLLAMA_CMD, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            if serveur_actif():
                break
            time.sleep(1)
    else:
        print("⚠️ Ollama n'est pas disponible. En local, installez-le depuis https://ollama.com/download")

print("Serveur Ollama actif :", serveur_actif())
'''))

cells.append(md("""\
## 1. Choisir une liste de SLM à comparer

Quelques petits modèles populaires (taille approximative du téléchargement) :

| Modèle Ollama | Paramètres | Taille |
|---------------|-----------|--------|
| `qwen2.5:0.5b` | 0,5 Md | ~0,4 Go |
| `gemma2:2b`    | 2 Md    | ~1,6 Go |
| `phi3:mini`    | 3,8 Md  | ~2,3 Go |

En **mode démo**, on n'en teste qu'un (pour aller vite). En mode complet, on les compare tous.
"""))

cells.append(code('''\
if MODE_DEMO:
    MODELES_SLM = ["qwen2.5:0.5b"]
else:
    MODELES_SLM = ["qwen2.5:0.5b", "gemma2:2b", "phi3:mini"]

for m in MODELES_SLM:
    print("Téléchargement de", m, "...")
    subprocess.run(["ollama", "pull", m], check=False)
print("✅ Modèles prêts :", MODELES_SLM)
'''))

cells.append(md("""\
## 2. Mesurer taille, vitesse et qualité

On interroge chaque modèle avec la **même** question agricole, et on **chronomètre** la réponse.
"""))

cells.append(code('''\
def chat(modele, prompt, temperature=0.3):
    corps = {"model": modele, "prompt": prompt, "stream": False,
             "options": {"temperature": temperature}}
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=json.dumps(corps).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]

def taille_modele(modele):
    """Renvoie la taille du modèle en Go (via l'API Ollama)."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            for m in json.loads(r.read()).get("models", []):
                if m["name"].startswith(modele.split(":")[0]):
                    return round(m["size"] / 1e9, 2)
    except Exception:
        pass
    return None

question = "En 2 phrases, explique pourquoi la rotation des cultures améliore le sol."

for m in MODELES_SLM:
    debut = time.time()
    rep = chat(m, question)
    duree = time.time() - debut
    print(f"\\n=== {m}  (taille ≈ {taille_modele(m)} Go, {duree:.1f} s) ===")
    print(rep.strip())
'''))

cells.append(md("""\
## 3. Quand préférer un SLM ?

- ✅ **Au champ, hors-ligne**, sur téléphone ou mini-PC.
- ✅ Pour des tâches **simples et cadrées** (FAQ, classification, extraction).
- ❌ Évitez-les pour du **raisonnement complexe** ou des connaissances très pointues
  → là, un gros LLM reste meilleur.

> 💡 Astuce : un SLM **bien guidé** (bon prompt + contexte fourni) rivalise souvent avec
> un gros modèle sur des tâches précises.
"""))

cells.append(md("""\
---
# 🏋️ Exercices
"""))

cells.append(md("""\
### 🏋️ Exercice 1 — Ajouter un modèle à la comparaison

Ajoutez le modèle **`llama3.2:1b`** à la liste `MODELES_SLM`, téléchargez-le,
et relancez la comparaison de la section 2.
*(En mode démo, vous pouvez garder un seul modèle si le temps manque.)*
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 1
"""))

cells.append(code('''\
nouveau = "qwen2.5:0.5b" if MODE_DEMO else "llama3.2:1b"
if nouveau not in MODELES_SLM:
    MODELES_SLM.append(nouveau)
subprocess.run(["ollama", "pull", nouveau], check=False)

for m in MODELES_SLM:
    debut = time.time()
    rep = chat(m, question)
    print(f"\\n=== {m}  ({time.time()-debut:.1f} s) ===\\n{rep.strip()}")
'''))

cells.append(md("""\
### 🏋️ Exercice 2 — Tâche de classification

Les SLM sont parfaits pour **classer** du texte. Demandez à un SLM de classer le message
d'un agriculteur dans l'une des catégories : `maladie`, `irrigation`, `fertilisation`.
Imposez de répondre **par un seul mot**.

Message : *« Mes plants fanent alors que la terre est sèche depuis une semaine. »*
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 2
"""))

cells.append(code('''\
message = "Mes plants fanent alors que la terre est sèche depuis une semaine."
prompt = (f"Classe ce message agricole dans UNE seule catégorie parmi : "
          f"maladie, irrigation, fertilisation. "
          f"Réponds par UN seul mot.\\n\\nMessage : {message}\\nCatégorie :")
print(chat(MODELES_SLM[0], prompt, temperature=0.0))
'''))

cells.append(md("""\
## ✅ Récapitulatif

- Un **SLM** échange un peu de « savoir » contre **rapidité, taille réduite et fonctionnement hors-ligne**.
- On compare les modèles sur **taille / vitesse / qualité** selon le besoin.
- Idéal pour des tâches **cadrées** (classification, extraction) sur le terrain.

**➡️ Notebook suivant : `04_tinyML.ipynb`** — l'IA minuscule pour les capteurs.
"""))

save(cells, "../02_SLM_petits_modeles.ipynb")
