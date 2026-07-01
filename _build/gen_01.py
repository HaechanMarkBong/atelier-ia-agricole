from gen_common import md, code, save, CONFIG_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 01. LLM avec Ollama

Un **LLM** (*Large Language Model*) est un modèle entraîné sur d'énormes quantités de texte.
Il sait **répondre à des questions**, **résumer**, **traduire**, **rédiger**…

Dans ce notebook, on l'utilise comme **assistant agronome** :
- charger un LLM avec Ollama ;
- lui poser des questions agricoles ;
- découvrir l'**ingénierie de prompt** (l'art de bien formuler ses demandes) ;
- comprendre le paramètre **température**.

> ⚙️ Exécutez d'abord la cellule de configuration, puis suivez l'ordre des cellules.
"""))

cells.append(code(CONFIG_CELL))

cells.append(md("""\
## 1. Préparer Ollama

On réutilise les fonctions du notebook 00 : démarrer le serveur, puis discuter avec un modèle.
(La cellule réinstalle/relance ce qu'il faut, au cas où vous repartez de zéro.)
"""))

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
## 2. Choisir et charger un modèle

On définit le modèle dans **une seule variable** : `MODELE`.
- En **mode démo** (test rapide) : un petit modèle.
- En **mode complet** (atelier) : `llama3.2` (~2 Go), plus performant.

👉 Pour changer de modèle, il suffira de modifier cette variable (voir les exercices).
"""))

cells.append(code('''\
MODELE = "qwen2.5:0.5b" if MODE_DEMO else "llama3.2"

print(f"Téléchargement du modèle : {MODELE} ...")
subprocess.run(["ollama", "pull", MODELE], check=False)
print("✅ Modèle prêt.")
'''))

cells.append(code('''\
def chat(prompt, modele=None, temperature=0.7, systeme=None):
    """Interroge le LLM. `systeme` = consigne de rôle ; `temperature` = créativité (0=strict)."""
    modele = modele or MODELE
    corps = {
        "model": modele,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if systeme:
        corps["system"] = systeme
    donnees = json.dumps(corps).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=donnees,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]

# Premier test
print(chat("En une phrase, qu'est-ce que la rotation des cultures ?"))
'''))

cells.append(md("""\
## 3. Un assistant agronome

Le paramètre **`system`** donne un **rôle** au modèle. Comparez la différence de ton :
"""))

cells.append(code('''\
consigne = ("Tu es un ingénieur agronome francophone. "
            "Tu réponds de façon claire, pratique et concise, pour de petits agriculteurs.")

question = "Mes feuilles de tomate ont des taches jaunes. Quelles peuvent être les causes ?"
print(chat(question, systeme=consigne))
'''))

cells.append(md("""\
## 4. L'ingénierie de prompt (prompt engineering)

La **qualité de la réponse dépend de la façon de poser la question**.
Trois leviers simples :

1. **Donner un rôle** (« Tu es un agronome… »).
2. **Donner du contexte** (région, culture, saison…).
3. **Imposer un format** (liste, tableau, nombre de mots…).

Comparez un prompt **vague** et un prompt **précis** :
"""))

cells.append(code('''\
prompt_vague = "Parle-moi du maïs."

prompt_precis = (
    "Tu es un conseiller agricole. Pour un agriculteur en Afrique de l'Ouest qui débute, "
    "donne 3 conseils pratiques pour réussir une culture de maïs en saison des pluies. "
    "Réponds sous forme de liste à puces, une phrase par conseil."
)

print("=== PROMPT VAGUE ===")
print(chat(prompt_vague))
print("\\n=== PROMPT PRÉCIS ===")
print(chat(prompt_precis))
'''))

cells.append(md("""\
## 5. La température

La **température** contrôle la créativité :
- `0.0` → réponses **déterministes**, factuelles, répétitives.
- `1.0` → réponses **variées**, créatives (mais parfois moins fiables).

Pour des **conseils techniques**, une température **basse** est préférable.
"""))

cells.append(code('''\
q = "Propose un nom de marque pour un miel bio de lavande."
for t in [0.0, 1.0]:
    print(f"--- température = {t} ---")
    print(chat(q, temperature=t))
    print()
'''))

cells.append(md("""\
---
# 🏋️ Exercices

Essayez par vous-même ! Les **solutions** se trouvent juste après chaque exercice
(repliez-les pour d'abord chercher seul).
"""))

cells.append(md("""\
### 🏋️ Exercice 1 — Changer de modèle

Téléchargez un autre petit modèle, par exemple **`gemma2:2b`** (ou `qwen2.5:0.5b` en démo),
puis posez-lui la question : *« Qu'est-ce que le paillage (mulching) ? »*
en utilisant l'argument `modele=...` de la fonction `chat`.
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 1
"""))

cells.append(code('''\
autre_modele = "qwen2.5:0.5b" if MODE_DEMO else "gemma2:2b"
subprocess.run(["ollama", "pull", autre_modele], check=False)
print(chat("Qu'est-ce que le paillage (mulching) ?", modele=autre_modele))
'''))

cells.append(md("""\
### 🏋️ Exercice 2 — Améliorer un prompt

Le prompt ci-dessous est trop vague. **Réécrivez-le** en ajoutant : un rôle, un contexte
(culture = vigne, problème = mildiou) et un format (liste de 3 actions).
"""))

cells.append(code('''\
prompt_a_ameliorer = "Comment traiter une maladie ?"
# 👉 Réécrivez ce prompt puis appelez chat(...)
'''))

cells.append(md("""\
### ✅ Solution 2
"""))

cells.append(code('''\
prompt_ameliore = (
    "Tu es un conseiller viticole. Un viticulteur observe du mildiou sur sa vigne. "
    "Donne 3 actions concrètes pour limiter la propagation, sous forme de liste numérotée."
)
print(chat(prompt_ameliore))
'''))

cells.append(md("""\
### 🏋️ Exercice 3 — Effet de la température

Demandez au modèle d'**inventer un slogan** pour une coopérative agricole,
en testant les températures `0.2` puis `0.9`. Observez la différence de créativité.
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 3
"""))

cells.append(code('''\
for t in [0.2, 0.9]:
    print(f"--- température = {t} ---")
    print(chat("Invente un slogan court pour une coopérative agricole.", temperature=t))
    print()
'''))

cells.append(md("""\
## ✅ Récapitulatif

- Un LLM se charge en **une variable** (`MODELE`) avec Ollama.
- La fonction `chat(...)` accepte un **rôle** (`system`) et une **température**.
- Un **bon prompt** = rôle + contexte + format.

**➡️ Notebook suivant : `03_VLM_vision.ipynb`** — les images, puis la génération d'images.
"""))

save(cells, "../01_LLM_ollama.ipynb")
