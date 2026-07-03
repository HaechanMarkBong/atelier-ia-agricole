from gen_common import md, code, save, CONFIG_CELL, KAGGLE_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 01. SLM : les petits modèles de langage

Un **SLM** (*Small Language Model*) est un modèle de langage **compact** (autour du **milliard
de paramètres**, au lieu de dizaines de milliards pour un LLM).

**Pourquoi est-ce crucial en agriculture ?**
- 📶 Fonctionne **hors-ligne** (utile au champ, sans Internet).
- 📱 Tourne sur du **matériel modeste** (téléphone, mini-PC, Raspberry Pi).
- ⚡ **Rapide** et **peu gourmand** en énergie.
- 🔒 Données qui **restent locales** (confidentialité).

Le compromis : un SLM est **moins savant** qu'un gros LLM (notebook 05). Ce notebook n'est pas
un cours de programmation : le code est volontairement court, pour se concentrer sur **comment
utiliser** un modèle de fondation et sur l'effet de réglages comme la **température**, la
**taille de réponse** et le **prompt engineering** (dont l'apprentissage *few-shot*).
"""))

cells.append(code(CONFIG_CELL))

cells.append(code('''\
pip_install("transformers>=4.56", "accelerate")
import gc, time
from transformers import pipeline
print("✅ Bibliothèques prêtes.")
'''))

cells.append(md("""\
## 1. Choisir des SLM à comparer

Deux modèles **ouverts** (aucune clé API requise), du plus petit au plus proche du milliard :

| Modèle (Hugging Face) | Paramètres |
|------------------------|-----------|
| `Qwen/Qwen2.5-0.5B-Instruct`          | 0,5 Md  |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  | 1,1 Md  |

En **mode démo**, on ne charge que le plus petit (pour aller vite). En mode complet, on compare
les deux et on observe l'effet de la taille.
"""))

cells.append(code('''\
if MODE_DEMO:
    MODELES_SLM = ["Qwen/Qwen2.5-0.5B-Instruct"]
else:
    MODELES_SLM = ["Qwen/Qwen2.5-0.5B-Instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]

print("Modèles à comparer :", MODELES_SLM)
'''))

cells.append(md("""\
## 2. Charger et interroger un SLM

On charge chaque modèle avec la fonction `pipeline(...)` de Transformers — la façon la plus
simple d'utiliser un modèle Hugging Face. Un **cache** garde **un seul modèle en mémoire à la
fois** (utile quand la RAM est limitée, comme sur un petit ordinateur ou au champ).
"""))

cells.append(code('''\
_pipe_cache = {}

def obtenir_pipeline(nom_modele):
    """Charge un pipeline de génération de texte (et vide le cache précédent pour la RAM)."""
    if nom_modele not in _pipe_cache:
        _pipe_cache.clear()
        gc.collect()
        print(f"Chargement de {nom_modele} ...")
        _pipe_cache[nom_modele] = pipeline("text-generation", model=nom_modele, dtype="auto")
    return _pipe_cache[nom_modele]

def chat(nom_modele, prompt, systeme=None, temperature=0.0, max_new_tokens=120):
    """Interroge le SLM.
    - `systeme`        : consigne de rôle (prompt engineering)
    - `temperature`    : créativité (0 = déterministe, 1 = très varié)
    - `max_new_tokens` : taille maximale de la réponse générée
    """
    pipe = obtenir_pipeline(nom_modele)
    messages = ([{"role": "system", "content": systeme}] if systeme else []) + \\
               [{"role": "user", "content": prompt}]
    options = {"max_new_tokens": max_new_tokens}
    # En mode déterministe (do_sample=False), on remet les réglages d'échantillonnage à leur
    # valeur neutre pour éviter les avertissements de transformers (« temperature is set... »).
    options.update({"do_sample": True, "temperature": temperature} if temperature > 0
                    else {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 50})
    sortie = pipe(messages, **options)
    return sortie[0]["generated_text"][-1]["content"].strip()

# Premier test
print(chat(MODELES_SLM[0], "En une phrase, qu'est-ce que la rotation des cultures ?"))
'''))

cells.append(md("""\
## 3. Comparer taille, vitesse et qualité

On pose la **même** question à chaque modèle et on **chronomètre** la réponse.
"""))

cells.append(code('''\
question = "En 2 phrases, explique pourquoi la rotation des cultures améliore le sol."

for m in MODELES_SLM:
    debut = time.time()
    reponse = chat(m, question)
    duree = time.time() - debut
    n_params = obtenir_pipeline(m).model.num_parameters() / 1e6
    print(f"\\n=== {m}  (~{n_params:.0f} M paramètres, {duree:.1f} s) ===")
    print(reponse)
'''))

cells.append(md("""\
## 4. Un vrai jeu de données agricole (Kaggle)

On utilise le jeu **Crop Recommendation Dataset** (Kaggle) : 2200 mesures de sol/climat
(azote N, phosphore P, potassium K, température, humidité, pH, pluviométrie) avec la
**culture recommandée** par des agronomes. `kagglehub` télécharge les jeux **publics** sans
clé API ; si Kaggle est injoignable, un petit échantillon de secours prend le relais.
"""))

cells.append(code(KAGGLE_CELL))

cells.append(code('''\
import pandas as pd

ECHANTILLON_SECOURS = [
    {"N": 20, "P": 129, "K": 201, "temperature": 23.4, "humidity": 91.7, "ph": 5.59, "rainfall": 116.1, "label": "apple"},
    {"N": 100, "P": 80, "K": 52, "temperature": 27.5, "humidity": 77.3, "ph": 6.05, "rainfall": 110.3, "label": "banana"},
    {"N": 43, "P": 68, "K": 20, "temperature": 29.6, "humidity": 66.2, "ph": 7.5, "rainfall": 69.4, "label": "blackgram"},
    {"N": 43, "P": 68, "K": 81, "temperature": 17.5, "humidity": 17.9, "ph": 6.76, "rainfall": 78.9, "label": "chickpea"},
    {"N": 21, "P": 20, "K": 31, "temperature": 25.6, "humidity": 99.7, "ph": 5.86, "rainfall": 165.8, "label": "coconut"},
    {"N": 107, "P": 31, "K": 31, "temperature": 23.2, "humidity": 53.0, "ph": 6.77, "rainfall": 153.1, "label": "coffee"},
    {"N": 122, "P": 40, "K": 17, "temperature": 25.0, "humidity": 81.3, "ph": 6.85, "rainfall": 80.0, "label": "cotton"},
    {"N": 22, "P": 133, "K": 201, "temperature": 23.8, "humidity": 80.1, "ph": 6.0, "rainfall": 67.3, "label": "grapes"},
    {"N": 80, "P": 43, "K": 43, "temperature": 23.8, "humidity": 74.4, "ph": 6.01, "rainfall": 172.6, "label": "jute"},
    {"N": 24, "P": 67, "K": 22, "temperature": 20.1, "humidity": 22.9, "ph": 5.62, "rainfall": 104.6, "label": "kidneybeans"},
    {"N": 18, "P": 66, "K": 22, "temperature": 25.9, "humidity": 67.6, "ph": 6.35, "rainfall": 47.9, "label": "lentil"},
    {"N": 76, "P": 48, "K": 18, "temperature": 19.3, "humidity": 69.6, "ph": 5.78, "rainfall": 83.2, "label": "maize"},
    {"N": 18, "P": 26, "K": 31, "temperature": 32.6, "humidity": 47.7, "ph": 5.42, "rainfall": 91.1, "label": "mango"},
    {"N": 25, "P": 51, "K": 18, "temperature": 27.8, "humidity": 54.8, "ph": 9.46, "rainfall": 50.3, "label": "mothbeans"},
    {"N": 21, "P": 44, "K": 18, "temperature": 27.1, "humidity": 86.9, "ph": 7.13, "rainfall": 50.5, "label": "mungbean"},
    {"N": 100, "P": 17, "K": 48, "temperature": 29.7, "humidity": 94.3, "ph": 6.37, "rainfall": 26.5, "label": "muskmelon"},
    {"N": 12, "P": 20, "K": 10, "temperature": 24.5, "humidity": 93.1, "ph": 6.53, "rainfall": 109.5, "label": "orange"},
    {"N": 54, "P": 67, "K": 52, "temperature": 35.7, "humidity": 93.3, "ph": 6.59, "rainfall": 141.3, "label": "papaya"},
    {"N": 27, "P": 71, "K": 23, "temperature": 23.5, "humidity": 46.5, "ph": 7.11, "rainfall": 150.9, "label": "pigeonpeas"},
    {"N": 21, "P": 21, "K": 38, "temperature": 22.6, "humidity": 89.3, "ph": 6.33, "rainfall": 104.9, "label": "pomegranate"},
    {"N": 81, "P": 53, "K": 42, "temperature": 23.7, "humidity": 81.0, "ph": 5.18, "rainfall": 233.7, "label": "rice"},
    {"N": 103, "P": 16, "K": 49, "temperature": 24.1, "humidity": 81.6, "ph": 6.92, "rainfall": 51.8, "label": "watermelon"},
]

dossier = telecharger_dataset_kaggle("atharvaingle/crop-recommendation-dataset")
df = None
if dossier:
    try:
        df = pd.read_csv(f"{dossier}/Crop_recommendation.csv")
    except Exception as e:
        print(f"⚠️ Lecture du CSV Kaggle impossible ({e}) → échantillon de secours.")
if df is None:
    df = pd.DataFrame(ECHANTILLON_SECOURS)

print(f"{len(df)} lignes disponibles, {df['label'].nunique()} cultures différentes.")
df.head(3)
'''))

cells.append(md("""\
## 5. Zero-shot : demander une recommandation sans exemple

On transforme chaque ligne en phrase, puis on demande au SLM de **deviner** la culture —
sans lui montrer aucun exemple au préalable (« zero-shot »).
"""))

cells.append(code('''\
def ligne_en_texte(ligne):
    return (f"N={ligne['N']}, P={ligne['P']}, K={ligne['K']}, "
            f"température={ligne['temperature']:.1f}°C, humidité={ligne['humidity']:.0f}%, "
            f"pH={ligne['ph']:.1f}, pluie={ligne['rainfall']:.0f}mm")

# Donner la liste FERMÉE des cultures possibles aide beaucoup un petit modèle : sans elle,
# il invente souvent des mots hors sujet plutôt que de choisir une vraie culture du jeu.
LISTE_CULTURES = sorted(df["label"].unique())
CULTURES_TXT = ", ".join(LISTE_CULTURES)

def prompt_zero_shot(ligne):
    return (f"Voici des mesures de sol et de climat : {ligne_en_texte(ligne)}.\\n"
            f"Quelle culture recommandes-tu, en choisissant UNIQUEMENT dans cette liste : "
            f"{CULTURES_TXT} ?\\nRéponds uniquement par le nom de la culture.")

N_EXEMPLES = 3 if MODE_DEMO else 10
echantillon = df.sample(n=N_EXEMPLES, random_state=42)

for _, ligne in echantillon.iterrows():
    prediction = chat(MODELES_SLM[-1], prompt_zero_shot(ligne), max_new_tokens=20)
    print(f"Vrai : {ligne['label']:12s} | Prédit (zero-shot) : {prediction}")
'''))

cells.append(md("""\
## 6. Few-shot : donner 2 exemples avant de demander

Le **prompt engineering** le plus efficace pour guider un petit modèle : montrer quelques
exemples résolus (« few-shot ») avant la vraie question. On réutilise le **même** échantillon
pour comparer équitablement avec le zero-shot.
"""))

cells.append(code('''\
exemples_fewshot = df.drop(echantillon.index).sample(n=2, random_state=7)

def prompt_few_shot(ligne, exemples):
    demo = "\\n".join(f"Mesures : {ligne_en_texte(e)} → {e['label']}"
                      for _, e in exemples.iterrows())
    return (f"Voici des exemples de mesures et la culture recommandée :\\n{demo}\\n\\n"
            f"Nouvelle mesure : {ligne_en_texte(ligne)}\\n"
            f"Culture recommandée (UNIQUEMENT parmi : {CULTURES_TXT}) :")

for _, ligne in echantillon.iterrows():
    prediction = chat(MODELES_SLM[-1], prompt_few_shot(ligne, exemples_fewshot), max_new_tokens=20)
    print(f"Vrai : {ligne['label']:12s} | Prédit (few-shot)   : {prediction}")
'''))

cells.append(md("""\
> 💡 Comparez les deux sections : le **few-shot** aide souvent un petit modèle à mieux
> respecter le format de réponse attendu, et parfois à être plus précis.
"""))

cells.append(md("""\
## 7. Un assistant agronome (rôle système)

Le paramètre **`systeme`** donne un **rôle** au modèle — un autre levier de prompt engineering.
"""))

cells.append(code('''\
consigne = ("Tu es un ingénieur agronome francophone. "
            "Tu réponds de façon claire, pratique et concise, pour de petits agriculteurs.")

question = "Mes feuilles de tomate ont des taches jaunes. Quelles peuvent être les causes ?"
print(chat(MODELES_SLM[-1], question, systeme=consigne))
'''))

cells.append(md("""\
## 8. Quand préférer un SLM ?

- ✅ **Au champ, hors-ligne**, sur téléphone ou mini-PC.
- ✅ Pour des tâches **simples et cadrées** (FAQ, classification, extraction).
- ❌ Évitez-les pour du **raisonnement complexe** ou des connaissances très pointues
  → là, un gros LLM quantifié (notebook 05) reste meilleur.
"""))

cells.append(md("""\
---
# 🏋️ Exercices

Ces exercices ne visent pas à *écrire du code*, mais à **observer l'effet** de quelques
réglages clés sur un modèle de fondation.
"""))

cells.append(md("""\
### 🏋️ Exercice 1 — Effet de la température

Demandez au plus gros modèle (`MODELES_SLM[-1]`) d'**inventer un slogan** pour une
coopérative agricole, avec `temperature=0.0`, `0.5`, puis `1.0`. Que remarquez-vous sur la
créativité et la cohérence des réponses ?
"""))

cells.append(code('''\
for t in [0.0, 0.5, 1.0]:
    print(f"--- température = {t} ---")
    print(chat(MODELES_SLM[-1], "Invente un slogan court pour une coopérative agricole.",
               temperature=t))
    print()
'''))

cells.append(md("""\
### 🏋️ Exercice 2 — Effet de la taille de réponse (`max_new_tokens`)

Posez la question *« Explique les avantages de l'agroforesterie. »* avec
`max_new_tokens=15`, puis `max_new_tokens=150`. Observez où la réponse est **coupée**.
"""))

cells.append(code('''\
for taille in [15, 150]:
    print(f"--- max_new_tokens = {taille} ---")
    print(chat(MODELES_SLM[-1], "Explique les avantages de l'agroforesterie.",
               max_new_tokens=taille))
    print()
'''))

cells.append(md("""\
### 🏋️ Exercice 3 — 0-shot vs 1-shot vs few-shot

Reprenez `prompt_few_shot(...)` avec **1 seul** exemple, puis avec **4 exemples**, pour une
même ligne du jeu de données. Le nombre d'exemples change-t-il la réponse ?
"""))

cells.append(code('''\
ligne_test = df.drop(echantillon.index).sample(n=1, random_state=99).iloc[0]

for k in [1, 4]:
    exemples_k = df.drop(echantillon.index).sample(n=k, random_state=7)
    prediction = chat(MODELES_SLM[-1], prompt_few_shot(ligne_test, exemples_k), max_new_tokens=20)
    print(f"{k}-shot → {prediction}  (vrai : {ligne_test['label']})")
'''))

cells.append(md("""\
## ✅ Récapitulatif

- Un **SLM** échange un peu de « savoir » contre **rapidité, taille réduite et fonctionnement
  hors-ligne**.
- La **température** contrôle la créativité, `max_new_tokens` la longueur de la réponse.
- Le **few-shot prompting** (montrer des exemples) guide un petit modèle sans le réentraîner.

**➡️ Notebook suivant : `02_VLM_vision.ipynb`** — comprendre des images de plantes.
"""))

save(cells, "../notebooks/01_SLM_petits_modeles.ipynb")
