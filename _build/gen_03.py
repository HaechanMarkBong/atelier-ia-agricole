from gen_common import md, code, save, CONFIG_CELL, KAGGLE_CELL, IMAGES_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 03. TinyLLM & TinyVLM: les modèles génératifs les plus petits

Jusqu'ici, on a utilisé des modèles **proches du milliard** de paramètres (SLM, VLM) et on en
utilisera un de **~9 milliards** (notebook 05). Ce notebook explore l'**autre extrême**: les
plus petits modèles génératifs disponibles sur Hugging Face — un **TinyLLM** (texte) et un
**TinyVLM** (image + texte), chacun sous les **300 millions** de paramètres.

| Famille | Paramètres | Notebook |
|---------|-----------|----------|
| SLM / VLM | ~1 Md | 01, 02 |
| **TinyLLM / TinyVLM (ici)** | **~0,1-0,3 Md** | 03 |
| LLM | ~9 Md | 05 |

**Objectif:** réutiliser **les mêmes tâches et le même jeu de données** que les notebooks 01
et 02, mais avec des modèles bien plus petits, pour **observer concrètement** la perte de
qualité — et l'effet encore plus marqué de réglages comme la température ou le few-shot sur
un modèle minuscule.
"""))

cells.append(code(CONFIG_CELL))

cells.append(code('''\
pip_install("transformers>=4.56", "accelerate", "pillow")
import gc
from transformers import pipeline
print("✅ Bibliothèques prêtes.")
'''))

cells.append(code(KAGGLE_CELL))
cells.append(code(IMAGES_CELL))

cells.append(md("""\
## 1. TinyLLM: `SmolLM2-135M-Instruct`

135 millions de paramètres — environ **8 fois plus petit** que le SLM du notebook 01.
"""))

cells.append(code('''\
MODELE_TINYLLM = "HuggingFaceTB/SmolLM2-135M-Instruct"
pipe_tinyllm = pipeline("text-generation", model=MODELE_TINYLLM, dtype="auto")
n_params = pipe_tinyllm.model.num_parameters() / 1e6
print(f"✅ {MODELE_TINYLLM} chargé (~{n_params:.0f} M paramètres)")

def chat_tiny(prompt, temperature=0.0, max_new_tokens=60):
    messages = [{"role": "user", "content": prompt}]
    options = {"max_new_tokens": max_new_tokens}
    options.update({"do_sample": True, "temperature": temperature} if temperature > 0
                    else {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 50})
    sortie = pipe_tinyllm(messages, **options)
    return sortie[0]["generated_text"][-1]["content"].strip()

print(chat_tiny("En une phrase, qu'est-ce que la rotation des cultures?"))
'''))

cells.append(md("""\
## 2. TinyLLM sur le jeu de données Kaggle (crop recommendation)

On reprend **exactement** la tâche du notebook 01 (recommander une culture à partir de
mesures de sol), en **zero-shot**, avec le TinyLLM.
"""))

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

def ligne_en_texte(ligne):
    return (f"N={ligne['N']}, P={ligne['P']}, K={ligne['K']}, "
            f"température={ligne['temperature']:.1f}°C, humidité={ligne['humidity']:.0f}%, "
            f"pH={ligne['ph']:.1f}, pluie={ligne['rainfall']:.0f}mm")

# Donner la liste FERMÉE des cultures possibles aide beaucoup un petit modèle: sans elle,
# il invente souvent des mots hors sujet plutôt que de choisir une vraie culture du jeu.
LISTE_CULTURES = sorted(df["label"].unique())
CULTURES_TXT = ", ".join(LISTE_CULTURES)

N_EXEMPLES = 3 if MODE_DEMO else 10
echantillon_crop = df.sample(n=N_EXEMPLES, random_state=42)

for _, ligne in echantillon_crop.iterrows():
    prompt = (f"Mesures: {ligne_en_texte(ligne)}. Quelle culture recommandes-tu, en "
              f"choisissant UNIQUEMENT dans cette liste: {CULTURES_TXT}?\\n"
              "Réponds uniquement par le nom de la culture.")
    prediction = chat_tiny(prompt, max_new_tokens=20)
    print(f"Vrai: {ligne['label']:12s} | TinyLLM: {prediction}")
'''))

cells.append(md("""\
> 💡 Comparez ces réponses à celles du notebook 01 (SLM ~1 Md) sur la même tâche: le TinyLLM
> se trompe ou dérape plus souvent — c'est le compromis taille/qualité en action.
"""))

cells.append(code('''\
del pipe_tinyllm
gc.collect()
'''))

cells.append(md("""\
## 3. TinyVLM: `SmolVLM-256M-Instruct`

256 millions de paramètres — environ **2 fois plus petit** que le VLM du notebook 02.
"""))

cells.append(code('''\
MODELE_TINYVLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
pipe_tinyvlm = pipeline("image-text-to-text", model=MODELE_TINYVLM, dtype="auto")
pipe_tinyvlm.image_processor.do_image_splitting = False
n_params = pipe_tinyvlm.model.num_parameters() / 1e6
print(f"✅ {MODELE_TINYVLM} chargé (~{n_params:.0f} M paramètres)")

def demander_image_tiny(image, question, max_new_tokens=40):
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": question}]}]
    sortie = pipe_tinyvlm(text=messages, max_new_tokens=max_new_tokens)
    return sortie[0]["generated_text"][-1]["content"].strip()
'''))

cells.append(md("""\
## 4. TinyVLM sur le jeu de données Kaggle (plant disease)

Même tâche que le notebook 02 (diagnostic Healthy / Powdery / Rust), avec le TinyVLM.
"""))

cells.append(code('''\
echantillon_images = echantillon_images_plantes(N_EXEMPLES)
question_diagnostic = ("Look at this plant leaf. In one short sentence, say if it looks "
                       "healthy or diseased, and why.")

for image, vrai_label in echantillon_images:
    reponse = demander_image_tiny(image, question_diagnostic)
    print(f"Vrai: {vrai_label:10s} | TinyVLM: {reponse}")
'''))

cells.append(md("""\
---
# 🏋️ Exercices

Ces exercices ne visent pas à *écrire du code*, mais à **observer l'effet** de quelques
réglages — souvent plus marqué sur un modèle aussi petit.
"""))

cells.append(md("""\
### 🏋️ Exercice 1 — La température casse-t-elle plus vite un tout petit modèle?

Redemandez au TinyLLM d'inventer un slogan agricole avec `temperature=0.0`, `0.7`, puis `1.2`.
Comparez à l'exercice équivalent du notebook 01 (SLM): le TinyLLM décroche-t-il plus tôt?
"""))

cells.append(code('''\
# 👉 Votre code ici (réutilisez chat_tiny — il faudra recharger pipe_tinyllm si vous l'avez supprimé)
'''))

cells.append(md("""\
### ✅ Solution 1
"""))

cells.append(code('''\
pipe_tinyllm = pipeline("text-generation", model=MODELE_TINYLLM, dtype="auto")
for t in [0.0, 0.7, 1.2]:
    print(f"--- température = {t} ---")
    print(chat_tiny("Invente un slogan court pour une coopérative agricole.", temperature=t))
    print()
'''))

cells.append(md("""\
### 🏋️ Exercice 2 — Le few-shot aide-t-il un TinyLLM?

Ajoutez 2 exemples résolus avant une nouvelle mesure de sol (comme au notebook 01), et
comparez la prédiction **zero-shot** à la prédiction **few-shot** pour la même ligne.
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 2
"""))

cells.append(code('''\
ligne_test = df.drop(echantillon_crop.index).sample(n=1, random_state=99).iloc[0]
exemples = df.drop(echantillon_crop.index).sample(n=2, random_state=7)

prompt_zero = (f"Mesures: {ligne_en_texte(ligne_test)}. Quelle culture recommandes-tu, en "
               f"choisissant UNIQUEMENT dans cette liste: {CULTURES_TXT}?\\n"
               "Réponds uniquement par le nom de la culture.")
demo = "\\n".join(f"Mesures: {ligne_en_texte(e)} → {e['label']}" for _, e in exemples.iterrows())
prompt_few = (f"Voici des exemples:\\n{demo}\\n\\nMesures: {ligne_en_texte(ligne_test)}\\n"
              f"Culture recommandée (UNIQUEMENT parmi: {CULTURES_TXT}):")

print("Zero-shot:", chat_tiny(prompt_zero, max_new_tokens=20))
print("Few-shot:", chat_tiny(prompt_few, max_new_tokens=20))
print("Vrai:", ligne_test["label"])
'''))

cells.append(md("""\
## ✅ Récapitulatif

- **TinyLLM** et **TinyVLM** (≤ 300M paramètres) sont les plus petits modèles génératifs
  disponibles — encore plus légers qu'un SLM/VLM (~1 Md).
- Sur la **même tâche**, ils se trompent plus souvent: la taille a un vrai coût en qualité.
- Les réglages (température, few-shot) ont un effet **encore plus marqué** sur un modèle
  minuscule que sur un modèle de taille moyenne.

**➡️ Notebook suivant: `04_OV_detection_ouverte.ipynb`** — détection ouverte, segmentation
et tracking.
"""))

save(cells, "../notebooks/03_TinyLLM_TinyVLM.ipynb")
