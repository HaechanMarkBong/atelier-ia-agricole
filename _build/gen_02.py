from gen_common import md, code, save, CONFIG_CELL, KAGGLE_CELL, IMAGES_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 02. VLM : comprendre des images de plantes

Un **VLM** (*Vision-Language Model*) comprend à la fois les **images** et le **texte**.
On peut lui montrer une **photo de feuille** et lui demander de la **décrire** ou de
répondre à une **question** à son sujet.

Ce notebook n'est pas un cours de programmation : le code reste minimal. L'objectif est
d'apprendre à **utiliser** un modèle de fondation vision-langage sur un vrai jeu de données,
et d'observer l'effet de réglages comme la **longueur de réponse** et le **prompt engineering**
(la façon de poser la question).

On utilise **SmolVLM** (Hugging Face), un VLM **compact (~250 à 500 M de paramètres)**, assez
léger pour tourner sur le CPU gratuit de Google Colab.
"""))

cells.append(code(CONFIG_CELL))

cells.append(code('''\
pip_install("transformers>=4.56", "accelerate", "pillow")
from transformers import pipeline
print("✅ Bibliothèques prêtes.")
'''))

cells.append(code(KAGGLE_CELL))
cells.append(code(IMAGES_CELL))

cells.append(md("""\
## 1. Charger le VLM

- En **mode démo** : `SmolVLM-256M-Instruct` (250M paramètres, rapide à tester).
- En **mode complet** : `SmolVLM-500M-Instruct` (500M paramètres, plus précis).

On désactive le **découpage en tuiles** de l'image (`do_image_splitting`) : cela suffit pour
décrire une photo simple, et ça reste **rapide** même sans GPU.
"""))

cells.append(code('''\
MODELE_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct" if MODE_DEMO else "HuggingFaceTB/SmolVLM-500M-Instruct"

pipe = pipeline("image-text-to-text", model=MODELE_VLM, dtype="auto")
pipe.image_processor.do_image_splitting = False

n_params = pipe.model.num_parameters() / 1e6
print(f"✅ {MODELE_VLM} chargé (~{n_params:.0f} M paramètres)")
'''))

cells.append(md("""\
## 2. Décrire et interroger une image

Une seule fonction suffit : on donne une **image** et une **question** (texte), le modèle
répond. `max_new_tokens` limite la longueur de la réponse générée.
"""))

cells.append(code('''\
def demander_image(image, question="Describe the image in one short sentence.", max_new_tokens=40):
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": question}]}]
    sortie = pipe(text=messages, max_new_tokens=max_new_tokens)
    return sortie[0]["generated_text"][-1]["content"].strip()
'''))

cells.append(md("""\
## 3. Un vrai jeu de données agricole (Kaggle)

On utilise le jeu **Plant Disease Recognition** (Kaggle) : des photos de feuilles réparties
en 3 catégories — `Healthy`, `Powdery`, `Rust`. On demande au VLM un **diagnostic rapide** sur
plusieurs photos, et on compare à la vraie catégorie.
"""))

cells.append(code('''\
N_EXEMPLES = 3 if MODE_DEMO else 10
echantillon = echantillon_images_plantes(N_EXEMPLES)
print(f"{len(echantillon)} images chargées.")

question_diagnostic = ("Look at this plant leaf. In one short sentence, say if it looks "
                       "healthy or diseased, and why.")

for image, vrai_label in echantillon:
    reponse = demander_image(image, question_diagnostic)
    print(f"Vrai : {vrai_label:10s} | VLM : {reponse}")
'''))

cells.append(md("""\
> 🌍 **Remarque langue.** Ces petits modèles VLM « pensent » surtout en **anglais**.
> Astuce : posez la question en anglais, puis **traduisez la réponse** avec un SLM/LLM
> (notebooks 01 et 05) !
"""))

cells.append(md("""\
---
# 🏋️ Exercices

Ces exercices ne visent pas à *écrire du code*, mais à **observer l'effet** de quelques
réglages sur un modèle vision-langage.
"""))

cells.append(md("""\
### 🏋️ Exercice 1 — Effet de `max_new_tokens`

Prenez la première image de `echantillon` et demandez une description avec
`max_new_tokens=10`, puis `max_new_tokens=80`. Que se passe-t-il avec la réponse courte ?
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 1
"""))

cells.append(code('''\
image_test, _ = echantillon[0]
for taille in [10, 80]:
    print(f"--- max_new_tokens = {taille} ---")
    print(demander_image(image_test, "Describe this leaf in detail.", max_new_tokens=taille))
    print()
'''))

cells.append(md("""\
### 🏋️ Exercice 2 — Prompt engineering : question vague vs question ciblée

Sur la même image, comparez une question **vague** (« What do you see? ») à une question
**ciblée** (« Does this leaf show signs of rust disease, powdery mildew, or is it healthy?
Answer with one word. »). Laquelle donne une réponse plus utile pour un diagnostic ?
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 2
"""))

cells.append(code('''\
prompt_vague = "What do you see?"
prompt_cible = ("Does this leaf show signs of rust disease, powdery mildew, or is it "
                "healthy? Answer with one word.")

print("=== Vague ===")
print(demander_image(image_test, prompt_vague))
print("\\n=== Ciblée ===")
print(demander_image(image_test, prompt_cible))
'''))

cells.append(md("""\
## ✅ Récapitulatif

- Un **VLM** relie **image** et **texte** : description ou question/réponse visuelle.
- `max_new_tokens` contrôle la longueur ; une question **ciblée** (prompt engineering) donne
  une réponse plus exploitable qu'une question vague.
- Très utile pour le **pré-diagnostic** à partir d'une photo — à confirmer toujours par un
  expert.

**➡️ Notebook suivant : `03_TinyLLM_TinyVLM.ipynb`** — les modèles génératifs les plus petits.
"""))

save(cells, "../notebooks/02_VLM_vision.ipynb")
