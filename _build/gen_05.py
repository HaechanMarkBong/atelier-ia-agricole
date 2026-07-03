from gen_common import md, code, save, CONFIG_CELL, KAGGLE_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 05. LLM : faire tenir un grand modèle dans Google Colab

Un **LLM** (*Large Language Model*) est un modèle de langage **massif** (ici, un modèle
d'environ **9 milliards de paramètres**) : bien plus savant qu'un SLM (notebook 01), mais
aussi bien plus **lourd**.

**Le problème :** stocker 9 milliards de paramètres en pleine précision (fp32) demande
**~35 Go de mémoire** — impossible sur le GPU gratuit de Google Colab (T4, 16 Go de VRAM).

Deux solutions, présentées dans ce notebook :
1. **La quantification** — réduire la précision numérique des paramètres (fp32 → int4) pour
   faire tenir le LLM **localement** sur le GPU Colab.
2. **Le cloud (Groq)** — envoyer la question à un LLM **hébergé**, sans aucun calcul local.
"""))

cells.append(code(CONFIG_CELL))

cells.append(code('''\
pip_install("transformers>=4.56", "accelerate", "bitsandbytes")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
print("✅ Bibliothèques prêtes. GPU disponible :", torch.cuda.is_available())
'''))

cells.append(md("""\
## 1. Combien de mémoire prend un LLM de ~9 Md paramètres ?

Chaque paramètre est normalement stocké sur 4 octets (fp32). La quantification réduit ce coût :
"""))

cells.append(code('''\
N_PARAMS_LLM = 8_829_407_232  # 01-ai/Yi-1.5-9B-Chat (~8,8 Md paramètres)
VRAM_COLAB_GRATUIT_GO = 16    # GPU T4, offert gratuitement par Colab

print(f"{'Précision':<14}{'Taille':>10}{'Tient sur Colab ?':>22}")
for nom, octets_par_param in [("fp32", 4), ("fp16 / bf16", 2), ("int8", 1), ("int4 (NF4)", 0.5)]:
    go = N_PARAMS_LLM * octets_par_param / 1e9
    tient = "✅ oui" if go < VRAM_COLAB_GRATUIT_GO else "❌ non"
    print(f"{nom:<14}{go:>8.1f} Go{tient:>22}")
'''))

cells.append(md("""\
**Conclusion :** sans quantification (fp32/fp16), ce LLM ne tient **pas** sur un GPU Colab
gratuit. En **int8** ou **int4**, il tient très confortablement. On utilise la bibliothèque
**bitsandbytes** (gratuite, intégrée à Hugging Face) pour quantifier le modèle **au chargement**.
"""))

cells.append(md("""\
## 2. Charger le LLM en 4 bits

- En **mode démo** : un petit modèle (`Qwen2.5-0.5B-Instruct`) pour tester rapidement le code.
- En **mode complet, avec un GPU** : le vrai LLM `01-ai/Yi-1.5-9B-Chat` (~8,8 Md paramètres),
  quantifié en **4 bits (NF4)** grâce à `BitsAndBytesConfig`.

> ⚠️ Le LLM de ~9 Md **a besoin d'un GPU** : la quantification `bitsandbytes` utilise des noyaux
> CUDA, et le modèle non quantifié (~18 Go) ne tiendrait pas dans la RAM CPU de Colab (~13 Go).
> **Sans GPU activé**, ce notebook bascule donc automatiquement sur le petit modèle pour rester
> exécutable. Pour le vrai LLM : `Exécution` → `Modifier le type d'exécution` → **GPU (T4)**.

`device_map="auto"` place automatiquement le modèle sur le GPU s'il est disponible, sinon sur
le CPU.
"""))

cells.append(code('''\
GPU_DISPO = torch.cuda.is_available()

# bitsandbytes quantifie les poids avec des noyaux CUDA : un GPU est donc indispensable pour
# quantifier. De plus, un LLM de ~9 Md en précision normale (~18 Go) ne tient PAS dans la RAM
# CPU d'un Colab gratuit (~13 Go) : le charger sans GPU planterait la session (OOM). Donc,
# sans GPU, on bascule sur un petit modèle pour que le notebook reste exécutable partout.
# 👉 Sur Colab : Exécution → Modifier le type d'exécution → GPU (T4) pour le vrai LLM 9 Md quantifié.
MODELE_LLM = ("Qwen/Qwen2.5-0.5B-Instruct"
              if (MODE_DEMO or not GPU_DISPO) else "01-ai/Yi-1.5-9B-Chat")

config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

def charger_llm(config_quant=None):
    """Charge MODELE_LLM en appliquant la quantification bitsandbytes si un GPU est disponible,
    sinon effectue un chargement normal (CPU) pour rester exécutable partout."""
    if config_quant is not None and GPU_DISPO:
        return AutoModelForCausalLM.from_pretrained(MODELE_LLM, quantization_config=config_quant,
                                                    device_map="auto")
    return AutoModelForCausalLM.from_pretrained(MODELE_LLM, dtype="auto", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(MODELE_LLM)
modele = charger_llm(config_4bit)
pipe_llm = pipeline("text-generation", model=modele, tokenizer=tokenizer)

empreinte_go = modele.get_memory_footprint() / 1e9
if GPU_DISPO:
    print(f"✅ {MODELE_LLM} chargé en 4 bits — empreinte mémoire réelle : {empreinte_go:.2f} Go")
else:
    print(f"ℹ️ Pas de GPU CUDA : petit modèle {MODELE_LLM} chargé en précision normale (CPU).")
    print(f"   Empreinte mémoire réelle : {empreinte_go:.2f} Go")
    print("   👉 Activez un GPU T4 sur Colab pour charger le vrai LLM 9 Md quantifié en 4 bits.")
'''))

cells.append(md("""\
## 3. Interroger le LLM quantifié

Même si les poids sont compressés, on utilise le modèle **normalement** : la quantification
est invisible pour l'utilisateur, seule la mémoire (et un peu la vitesse) change.
`temperature` et `max_new_tokens` fonctionnent comme dans les notebooks précédents.
"""))

cells.append(code('''\
def chat_llm(prompt, systeme=None, temperature=0.0, max_new_tokens=150):
    messages = ([{"role": "system", "content": systeme}] if systeme else []) + \\
               [{"role": "user", "content": prompt}]
    options = {"max_new_tokens": max_new_tokens}
    options.update({"do_sample": True, "temperature": temperature} if temperature > 0
                    else {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 50})
    sortie = pipe_llm(messages, **options)
    return sortie[0]["generated_text"][-1]["content"].strip()

print(chat_llm("En 3 phrases, explique les avantages de l'agriculture de précision."))
'''))

cells.append(code(KAGGLE_CELL))

cells.append(md("""\
## 4. Le LLM sur un vrai jeu de données (Kaggle)

On reprend le jeu **Crop Recommendation Dataset** (notebook 01), mais on demande cette fois
une **recommandation justifiée en une phrase** — une tâche plus riche, à la portée d'un LLM
de ~9 Md paramètres.
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

N_EXEMPLES = 3 if MODE_DEMO else 10
echantillon = df.sample(n=N_EXEMPLES, random_state=42)

for _, ligne in echantillon.iterrows():
    prompt = (f"Mesures de sol : {ligne_en_texte(ligne)}. Quelle culture recommandes-tu, "
              "et pourquoi (une phrase) ? La vraie culture attendue est en anglais.")
    reponse = chat_llm(prompt, max_new_tokens=60)
    print(f"--- Vrai : {ligne['label']} ---\\n{reponse}\\n")
'''))

cells.append(md("""\
## 5. Utiliser un LLM cloud sans calcul local (Groq)

**Groq** héberge de gros LLM (par exemple **Llama 3.3 70B**) et répond très vite via une API —
utile quand même la quantification ne suffit pas (pas de GPU du tout, modèle encore plus gros).

1. Créez un compte gratuit sur [console.groq.com](https://console.groq.com).
2. Récupérez une clé gratuite sur **[console.groq.com/keys](https://console.groq.com/keys)**
   (`Create API Key`), puis collez-la ci-dessous.

> 🔒 Ne partagez jamais votre clé publiquement (ne la commitez pas sur GitHub).
"""))

cells.append(code('''\
GROQ_API_KEY = ""  # 👉 Collez votre clé ici (gratuite sur console.groq.com)
MODELE_GROQ = "llama-3.3-70b-versatile"  # ~70 Md paramètres, hébergé par Groq

if not GROQ_API_KEY:
    print("ℹ️ Pas de clé Groq : cette section sera ignorée.")
    print("   Créez une clé gratuite sur https://console.groq.com/keys et collez-la ci-dessus.")
else:
    pip_install("groq")
    from groq import Groq
    client_groq = Groq(api_key=GROQ_API_KEY)

    def chat_groq(prompt, systeme=None, temperature=0.0):
        messages = ([{"role": "system", "content": systeme}] if systeme else []) + \\
                   [{"role": "user", "content": prompt}]
        reponse = client_groq.chat.completions.create(
            model=MODELE_GROQ, messages=messages, temperature=temperature,
        )
        return reponse.choices[0].message.content.strip()

    print(chat_groq("En 3 phrases, explique les avantages de l'agriculture de précision."))
'''))

cells.append(md("""\
Si une clé est fournie, on peut reprendre la **même tâche** que la section 4 (recommandation
de culture), mais via Groq — sans aucun modèle chargé localement.
"""))

cells.append(code('''\
if GROQ_API_KEY:
    for _, ligne in echantillon.head(3).iterrows():
        prompt = (f"Mesures de sol : {ligne_en_texte(ligne)}. Quelle culture recommandes-tu, "
                  "et pourquoi (une phrase) ?")
        print(f"--- Vrai : {ligne['label']} ---\\n{chat_groq(prompt)}\\n")
else:
    print("ℹ️ Section ignorée (pas de clé Groq).")
'''))

cells.append(md("""\
## 6. int8 vs int4 : quel compromis choisir ?

| Précision | Mémoire (~9 Md params) | Qualité | Quand l'utiliser |
|-----------|------------------------|---------|-------------------|
| fp16      | ~17,7 Go               | Référence | GPU avec beaucoup de VRAM |
| int8      | ~8,8 Go                | Très proche du fp16 | Bon compromis si la VRAM le permet |
| int4 (NF4)| ~4,4 Go                | Légère perte, souvent imperceptible | GPU limité (Colab gratuit) |

| Approche | Calcul local ? | Limite principale |
|----------|-----------------|--------------------|
| Quantification (int4) | Oui (GPU Colab) | Taille max du modèle ≈ VRAM disponible |
| API cloud (Groq) | Non | Nécessite une connexion + une clé API |

> 💡 En pratique : quantifier en **int4** pour rester autonome sur Colab, ou utiliser **Groq**
> pour des modèles encore plus gros, sans se soucier du matériel.
"""))

cells.append(md("""\
---
# 🏋️ Exercices

Ces exercices ne visent pas à *écrire du code*, mais à **observer l'effet** de la
quantification et des réglages de génération.
"""))

cells.append(md("""\
### 🏋️ Exercice 1 — Comparer avec une quantification 8 bits

Rechargez `MODELE_LLM` avec `BitsAndBytesConfig(load_in_8bit=True)` et comparez l'empreinte
mémoire réelle (`get_memory_footprint()`) avec la version 4 bits.
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 1
"""))

cells.append(code('''\
import gc

del modele, pipe_llm
gc.collect()

config_8bit = BitsAndBytesConfig(load_in_8bit=True)
modele_8bit = charger_llm(config_8bit)
print(f"4 bits  : {empreinte_go:.2f} Go")
print(f"8 bits  : {modele_8bit.get_memory_footprint() / 1e9:.2f} Go")
if not GPU_DISPO:
    print("(Sans GPU, les deux valeurs sont identiques : la quantification n'a pas été appliquée.)")

del modele_8bit
gc.collect()
'''))

cells.append(md("""\
### 🏋️ Exercice 2 — Effet de la température sur le LLM

Redemandez `chat_llm(...)` d'inventer un slogan pour une coopérative agricole avec
`temperature=0.0` puis `temperature=1.0`. Le grand modèle réagit-il différemment du
SLM (notebook 01) au même réglage ?
"""))

cells.append(code('''\
# 👉 Votre code ici (rechargez pipe_llm si besoin, avec le modèle 4 bits par exemple)
'''))

cells.append(md("""\
### ✅ Solution 2
"""))

cells.append(code('''\
modele = charger_llm(config_4bit)
pipe_llm = pipeline("text-generation", model=modele, tokenizer=tokenizer)

for t in [0.0, 1.0]:
    print(f"--- température = {t} ---")
    print(chat_llm("Invente un slogan court pour une coopérative agricole.", temperature=t))
    print()
'''))

cells.append(md("""\
### 🏋️ Exercice 3 — Pourquoi le fp16 ne tient pas sur Colab

Sans charger le modèle, calculez la taille en Go d'un modèle de **13 milliards** de
paramètres en fp16, et comparez-la aux 16 Go de VRAM d'un GPU T4.
"""))

cells.append(code('''\
# 👉 Votre code ici
'''))

cells.append(md("""\
### ✅ Solution 3
"""))

cells.append(code('''\
n_params_13b = 13_000_000_000
go_fp16 = n_params_13b * 2 / 1e9
print(f"13 Md paramètres en fp16 → {go_fp16:.1f} Go (> 16 Go : ne tient pas sur un T4 gratuit)")
'''))

cells.append(md("""\
## ✅ Récapitulatif de l'atelier

| # | Notebook | Famille | Taille | Ce que vous avez appris |
|---|----------|---------|--------|--------------------------|
| 01 | SLM | ~1 Md | conseils textuels, few-shot, comparaison taille/vitesse |
| 02 | VLM | ~1 Md | décrire une image de plante, prompt engineering visuel |
| 03 | TinyLLM / TinyVLM | ≤ 0,3 Md | les modèles génératifs les plus petits |
| 04 | OV | ~150 M | détection ouverte, segmentation, tracking |
| 05 | LLM | ~9 Md | **quantification** et **API cloud (Groq)** pour un gros modèle sur Colab |

**Bravo ! 🌾** Vous savez maintenant choisir — et faire tourner — le bon modèle d'IA selon
la tâche et les ressources disponibles sur le terrain.
"""))

save(cells, "../notebooks/05_LLM_quantization.ipynb")
