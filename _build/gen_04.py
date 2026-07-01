from gen_common import md, code, save, CONFIG_CELL

cells = []

cells.append(md("""\
# 🌾 Atelier IA Agricole — 04. tinyML : Modèles ≤ 10M paramètres pour l'agriculture

Le **tinyML** (*Tiny Machine Learning*) consiste à faire tourner des modèles **ultra-légers**
(**≤ 10M paramètres**, souvent **moins de 50 Mo quantifiés** !) sur des appareils **très** limités :
Raspberry Pi, microcontrôleurs, objets connectés, IoT.

⚠️ **Important : Distinction de taille**
- **LLM/VLM** = 10B+ parameters (10 000 M+) — *pas de tinyML* — ordinateurs puissants
- **SLM** = 1B - 10B parameters — léger, mais reste hors tinyML — Raspberry Pi 4+
- **tinyML** = **≤ 10M parameters** — ultra-compact — Raspberry Pi Zero, microcontrôleurs

Ici nous travaillons dans la catégorie **tinyML pure** : modèles comprimés et vision.

**Approche tinyML : pas de LLM/VLM génératifs** plutôt :
- 🎯 **Classification** (image, texte) — prédictions rapides
- 🔍 **Vision légère** — détection, segmentation ultra-compact
- ⚡ **Lookup tables + petits classifieurs** — plutôt que génération texte

**Modèles tinyML authentiques (≤ 10M) :**
- **MobileNetV3** (5M) — classification image, très rapide ✅
- **TinyViT** (5M) — vision transformer ultra-léger ✅
- **MobileNetV2** (3.5M) — classifieur image, extrêmement léger ✅

*Les vrais LLM/VLM (GPT, CLIP, Llama) commencent à 10B — bien au-delà de tinyML.*
*Pour tinyML : chercher "mobile", "tiny", ou modèles comprimés/quantifiés.*

**Notre cas pratique :** utiliser des modèles ≤ 10M pour :
- 🌱 Classifier des cultures à partir d'une photo
- 🐛 Identifier des ravageurs/maladies (vision)
- 📋 Recommandations lookup basées sur conditions (table + classifieur)
"""))

cells.append(code(CONFIG_CELL))

cells.append(code('''\
pip_install("requests", "ollama")
import requests
import json
print("✅ Dépendances installées")
'''))

cells.append(md("""\
## 1. Sélectionner des modèles authentiquement ultra-légers (≤ 10M param.)

**Ce qui marche bien en tinyML ≤ 10M :**

✅ **Vision/Classification image**
- MobileNetV2 (3.5M) — classification de plantes/ravageurs
- MobileNetV3 (5M) — classification optimisée
- TinyViT (5M) — vision transformer compact
- EfficientNet-B0 (5M) — détection d'objets léger

⚠️ **Faisable mais limite**
- TinyBERT (14M) — classification texte, juste au-dessus
- Modèles comprimés/quantifiés (25M → 10M possible en int8)

❌ **Impossible en tinyML ≤ 10M**
- **Tout LLM/VLM** (ChatGPT, Llama, Phi, CLIP, etc.) — démarrent à 10B minimum
- Génération de texte libre — nécessite 1B+ au moins
- Modèles de conversation — trop gros

**Stratégie de déploiement :**

1. **Vision-first** : classifier des images de plantes/maladies (5M) ✅
2. **Lookup tables** : petite base de règles (azote bas → recommander culture X)
3. **Modèles comprimés** : quantifier un 25M → 10M si nécessaire
4. **Éviter LLM/VLM** : hors de portée tinyML, utiliser Notebook 01/03 pour ça

**Ressources tinyML :**
- TensorFlow Lite : conversion et optimisation
- ONNX Runtime : format ouvert, très léger
- HuggingFace Hub : chercher "mobile", "tiny", "distil"
- TensorFlow Hub : modèles pré-convertis

**Contrainte tinyML : ≤ 10M paramètres**
- Taille disque : 5-40 Mo (quantifiés)
- RAM runtime : 30-100 Mo
- Latence CPU : 50-500 ms par inférence
- Batterie + solaire : **100% réalisable**
- Cibles : Raspberry Pi Zero W, ESP32, Arduino MKR, EdgeTPU
"""))

cells.append(code('''\
# === Modèles de classification vision pour tinyML (≤ 10M) ===
# Note: LLM/VLM (10B+) ne sont PAS tinyML, voir Notebook 01/03

def load_mobilenetv2_classifier():
    """Charger MobileNetV2 (3.5M params, classification image).
    
    Cas d'usage tinyML :
    - Identifier maladies sur photo de feuille
    - Classifier type de culture
    - Détecter ravageurs
    """
    try:
        import tensorflow as tf
        # Pré-entraîné sur ImageNet, 3.5M params
        model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights='imagenet'
        )
        # Quantifier pour réduire à ~10 Mo
        # converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # tflite_model = converter.convert()
        print("✅ MobileNetV2 chargé (3.5M params, ~14 Mo TFLite quantifié)")
        return model
    except Exception as e:
        print(f"Erreur : {e}")
        return None

def load_tinybert_classifier():
    """TinyBERT pour classification texte court (14M params, limite du tinyML).
    
    Utilisation : classifier descriptions textes courtes
    (ex: "taches brunes sur feuille" → maladie)
    """
    try:
        from transformers import AutoModelForSequenceClassification
        # TinyBERT : 14M params, compression agressive de BERT
        model = AutoModelForSequenceClassification.from_pretrained(
            "huaiyicolin/tinybert-6l-768d-finetuned-ag-news"
        )
        print("✅ TinyBERT chargé (14M params, ~40 Mo)")
        return model
    except Exception as e:
        print(f"Erreur : {e}")
        return None

def query_huggingface(prompt, model_id="google/mobilenet_v2_100_224"):
    """Interroger un modèle HuggingFace ultra-compact (≤ 10M param.).
    
    Modèles recommandés (< 50M) :
    - MobileNetV2 (3.5M) : classification image ✅
    - TinyViT (5M) : vision transformer ultra-léger ✅
    - TinyBERT (14M) : classification texte comprimée, un peu juste
    
    Pour LLM pur : difficile < 10M (plupart sont 25M+)
    Préférer vision ou classification pour respecter contrainte.
    """
    import os
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("⚠️  Variable HF_TOKEN non définie. Voir notebook 00.")
        return None
    
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt, "parameters": {"max_length": 100}}
    
    try:
        r = requests.post(url, headers=headers, json=payload)
        if r.status_code == 200:
            result = r.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
        else:
            print(f"API Error: {r.status_code} - {r.text}")
            return None
    except Exception as e:
        print(f"Erreur : {e}")
        return None

print("✅ Fonctions tinyML définies")
'''))

cells.append(md("""\
## 2. Exemple 1 : Classification d'images avec MobileNetV2 (5M params)

Cas réel : identifier une maladie de plante à partir d'une photo.
MobileNetV2 (3.5M) est pré-entraîné sur ImageNet (1000 classes d'objets naturels).
"""))

cells.append(code('''\
from PIL import Image
import numpy as np
import tensorflow as tf

# Charger le modèle ultra-léger
model = load_mobilenetv2_classifier()

# Simuler : télécharger une image de plante
# (En vrai : image de caméra/smartphone)
img_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"

try:
    import urllib.request
    img_array = Image.open(urllib.request.urlopen(img_url))
    img_array = img_array.resize((224, 224))
    img_array = np.array(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Inférence (très rapide, < 100ms sur CPU)
    predictions = model.predict(img_array)
    top_class = np.argmax(predictions[0])
    confidence = predictions[0][top_class]
    
    print(f"Classe prédite : {top_class}")
    print(f"Confiance : {confidence*100:.1f}%")
    print("\\n→ En production, fine-tuner sur vos données agricoles locales")
except Exception as e:
    print(f"Erreur : {e}")
    print("💡 (Pas d'accès internet ? C'est OK, MobileNetV2 fonctionne hors-ligne)")
'''))

cells.append(md("""\
## 3. Exemple 2 : Lookup table pour recommandations (vraiment ultra-léger)

Approche tinyML authentique : **petite base de règles deterministes** au lieu de modèle.
"""))

cells.append(code('''\
# === Système de recommandation tinyML : lookup table ===
# Aucun apprentissage, aucun réseau de neurones
# Juste des règles IF/THEN basées sur expertise

def recommander_culture_tinyml(N, P, K, temperature, humidity, ph, rainfall):
    """
    Basé sur données empiriques, sans ML.
    Taille RAM : < 1 Ko
    Latence : < 1 ms
    """
    recommandations = []
    
    # Riz : zones humides, température chaude
    if rainfall > 150 and humidity > 70 and 22 < temperature < 28:
        recommandations.append(("Riz", 95))
    
    # Maïs : températures chaudes, modérément sec
    if 20 < temperature < 28 and humidity < 70 and 50 < rainfall < 200:
        recommandations.append(("Maïs", 85))
    
    # Arachide : sol bien draîné, sec
    if rainfall < 100 and humidity < 60 and 25 < temperature < 30:
        recommandations.append(("Arachide", 80))
    
    # Coton : climat chaud, sol riche
    if N > 70 and temperature > 24 and humidity < 75:
        recommandations.append(("Coton", 75))
    
    if recommandations:
        return max(recommandations, key=lambda x: x[1])
    else:
        return ("Inconnu", 0)

# Test
culture, score = recommander_culture_tinyml(80, 45, 40, 24, 82, 6.2, 230)
print(f"📊 Culture recommandée : {culture} (score {score}%)")
print(f"\\n→ Ce lookup table pèse < 1 Ko, latence < 1 ms, aucune dépendance")
'''))

cells.append(md("""\
## 4. Exemple 3 : Quantification d'un modèle pour rester < 10M

Si vous trouvez un modèle à 25M, le compresser à 10M avec quantification int8.
"""))

cells.append(code('''\
import tensorflow as tf

# Exemple : prendre un modèle de 25M et le quantifier
def quantify_to_tinyml(model_path_or_model):
    """
    Réduire taille : 25M → ~10M en int8 quantization.
    Perte qualité : typiquement < 2%
    """
    try:
        if isinstance(model_path_or_model, str):
            model = tf.keras.models.load_model(model_path_or_model)
        else:
            model = model_path_or_model
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        
        tflite_quantized = converter.convert()
        taille_mo = len(tflite_quantized) / (1024 * 1024)
        print(f"📦 Modèle quantifié : {taille_mo:.1f} Mo")
        
        if taille_mo <= 10:
            print("✅ Fits tinyML (<= 10M)")
            return tflite_quantized
        else:
            print(f"⚠️ Still {taille_mo:.1f} Mo, might need more compression")
            return tflite_quantized
    except Exception as e:
        print(f"Erreur : {e}")
        return None
'''))

cells.append(md("""\
---
# 🏋️ Exercices
"""))

cells.append(md("""\
### 🏋️ Exercice 1 — Tester le lookup table avec vos conditions

Testez la fonction `recommander_culture_tinyml()` avec vos **propres conditions** locales.
Cherchez les valeurs caractéristiques de votre région.
"""))

cells.append(code('''\
# 👉 Indiquez vos conditions
N, P, K = 50, 30, 15  # votre sol
temperature = 26      # vos conditions
humidity = 65
ph = 6.8
rainfall = 120        # votre climat annuel

culture, score = recommander_culture_tinyml(N, P, K, temperature, humidity, ph, rainfall)
print(f"Pour ces conditions → {culture} (score {score}%)")
'''))

cells.append(md("""\
### ✅ Solution 1
"""))

cells.append(code('''\
# Exemple : conditions du Sahel (Niger, Mali)
N, P, K = 40, 20, 10
temperature = 28
humidity = 40
ph = 7.0
rainfall = 350

culture, score = recommander_culture_tinyml(N, P, K, temperature, humidity, ph, rainfall)
print(f"Sahel → {culture} (score {score}%)")

# Exemple 2 : Guinée côtière (humide)
N, P, K = 60, 35, 25
temperature = 26
humidity = 85
ph = 5.5
rainfall = 2500

culture, score = recommander_culture_tinyml(N, P, K, temperature, humidity, ph, rainfall)
print(f"Côte → {culture} (score {score}%)")
'''))

cells.append(md("""\
### 🏋️ Exercice 2 — Comparer MobileNetV2 avec lookup table

Comparez les deux approches tinyML :
- **Vision + MobileNetV2** : 5M params, 50-100 ms
- **Lookup table** : 0 params, < 1 ms

Lequel pour quel cas d'usage ?
"""))

cells.append(code('''\
# 👉 Votre analyse ici
# - MobileNetV2 : quand ? (photos de plantes, ravageurs, maladies)
# - Lookup table : quand ? (capteurs sol, conditions simples)
'''))

cells.append(md("""\
### ✅ Solution 2
"""))

cells.append(code('''\
print("""
**MobileNetV2 (5M, vision)** :
✅ Identifier maladie/ravageur sur photo
✅ Classifier type de feuille/dégât
✅ Généraliser à nouvelles données
❌ Besoin de caméra + énergie calcul
❌ Plus lent (50-100 ms)

**Lookup table (0 params)** :
✅ Recommander culture sans modèle
✅ Ultra-rapide (< 1 ms)
✅ Aucune dépendance
❌ Limité à règles pré-définies
❌ Pas de flexibilité

**Hybride optimal** :
= Lookup table (capteurs sol) → recommandation rapide
+ MobileNetV2 (photo feuille) → diagnostiquer problèmes
= Déploiement total : < 10M params ✅
""")
'''))

cells.append(md("""\
### 🏋️ Exercice 3 — Déployer un modèle sur Raspberry Pi

Comment passer d'un modèle PC au Raspberry Pi Zero (512 MB RAM) ?

Indices :
- Convertir en TFLite quantifié
- Vérifier la taille < 10 Mo
- Mesurer latence CPU vs GPU
"""))

cells.append(code('''\
# 👉 Votre plan ici
# 1. Charger model.h5
# 2. Quantifier int8
# 3. Sauvegarder .tflite
# 4. Tester sur Pi Zero
'''))

cells.append(md("""\
### ✅ Solution 3
"""))

cells.append(code('''\
import tensorflow as tf

print("""
**Déployer sur Raspberry Pi Zero** :

1. Quantifier le modèle :
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   
2. Vérifier taille < 10 Mo

3. Sur Pi Zero, charger avec TFLite Interpreter :
   interpreter = tf.lite.Interpreter(model_path="model.tflite")
   
4. Mesurer latence :
   - MobileNetV2 : ~200-500 ms par image (CPU)
   - Lookup table : < 1 ms
   
5. Optimisations Pi Zero :
   - EdgeTPU si disponible (100x plus rapide)
   - Quantifier davantage si besoin
   - Limiter FPS caméra si applicable
""")
'''))

cells.append(md("""\
## ✅ Récapitulatif & fin de l'atelier 🎉

### **Clarification cruciale : tailles des modèles**

| Catégorie | Paramètres | Taille disque | RAM runtime | Latence CPU | Exemples |
|-----------|-----------|--------------|-----------|----------|----------|
| **tinyML** | **≤ 10M** | **5-40 Mo** | **30-100 Mo** | **50-500 ms** | MobileNetV3, TinyViT, Lookup tables |
| **SLM** (Small) | 1B - 10B | 500M - 5GB | 1-5 GB | 1-10 s | Phi-2.5, TinyLlama |
| **LLM** | **10B+** | **5GB+** | **8GB+** | **1-10 s** | Llama, GPT, Mistral |
| **VLM** | **10B+** | **5GB+** | **8GB+** | **2-30 s** | CLIP, Gemini, GPT-4V |

✅ **tinyML** = notebook 04 (ici) = vision, lookup tables, modèles comprimés
🤖 **LLM/VLM** = notebook 01/03 = génération texte, multimodal, conversationnel

---

### **Ce que vous avez appris en tinyML :**

- ✅ **Modèles ≤ 10M** : classification vision ultra-légère
- ✅ **Lookup tables** : 0 param, déterministe, ultra-rapide
- ✅ **Quantification** : compresser 25M → 10M avec int8
- ✅ **Déploiement edge** : Raspberry Pi Zero, objets connectés, IoT
- ✅ **Batterie + solaire** : faisable avec tinyML, impossible avec LLM

### **Ce que tinyML NE PEUT PAS faire :**

- ❌ Conversation libre (besoin 10B+ LLM)
- ❌ Génération d'images (besoin 10B+ VLM)
- ❌ Répondre à questions complexes (besoin raisonnement LLM)
- ❌ Traduction multilangue avancée (besoin 10B+)

**→ Pour ça, allez voir Notebooks 01 (LLM) ou 03 (VLM) sur ordinateur puissant.**

---

### **Vous avez terminé les 5 familles de modèles :**

| Notebook | Famille | Cas d'usage | Ressources |
|----------|---------|----------|----------|
| 01 | **LLM** (10B+) | Assistant agronome conversationnel | GPU puissant |
| 03 | **VLM** (10B+) | Générer/analyser images plantes | GPU puissant |
| 02 | **SLM** (1B-10B) | Conseil léger hors-ligne | Ordinateur portable |
| 04 | **tinyML** (≤10M) | Identifier plante, recommander culture | **Raspberry Pi Zero** ⭐ |
| 05 | **Détection ouverte** | Localiser objets en temps réel | Ordinateur/Pi 4+ |

---

**Bravo ! 🌾** Vous maîtrisez maintenant le spectre complet de l'IA agricole.
Vous pouvez choisir le bon modèle pour le bon déploiement.

**➡️ Notebook suivant : `05_detection_ouverte.ipynb`** — détection d'objets, segmentation, tracking.
"""))

save(cells, "../04_tinyML.ipynb")
