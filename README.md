# 🌾 Atelier IA Agricole — LLM · SLM · VLM · tinyML

Une série de **notebooks Google Colab** (en français) pour débuter avec l'intelligence
artificielle appliquée à l'**agriculture**. **100 % gratuit** : aucun service payant requis.

## 📚 Contenu

| # | Notebook | Famille de modèle | Ce que vous apprenez |
|---|----------|-------------------|----------------------|
| 00 | `00_introduction_et_installation.ipynb` | — | Concepts, installation d'Ollama, HuggingFace, Kaggle |
| 01 | `01_LLM_ollama.ipynb` | **LLM** | Assistant agronome, prompt, température |
| 03 | `03_VLM_vision.ipynb` | **VLM** | Analyser une image de plante et générer une image |
| 02 | `02_SLM_petits_modeles.ipynb` | **SLM** | Petits modèles hors-ligne, comparaison taille/vitesse |
| 04 | `04_tinyML.ipynb` | **tinyML** | Recommander une culture, modèle `.tflite` de quelques Ko |
| 05 | `05_detection_ouverte.ipynb` | Détection ouverte | Bounding box, segmentation et tracking |

Chaque notebook contient des **exercices** avec leurs **solutions**.

## 🚀 Démarrage rapide (Google Colab)

1. Ouvrez [Google Colab](https://colab.research.google.com).
2. `Fichier` → `Importer le notebook` → onglet **Importer**, et déposez le fichier `.ipynb`.
   (Ou hébergez le dossier sur GitHub/Drive et ouvrez-le depuis Colab.)
3. *(Optionnel mais conseillé)* `Exécution` → `Modifier le type d'exécution` → **GPU** (T4, gratuit).
4. Commencez par le notebook **00**, puis suivez l'ordre 01 → 03 → 02 → 04 → 05.
5. Exécutez les cellules **de haut en bas** (`Maj + Entrée`).

## 🔑 Jeton Kaggle (facultatif)

Pour télécharger des jeux de données depuis Kaggle :

1. Compte gratuit sur [kaggle.com](https://www.kaggle.com).
2. **Settings** → section **API** → **Create New Token** → télécharge `kaggle.json`.
3. Le notebook 00 vous propose de téléverser ce fichier.

> Sans jeton, **tout fonctionne quand même** : chaque notebook prévoit une source de
> données **de secours sans authentification**.

## ⚙️ Le « mode démo »

Tout en haut de chaque notebook, une variable contrôle la taille des modèles :

```python
MODE_DEMO = os.environ.get("ATELIER_DEMO", "0") == "1"
```

- `MODE_DEMO = True` → **petits** modèles, exécution **rapide** (idéal sans GPU / pour tester).
- `MODE_DEMO = False` (défaut) → modèles **complets** de l'atelier (`llama3.2`, etc.).

Pour forcer le mode démo lors d'un test automatique :
`os.environ["ATELIER_DEMO"] = "1"` **avant** la cellule de configuration.

## 🛠️ Outils utilisés (tous gratuits)

- [**Ollama**](https://ollama.com) — exécuter des LLM/SLM localement.
- [**HuggingFace Transformers**](https://huggingface.co) — modèles de vision (VLM).
- [**OpenCV**](https://opencv.org) — segmentation simple et tracking.
- [**TensorFlow Lite**](https://www.tensorflow.org/lite) — tinyML.
- [**Kaggle**](https://www.kaggle.com) — jeux de données agricoles.

## ✅ Notes pédagogiques

- Les sorties des modèles sont des **aides** : un diagnostic agronomique doit toujours être
  confirmé par un·e professionnel·le.
- Les petits modèles répondent souvent mieux en **anglais** ; on peut traduire avec un LLM.
