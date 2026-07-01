# 🌾 Atelier IA Agricole — LLM · SLM · VLM · tinyML · détection ouverte

Une série de **notebooks Google Colab** (en français) pour découvrir des usages concrets de
l'intelligence artificielle appliquée à l'**agriculture**. **100 % gratuit** : aucun service
payant requis.

## 📚 Contenu

| # | Notebook | Famille de modèle | Ce que vous apprenez |
|---|----------|-------------------|----------------------|
| 01 | `01_LLM_ollama.ipynb` | **LLM** | Assistant agronome, prompt, température |
| 02 | `02_SLM_petits_modeles.ipynb` | **SLM** | Petits modèles hors-ligne, comparaison taille/vitesse |
| 03 | `03_VLM_vision.ipynb` | **VLM** | Analyser une image de plante avec un modèle vision-langage |
| 04 | `04_tinyML.ipynb` | **tinyML** | Modèles ≤ 10M paramètres, classification légère, déploiement embarqué |
| 05 | `05_detection_ouverte.ipynb` | Détection ouverte | Bounding box, segmentation et tracking |

Chaque notebook contient des **exercices** avec leurs **solutions**. Les cellules de chaque
notebook commencent par une configuration à exécuter en premier.

## 🚀 Démarrage rapide (Google Colab)

1. Ouvrez [Google Colab](https://colab.research.google.com).
2. `Fichier` → `Importer le notebook` → onglet **Importer**, et déposez le fichier `.ipynb`.
  (Ou hébergez le dossier sur GitHub/Drive et ouvrez-le depuis Colab.)
3. *(Optionnel mais conseillé)* `Exécution` → `Modifier le type d'exécution` → **GPU** (T4, gratuit).
4. Commencez par le notebook **01**, puis suivez l'ordre 02 → 03 → 04 → 05.
5. Exécutez les cellules **de haut en bas** (`Maj + Entrée`).

## ⚠️ Bon à savoir

Le notebook 01 contient une cellule de configuration pour préparer l'environnement avant
l'exécution. Les notebooks sont conçus pour fonctionner en mode démo ou avec les modèles
complètement chargés selon les ressources disponibles.

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
