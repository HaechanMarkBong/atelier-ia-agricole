# 🌾 Atelier IA Agricole — SLM · VLM · TinyLLM/TinyVLM · OV · LLM

Une série de **notebooks Google Colab** (en français) pour découvrir des usages concrets de
l'intelligence artificielle appliquée à l'**agriculture**. **100 % gratuit**: aucun service
payant requis, tout passe par **Hugging Face**.

## 📚 Contenu

| # | Notebook | Famille de modèle | Taille du modèle | Ce que vous apprenez |
|---|----------|-------------------|-------------------|----------------------|
| 01 | `01_SLM_petits_modeles.ipynb` | **SLM** | ~1 Md paramètres | Petits modèles hors-ligne, comparaison taille/vitesse |
| 02 | `02_VLM_vision.ipynb` | **VLM** | ~1 Md paramètres | Analyser une image de plante (description, VQA) |
| 03 | `03_TinyLLM_TinyVLM.ipynb` | **TinyLLM / TinyVLM** | ~0,1–0,3 Md paramètres | Les plus petits modèles génératifs (texte + image), compromis taille/qualité |
| 04 | `04_OV_detection_ouverte.ipynb` | **OV** (détection ouverte) | ~150 M paramètres | Bounding box, segmentation et tracking |
| 05 | `05_LLM_quantization.ipynb` | **LLM** | ~9 Md paramètres | Faire tenir un gros modèle sur Colab grâce à la **quantification** |

Chaque notebook contient des **exercices** avec leurs **solutions**. Les cellules de chaque
notebook commencent par une configuration à exécuter en premier, et fonctionnent **de façon
indépendante** (vous pouvez ouvrir n'importe lequel sans avoir exécuté les autres).

## 🚀 Démarrage rapide (Google Colab)

1. Ouvrez [Google Colab](https://colab.research.google.com).
2. `Fichier` → `Importer le notebook` → onglet **Importer**, et déposez le fichier `.ipynb`.
  (Ou hébergez le dossier sur GitHub/Drive et ouvrez-le depuis Colab.)
3. *(Conseillé)* `Exécution` → `Modifier le type d'exécution` → **GPU** (T4, gratuit) —
  **nécessaire** pour le vrai LLM ~9 Md du notebook 05 (sans GPU, il bascule sur un petit modèle).
4. Commencez par le notebook **01**, puis suivez l'ordre 02 → 03 → 04 → 05.
5. Exécutez les cellules **de haut en bas** (`Maj + Entrée`).

## ⚙️ Le « mode démo »

Tout en haut de chaque notebook, une variable contrôle la taille des modèles:

```python
MODE_DEMO = os.environ.get("ATELIER_DEMO", "0") == "1"
```

- `MODE_DEMO = True` → **petits** modèles de substitution, exécution **rapide** (idéal sans
  GPU / pour tester automatiquement).
- `MODE_DEMO = False` (défaut) → modèles **complets** de l'atelier (~1 Md pour SLM/VLM, ~9 Md
  pour le LLM).

Pour forcer le mode démo lors d'un test automatique:
`os.environ["ATELIER_DEMO"] = "1"` **avant** la cellule de configuration.

## 🛠️ Outils utilisés (tous gratuits)

- [**Hugging Face Transformers**](https://huggingface.co) — chargement des SLM, VLM, TinyLLM/TinyVLM,
  détection ouverte et LLM, via `pipeline(...)`. Aucun compte requis pour les modèles utilisés.
- [**bitsandbytes**](https://github.com/bitsandbytes-foundation/bitsandbytes) — quantification
  4/8 bits du LLM (notebook 05) pour qu'il tienne sur un GPU Colab gratuit.
- [**OpenCV**](https://opencv.org) — segmentation simple et tracking (notebook 04).
- [**Kaggle**](https://www.kaggle.com) — jeux de données agricoles (optionnel).

## ✅ Notes pédagogiques

- Les sorties des modèles sont des **aides**: un diagnostic agronomique doit toujours être
  confirmé par un·e professionnel·le.
- Les petits modèles répondent souvent mieux en **anglais** ; on peut traduire avec le SLM ou
  le LLM.
- Le notebook 05 explique **pourquoi** la quantification est nécessaire: un LLM de ~9 Md
  paramètres pèse ~35 Go en fp32, alors qu'un GPU Colab gratuit (T4) n'offre que 16 Go de VRAM.
