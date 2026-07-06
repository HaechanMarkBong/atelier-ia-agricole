# 🌾 Atelier IA Agricole — SLM · LLM · TinyLLM/TinyVLM · VLM · OV vidéo

Une série de **notebooks Google Colab** (en français) pour découvrir des usages concrets de
l'intelligence artificielle appliquée à l'**agriculture**. **100 % gratuit**: aucun service
payant requis, tout passe par **Hugging Face**.

## 📚 Contenu

| # | Notebook | Famille de modèle | Taille du modèle | Ce que vous apprenez |
|---|----------|-------------------|-------------------|----------------------|
| 01 | `01_SLM_petits_modeles.ipynb` | **SLM** | ~1 Md paramètres | Petits modèles hors-ligne, comparaison taille/vitesse |
| 02 | `02_LLM_quantization.ipynb` | **LLM** | ~9 Md paramètres | Faire tenir un gros modèle sur Colab grâce à la **quantification** |
| 03 | `03_TinyLLM_TinyVLM.ipynb` | **TinyLLM / TinyVLM** | ~0,1–0,3 Md paramètres | Les plus petits modèles génératifs (texte + image), compromis taille/qualité |
| 04 | `04_VLM_vision.ipynb` | **VLM** | ~1 Md paramètres | Analyser une image de plante (description, VQA) |
| 05 | `05_OV_video_tracking.ipynb` | **OV vidéo** (*Grounded SAM 2*) | modèles « tiny » (~0,07–0,15 Md) | À partir d'un **mot**: boîte, **segmentation** et **suivi** d'une cible dans une **vidéo** |

Chaque notebook contient des **exercices** avec leurs **solutions**. Les cellules de chaque
notebook commencent par une configuration à exécuter en premier, et fonctionnent **de façon
indépendante** (vous pouvez ouvrir n'importe lequel sans avoir exécuté les autres).

## 🚀 Démarrage rapide (Google Colab)

1. Ouvrez [Google Colab](https://colab.research.google.com).
2. `Fichier` → `Importer le notebook` → onglet **Importer**, et déposez le fichier `.ipynb`.
  (Ou hébergez le dossier sur GitHub/Drive et ouvrez-le depuis Colab.)
3. *(Conseillé)* `Exécution` → `Modifier le type d'exécution` → **GPU** (T4, gratuit) —
  **nécessaire** pour le vrai LLM ~9 Md du notebook 02 (sans GPU, il bascule sur un petit modèle),
  et **fortement conseillé** pour la vidéo du notebook 05 (sur CPU, elle tourne mais lentement).
4. Commencez par le notebook **01**, puis suivez l'ordre 02 → 03 → 04 → 05.
5. Exécutez les cellules **de haut en bas** (`Maj + Entrée`).

## ⚙️ Le « mode démo »

Tout en haut de chaque notebook, une variable contrôle la taille des modèles / la charge de calcul:

```python
MODE_DEMO = os.environ.get("ATELIER_DEMO", "0") == "1"
```

- `MODE_DEMO = True` → **petits** modèles de substitution (et **moins d'images** en vidéo),
  exécution **rapide** (idéal sans GPU / pour tester automatiquement).
- `MODE_DEMO = False` (défaut) → modèles **complets** de l'atelier (~1 Md pour SLM/VLM, ~9 Md
  pour le LLM ; plus d'images échantillonnées pour la vidéo).

Pour forcer le mode démo lors d'un test automatique:
`os.environ["ATELIER_DEMO"] = "1"` **avant** la cellule de configuration.

## 🛠️ Outils utilisés (tous gratuits)

- [**Hugging Face Transformers**](https://huggingface.co) — chargement des SLM, LLM, VLM,
  TinyLLM/TinyVLM, de la détection ouverte et de SAM 2, via `pipeline(...)` ou les classes
  dédiées. Aucun compte requis pour les modèles utilisés.
- [**bitsandbytes**](https://github.com/bitsandbytes-foundation/bitsandbytes) — quantification
  4/8 bits du LLM (notebook 02) pour qu'il tienne sur un GPU Colab gratuit.
- **Grounding DINO + SAM 2** (Hugging Face) — détection ouverte par **prompt texte**, puis
  **segmentation** et **suivi** (tracking) d'une cible dans une **vidéo** (notebook 05).
- [**OpenCV**](https://opencv.org) — lecture et échantillonnage des images de la vidéo (notebook 05).
- [**Kaggle**](https://www.kaggle.com) — jeux de données agricoles (optionnel).

## ✅ Notes pédagogiques

- Les sorties des modèles sont des **aides**: un diagnostic agronomique doit toujours être
  confirmé par un·e professionnel·le.
- Les petits modèles répondent souvent mieux en **anglais** ; on peut traduire avec le SLM ou
  le LLM.
- Le notebook 02 explique **pourquoi** la quantification est nécessaire: un LLM de ~9 Md
  paramètres pèse ~35 Go en fp32, alors qu'un GPU Colab gratuit (T4) n'offre que 16 Go de VRAM.
- Le notebook 05 illustre **« Grounded SAM 2 »**: on écrit un mot (« cow »), **Grounding DINO**
  en déduit des boîtes, puis **SAM 2 vidéo** segmente et suit chaque cible d'image en image —
  aucune boîte n'est dessinée à la main.
