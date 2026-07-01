"""Helpers partagés pour générer les notebooks de l'atelier (nbformat v4)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


def save(cells, path):
    nb = new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"provenance": []},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("écrit:", path)


# Cellule de configuration commune insérée en haut de chaque notebook.
# Détecte Colab, expose MODE_DEMO (via variable d'env pour le test automatique)
# et un utilitaire d'installation de paquets silencieux.
CONFIG_CELL = '''\
# === Configuration de l'environnement (exécuter en premier) ===
import os, sys, subprocess

# Sommes-nous dans Google Colab ?
try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# MODE_DEMO = True  -> utilise de tout petits modèles / jeux de données réduits
# (utile pour tester rapidement, ou hors GPU). Mettez False pour l'atelier complet.
# La variable d'environnement ATELIER_DEMO permet de forcer le mode pour les tests.
MODE_DEMO = os.environ.get("ATELIER_DEMO", "0") == "1"

def pip_install(*paquets):
    """Installe des paquets pip de façon silencieuse (Colab ou local)."""
    cmd = [sys.executable, "-m", "pip", "install", "-q", *paquets]
    print("Installation :", " ".join(paquets), "...")
    subprocess.run(cmd, check=False)

print(f"IS_COLAB = {IS_COLAB}")
print(f"MODE_DEMO = {MODE_DEMO}")
print("Python :", sys.version.split()[0])
'''
