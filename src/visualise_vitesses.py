"""
visualise_vitesses.py
---------------------
Fait une IMAGE du champ de vitesse calcule par `vitesses.py`, pour verifier
a l'oeil que l'eau suit bien les couloirs de l'entree (haut) vers la sortie (bas).

Ce script ne modifie aucun fichier existant : il lit les resultats du prof
(data/out_pot.npy, data/out_domain.txt) et ecrit seulement data/champ_vitesse.png.

Lancer :  python src/visualise_vitesses.py

Deux panneaux :
  - a gauche : tout le labyrinthe. La COULEUR donne la vitesse (la norme,
    sqrt(u^2 + v^2)) et de petites fleches donnent la DIRECTION.
  - a droite : zoom autour de l'entree, avec plus de fleches.
Les murs sont en gris.

Pourquoi un calcul "au centre des mailles" ?
  Les vitesses de `vitesses.py` sont calculees sur les BORDS des mailles.
  Pour dessiner une fleche par maille, on prend la moyenne des bords de la
  maille ou la formule est valable (les deux mailles voisines sont dans le
  domaine). Les bords qui touchent un mur sont ignores (vitesse fictive).
"""

import os

import matplotlib

matplotlib.use("Agg")  # on enregistre une image, pas de fenetre
import matplotlib.pyplot as plt
import numpy as np

from vitesses import vitesses_aux_bords


def vitesse_aux_centres(u_bord, v_bord, domain):
    """
    Ramene les vitesses des bords vers les centres des mailles (pour le dessin).

    u_bord[i, j] est le bord entre [i, j] et [i, j+1]  (forme (nI, nJ-1))
    v_bord[i, j] est le bord entre [i, j] et [i+1, j]  (forme (nI-1, nJ))

    Retourne uc, vc de la forme de `domain`. Pour chaque maille on fait la
    moyenne de ses bords VALIDES (maille voisine aussi dans le domaine).
    """
    fluide = domain > 0  # 1 = interieur, 2 = entree/sortie

    # un bord est valide si les deux mailles qu'il separe sont dans le fluide
    u_ok = fluide[:, 1:] & fluide[:, :-1]
    v_ok = fluide[1:, :] & fluide[:-1, :]

    nI, nJ = domain.shape
    u_somme = np.zeros((nI, nJ))
    u_nb = np.zeros((nI, nJ))
    v_somme = np.zeros((nI, nJ))
    v_nb = np.zeros((nI, nJ))

    # bord a DROITE de la maille [i, j] : u_bord[i, j]
    u_somme[:, :-1] += np.where(u_ok, u_bord, 0.0)
    u_nb[:, :-1] += u_ok
    # bord a GAUCHE de la maille [i, j] : u_bord[i, j-1]
    u_somme[:, 1:] += np.where(u_ok, u_bord, 0.0)
    u_nb[:, 1:] += u_ok

    # bord en BAS de la maille [i, j] : v_bord[i, j]
    v_somme[:-1, :] += np.where(v_ok, v_bord, 0.0)
    v_nb[:-1, :] += v_ok
    # bord en HAUT de la maille [i, j] : v_bord[i-1, j]
    v_somme[1:, :] += np.where(v_ok, v_bord, 0.0)
    v_nb[1:, :] += v_ok

    uc = np.divide(u_somme, u_nb, out=np.zeros_like(u_somme), where=u_nb > 0)
    vc = np.divide(v_somme, v_nb, out=np.zeros_like(v_somme), where=v_nb > 0)
    uc[~fluide] = 0.0
    vc[~fluide] = 0.0
    return uc, vc


def dessine_fleches(ax, uc, vc, fluide, pas, i0=0, j0=0):
    """Fleches de LONGUEUR FIXE (on ne montre que la direction) tous les `pas` pixels."""
    nI, nJ = uc.shape
    ii, jj = np.meshgrid(np.arange(i0, nI, pas), np.arange(j0, nJ, pas), indexing="ij")
    u = uc[ii, jj]
    v = vc[ii, jj]
    norme = np.hypot(u, v)
    garde = fluide[ii, jj] & (norme > 1e-12)
    # dans l'image, x = colonne j (vers la droite), y = ligne i (vers le BAS)
    ax.quiver(
        jj[garde],
        ii[garde],
        (u / np.where(norme > 0, norme, 1))[garde],
        (v / np.where(norme > 0, norme, 1))[garde],
        angles="xy",
        scale_units="xy",
        scale=1.0 / pas * 1.2,
        width=0.004,
        color="white",
        edgecolor="black",
        linewidth=0.3,
    )


if __name__ == "__main__":
    ici = os.path.dirname(os.path.abspath(__file__))
    dossier_data = os.path.join(ici, "..", "data")

    pot = np.load(os.path.join(dossier_data, "out_pot.npy"))
    domain = np.loadtxt(os.path.join(dossier_data, "out_domain.txt")).astype(int)

    u_bord, v_bord = vitesses_aux_bords(pot)
    uc, vc = vitesse_aux_centres(u_bord, v_bord, domain)

    fluide = domain > 0
    vitesse = np.hypot(uc, vc)
    vitesse_aff = np.where(fluide, vitesse, np.nan)  # murs = NaN -> gris

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.35")  # murs en gris fonce

    # zone autour de l'entree (premiere ligne ou domain == 2)
    cols_entree = np.where(domain[1, :] == 2)[0]
    jc = int(cols_entree.mean())
    demi = 30
    j_min, j_max = max(jc - demi, 0), min(jc + demi, domain.shape[1])
    i_min, i_max = 0, min(2 * demi, domain.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- panneau 1 : tout le labyrinthe ---
    ax = axes[0]
    vmax = np.nanpercentile(vitesse_aff, 99)  # evite qu'un point extreme ecrase la couleur
    im = ax.imshow(vitesse_aff, cmap=cmap, vmin=0, vmax=vmax)
    dessine_fleches(ax, uc, vc, fluide, pas=12, i0=6, j0=6)
    ax.set_title("Champ de vitesse : couleur = vitesse, fleches = direction")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="vitesse (unites arbitraires, par pixel)")
    ax.plot([j_min, j_max, j_max, j_min, j_min], [i_min, i_min, i_max, i_max, i_min], "r-", lw=1)
    ax.text(j_max + 3, i_min + 8, "zoom", color="red")

    # --- panneau 2 : zoom sur l'entree ---
    ax = axes[1]
    ax.imshow(vitesse_aff, cmap=cmap, vmin=0, vmax=vmax)
    dessine_fleches(ax, uc, vc, fluide, pas=3, i0=i_min, j0=j_min)
    ax.set_xlim(j_min, j_max)
    ax.set_ylim(i_max, i_min)  # ligne 0 en haut
    ax.set_title("Zoom autour de l'entree (haut) : l'eau doit descendre")

    fig.tight_layout()
    sortie = os.path.join(dossier_data, "champ_vitesse.png")
    fig.savefig(sortie, dpi=150)
    print("Image ecrite :", sortie)
