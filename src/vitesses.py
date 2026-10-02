"""
vitesses.py
-----------
Premier module du projet "Volumes Finis - Labyrinthe".

Objectif : a partir du champ de potentiel de vitesse (calcule par le code du
prof, laplace_only.py), calculer les vitesses NORMALES a chaque bord (face)
des mailles de la grille, puis VERIFIER que ce champ de vitesse est correct
(divergence nulle).

Rappel theorique (cf. "Enonce_partie_vf.pdf", note NB, et slide TP0 "Schema
differences finies") :
    - L'ecoulement est irrotationnel : la vitesse derive d'un potentiel phi
          u_i = d(phi)/d(x_i)
      (u dans la direction x, v dans la direction y)
    - On ne connait phi qu'aux CENTRES des mailles (une valeur par case de
      la grille). Mais pour les Volumes Finis, on aura besoin des vitesses
      aux BORDS des mailles (normales aux faces), pas aux centres.
    - Entre deux mailles voisines separees de dx (ou dy), on approxime la
      derivee au niveau du bord qui les separe par :

            d(phi)/dx  =  (phi[voisin] - phi[moi]) / dx

      Cette formule a l'air d'une difference "avant" (forward), mais vue
      depuis le BORD (qui est exactement au milieu entre les deux mailles),
      c'est en realite un schema CENTRE : elle est donc precise au second
      ordre par rapport a CE bord (voir TP0, diapo "Truncation error and
      order of accuracy", le cas i+1/2).

Convention matricielle du projet (voir README du prof) :
    - [0,0] est en haut a gauche
    - voisin de gauche de [i,j] : [i, j-1]      -> direction x (colonnes)
    - voisin de droite  de [i,j] : [i, j+1]      -> direction x (colonnes)
    - voisin du haut    de [i,j] : [i-1, j]      -> direction y (lignes)
    - voisin du bas     de [i,j] : [i+1, j]      -> direction y (lignes)

ATTENTION - piege important (a lire avant de faire tourner le code) :
----------------------------------------------------------------------
La formule ci-dessus n'a de sens que si les DEUX mailles de part et d'autre
du bord sont dans le domaine d'ecoulement (domain == 1). Si une des deux
mailles est un MUR (domain == 0), le potentiel y vaut 0 par defaut (ce n'est
pas une vraie valeur physique, juste un remplissage), et appliquer la
formule la-dessus donne une "vitesse" totalement fictive, comme si l'eau
traversait le mur. On le verifie plus bas avec `verifie_divergence_nulle`.
"""

import numpy as np


def vitesses_aux_bords(pot, dx=1.0, dy=1.0):
    """
    Calcule la vitesse normale a CHAQUE bord de la grille (y compris les
    bords qui touchent un mur - ceux-la seront a ignorer/filtrer ensuite).

    Parametres
    ----------
    pot : numpy.ndarray, shape (nI, nJ)
        Champ de potentiel de vitesse (une valeur par maille).
    dx, dy : float
        Taille d'une maille dans chaque direction (1.0 par defaut, car la
        grille vient directement des pixels de l'image : chaque maille
        = 1 pixel).

    Retourne
    --------
    u_bord : numpy.ndarray, shape (nI, nJ-1)
        Vitesse normale a chaque bord VERTICAL (= bord entre une maille et
        sa voisine de DROITE). u_bord[i, j] = bord entre [i, j] et [i, j+1].
    v_bord : numpy.ndarray, shape (nI-1, nJ)
        Vitesse normale a chaque bord HORIZONTAL (= bord entre une maille
        et sa voisine du BAS). v_bord[i, j] = bord entre [i, j] et [i+1, j].

    Notes sur le code Python (si c'est nouveau pour toi) :
    --------------------------------------------------------
    - `pot[:, 1:]`  veut dire : "toutes les lignes, toutes les colonnes
      SAUF la premiere" -> decale la vue d'une colonne vers la droite.
    - `pot[:, :-1]` veut dire : "toutes les colonnes SAUF la derniere".
    - `pot[:, 1:] - pot[:, :-1]` soustrait donc, colonne par colonne,
      chaque maille a sa voisine de droite, pour TOUS les bords verticaux
      EN UNE SEULE OPERATION (pas de boucle `for` : c'est la
      "vectorisation" numpy, beaucoup plus rapide qu'une boucle Python).
    - Meme logique pour les lignes avec `pot[1:, :]` et `pot[:-1, :]`.
    """
    u_bord = (pot[:, 1:] - pot[:, :-1]) / dx
    v_bord = (pot[1:, :] - pot[:-1, :]) / dy
    return u_bord, v_bord


def verifie_divergence_nulle(u_bord, v_bord, domain, dx=1.0, dy=1.0, tol=1e-9):
    """
    Verifie que du/dx + dv/dy = 0 (ecoulement incompressible), UNIQUEMENT
    sur les mailles "pleinement interieures" : celles qui sont dans le
    domaine (domain == 1) ET dont les 4 voisines le sont aussi.

    Pourquoi se restreindre a ces mailles-la ?
    -------------------------------------------
    Comme explique en haut du fichier, la formule de vitesse aux bords n'a
    de sens que si les deux mailles de part et d'autre d'un bord sont dans
    le domaine. Une maille qui touche un mur a donc au moins un bord dont
    la "vitesse" est fictive -> la divergence calculee la-dessus n'a aucune
    raison d'etre nulle, et ce n'est PAS un bug : c'est juste que ce n'est
    pas une maille a tester. On ne garde donc que les mailles entourees de
    4 vraies voisines "ecoulement" (domain == 1), la ou la formule est 100%
    valide des deux cotes.

    Parametres
    ----------
    u_bord, v_bord : sortie de `vitesses_aux_bords`
    domain : numpy.ndarray, meme forme que pot (0 = mur, 1 = ecoulement,
        2 = condition limite entree/sortie)
    tol : float
        Seuil en dessous duquel on considere la divergence comme nulle.

    Retourne
    --------
    divergence : numpy.ndarray
        Divergence calculee sur chaque maille interieure (meme hors du
        masque "pleinement interieure", pour que tu puisses l'afficher et
        voir OU elle n'est pas nulle si tu veux explorer par toi-meme).
    masque_interieur : numpy.ndarray (bool)
        True la ou la maille est "pleinement interieure" (donc la ou la
        divergence DOIT etre nulle).
    ok : bool
        True si la divergence max (en valeur absolue), restreinte au
        masque, est en dessous de `tol`.
    """
    # Divergence "brute", calculee partout (y compris pres des murs, ou
    # elle n'a pas de sens physique - voir docstring).
    du_dx = (u_bord[1:-1, 1:] - u_bord[1:-1, :-1]) / dx
    dv_dy = (v_bord[1:, 1:-1] - v_bord[:-1, 1:-1]) / dy
    divergence = du_dx + dv_dy

    # Masque : la maille elle-meme et ses 4 voisines doivent valoir 1.
    centre = domain[1:-1, 1:-1]
    haut   = domain[0:-2, 1:-1]
    bas    = domain[2:,   1:-1]
    gauche = domain[1:-1, 0:-2]
    droite = domain[1:-1, 2:]
    masque_interieur = (centre == 1) & (haut == 1) & (bas == 1) & (gauche == 1) & (droite == 1)

    div_max = np.max(np.abs(divergence[masque_interieur]))
    ok = bool(div_max < tol)
    return divergence, masque_interieur, ok


if __name__ == "__main__":
    # Ce bloc ne s'execute que si on lance CE fichier directement
    # (pas si on fait "import vitesses" depuis un autre script).
    import os

    ici = os.path.dirname(os.path.abspath(__file__))
    chemin_pot = os.path.join(ici, "..", "data", "out_pot.npy")
    chemin_dom = os.path.join(ici, "..", "data", "out_domain.txt")

    pot = np.load(chemin_pot)
    domain = np.loadtxt(chemin_dom)

    print("Potentiel charge, taille :", pot.shape)

    u_bord, v_bord = vitesses_aux_bords(pot)
    print("Taille u_bord (bords verticaux)   :", u_bord.shape)
    print("Taille v_bord (bords horizontaux) :", v_bord.shape)

    divergence, masque, ok = verifie_divergence_nulle(u_bord, v_bord, domain)
    print("Nombre de mailles verifiees (pleinement interieures) :", masque.sum())
    print("Divergence max (valeur absolue) sur ces mailles       :", np.max(np.abs(divergence[masque])))
    print("Divergence nulle partout ou ca compte ? ->", ok)

    # Bonus pedagogique : montre que pres des murs, la formule brute donne
    # bien une divergence NON nulle (normal, voir explications ci-dessus).
    touche_mur = (domain[1:-1, 1:-1] == 1) & ~masque
    if touche_mur.sum() > 0:
        print()
        print("Pour comparaison, sur les", touche_mur.sum(), "mailles qui touchent un mur :")
        print("Divergence max (valeur absolue) :", np.max(np.abs(divergence[touche_mur])))
        print("-> grand et normal : la formule brute invente une vitesse a travers le mur.")
