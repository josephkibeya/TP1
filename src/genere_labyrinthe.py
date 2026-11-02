"""
genere_labyrinthe.py
--------------------
Fabrique l'image du labyrinthe, au format attendu par le code du professeur :
murs en noir, zone de recherche en blanc, entree sur la ligne du haut et
sortie sur la ligne du bas.

LA CHAINE COMPLETE
==================
    genere_labyrinthe.py   ->  data/laby14_maze.png
    laplace_only.py        ->  data/laby14_pot.npy + laby14_domain.txt
    vitesses.py            ->  vitesses aux faces
    advection.py           ->  flux, bilan, pas de temps
    simulation_labyrinthe  ->  resultats/

COMMENT LA TAILLE A ETE CHOISIE
===============================
Le cout du calcul varie comme le CUBE du cote de l'image : le nombre de
mailles croit comme le carre, le nombre de pas comme la longueur du chemin.
Six tailles mesurees, pour les deux versions reunies :

    10 x 10, 153 px ->   8 s        16 x 16, 243 px ->  42 s
    12 x 12, 183 px ->  15 s        18 x 18, 273 px ->  94 s
    14 x 14, 213 px ->  22 s        20 x 20, 303 px -> 140 s

Le 14 x 14 a ete retenu : assez de couloirs pour que le cas soit riche,
assez petit pour etre relance souvent.

L'ALGORITHME
============
"Recursive backtracker" : on creuse vers une voisine non visitee prise au
hasard, et on recule quand on est bloque. Il produit un labyrinthe PARFAIT :
un seul chemin entre deux cases, aucune boucle, beaucoup d'impasses.

Les impasses sont voulues. L'eau y est a l'arret (le debit net par la seule
ouverture est forcement nul en regime permanent incompressible), et le
constituant est purement advecte. Toute encre qu'on y mesure est donc de la
diffusion numerique : les impasses servent d'instrument de mesure.

DETERMINISTE
============
La graine est fixee. Relancer ce fichier reecrit exactement la meme image,
octet pour octet, au meme endroit. On ne risque donc jamais de desynchroniser
l'image et les donnees calculees a partir d'elle.
"""

import os
import numpy as np
from PIL import Image

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DATA = os.path.join(ICI, "..", "data")


def labyrinthe_parfait(largeur, hauteur, graine=11):
    """
    Creuse un labyrinthe parfait.

    Retourne ouvert_bas[i, j] et ouvert_droite[i, j] : y a-t-il un passage
    vers la case du dessous, vers la case de droite.
    """
    rng = np.random.default_rng(graine)
    ouvert_bas = np.zeros((hauteur, largeur), bool)
    ouvert_droite = np.zeros((hauteur, largeur), bool)
    visitee = np.zeros((hauteur, largeur), bool)

    pile = [(0, 0)]
    visitee[0, 0] = True
    while pile:
        i, j = pile[-1]
        voisines = []
        if i > 0 and not visitee[i - 1, j]:
            voisines.append((i - 1, j, "haut"))
        if i < hauteur - 1 and not visitee[i + 1, j]:
            voisines.append((i + 1, j, "bas"))
        if j > 0 and not visitee[i, j - 1]:
            voisines.append((i, j - 1, "gauche"))
        if j < largeur - 1 and not visitee[i, j + 1]:
            voisines.append((i, j + 1, "droite"))

        if not voisines:
            pile.pop()                       # cul-de-sac : on recule
            continue

        ni, nj, sens = voisines[rng.integers(len(voisines))]
        if sens == "bas":
            ouvert_bas[i, j] = True
        elif sens == "haut":
            ouvert_bas[ni, nj] = True
        elif sens == "droite":
            ouvert_droite[i, j] = True
        else:
            ouvert_droite[ni, nj] = True
        visitee[ni, nj] = True
        pile.append((ni, nj))

    return ouvert_bas, ouvert_droite


def dessine(largeur, hauteur, ouvert_bas, ouvert_droite,
            couloir=12, mur=3, colonne_entree=3, colonne_sortie=10):
    """
    Transforme le labyrinthe en image noir et blanc.

    couloir : largeur d'un couloir en pixels. En dessous de 8 environ,
        l'ecoulement n'est plus assez bien resolu dans la largeur.
    mur : epaisseur d'un mur. Deux pixels suffisent a bloquer.
    """
    pas = couloir + mur
    image = np.zeros((hauteur * pas + mur, largeur * pas + mur), np.uint8)

    def blanc(i0, i1, j0, j1):
        image[i0:i1, j0:j1] = 255

    for i in range(hauteur):
        for j in range(largeur):
            y, x = mur + i * pas, mur + j * pas
            blanc(y, y + couloir, x, x + couloir)                  # la case
            if ouvert_droite[i, j]:
                blanc(y, y + couloir, x + couloir, x + couloir + mur)
            if ouvert_bas[i, j]:
                blanc(y + couloir, y + couloir + mur, x, x + couloir)

    blanc(0, mur, mur + colonne_entree * pas,
          mur + colonne_entree * pas + couloir)                     # entree
    blanc(image.shape[0] - mur, image.shape[0], mur + colonne_sortie * pas,
          mur + colonne_sortie * pas + couloir)                     # sortie
    return image


def fabrique(nom="laby14", cases=14, couloir=12, mur=3, graine=11,
             colonne_entree=3, colonne_sortie=10):
    """
    Genere l'image et l'enregistre dans ../data/<nom>_maze.png, quel que soit
    le dossier depuis lequel on lance le script.

    Le suffixe "_maze" evite un piege : compute(image, nom_sortie) du
    professeur ecrit un fichier <nom_sortie>.png, qui est la FIGURE du
    potentiel. Si l'image d'entree portait le meme nom, elle serait ecrasee.
    """
    os.makedirs(DOSSIER_DATA, exist_ok=True)
    image = dessine(cases, cases, *labyrinthe_parfait(cases, cases, graine),
                    couloir=couloir, mur=mur,
                    colonne_entree=colonne_entree, colonne_sortie=colonne_sortie)
    chemin = os.path.join(DOSSIER_DATA, nom + "_maze.png")
    Image.fromarray(np.dstack([image] * 3)).save(chemin)
    print("ecrit : data/%s_maze.png   %d x %d pixels"
          % (nom, image.shape[1], image.shape[0]))
    print("        %d x %d cases, couloir %d px, mur %d px"
          % (cases, cases, couloir, mur))
    return chemin


if __name__ == "__main__":
    fabrique()
    print()
    print("Etape suivante, avec le code du professeur, sans le modifier :")
    print("    import laplace_only")
    print("    laplace_only.compute('../data/laby14_maze.png', 'laby14')")
    print("Placer laby14_pot.npy et laby14_domain.txt dans ../data/ ,")
    print("puis lancer :  python simulation_labyrinthe.py")
