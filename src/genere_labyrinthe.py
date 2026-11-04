"""
genere_labyrinthe.py
--------------------
Produit l'image du labyrinthe au format attendu par le solveur fourni :
murs en noir, zone de recherche en blanc, entree sur la ligne du haut et
sortie sur la ligne du bas.

CHAINE COMPLETE
===============
    genere_labyrinthe.py   ->  data/laby14_maze.png
    laplace_only.py        ->  data/laby14_pot.npy + laby14_domain.txt
    vitesses.py            ->  vitesses aux faces
    advection.py           ->  flux, bilan, pas de temps
    simulation_labyrinthe  ->  resultats/

CHOIX DE LA TAILLE
==================
Le cout du calcul varie comme le cube du cote de l'image : le nombre de
mailles croit comme le carre, le nombre de pas comme la longueur du chemin.
Six tailles ont ete mesurees, pour les deux versions reunies :

    10 x 10, 153 px ->   8 s        16 x 16, 243 px ->  42 s
    12 x 12, 183 px ->  15 s        18 x 18, 273 px ->  94 s
    14 x 14, 213 px ->  22 s        20 x 20, 303 px -> 140 s

La taille 14 x 14 a ete retenue : elle offre un reseau de couloirs
suffisamment developpe tout en restant assez petite pour autoriser des
executions repetees.

ALGORITHME
==========
Exploration en profondeur avec retour arriere : le creusement progresse vers
une voisine non visitee tiree au hasard et recule des qu'il est bloque.
L'algorithme produit un labyrinthe parfait, c'est-a-dire un unique chemin
entre deux cases, sans boucle et comportant de nombreuses impasses.

Ces impasses sont recherchees. Le fluide y est au repos, le debit net par
leur unique ouverture etant necessairement nul en regime permanent
incompressible, et le constituant est purement advecte. Toute masse mesuree
dans une impasse releve donc de la diffusion numerique : les impasses
constituent a ce titre un instrument de mesure.

CARACTERE DETERMINISTE
======================
La graine du generateur aleatoire est fixee. Une nouvelle execution reecrit
exactement la meme image, octet pour octet, au meme emplacement, ce qui
exclut toute desynchronisation entre l'image et les donnees calculees a
partir d'elle.
"""

import os
import numpy as np
from PIL import Image

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DATA = os.path.join(ICI, "..", "data")


def labyrinthe_parfait(largeur, hauteur, graine=11):
    """
    Creuse un labyrinthe parfait.

    Retourne ouvert_bas[i, j] et ouvert_droite[i, j], qui indiquent
    l'existence d'un passage vers la case inferieure et vers la case de
    droite.
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
            pile.pop()                       # impasse : retour arriere
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
    Convertit le labyrinthe en image noir et blanc.

    couloir : largeur d'un couloir, en pixels. Une maille correspondant a un
        pixel, cette valeur est aussi le nombre de mailles sur la largeur du
        couloir. Plus elle est grande, mieux l'ecoulement est resolu
        transversalement et plus le calcul est long. Aucune etude de
        convergence en maillage n'a ete menee sur le labyrinthe : la valeur
        retenue ne resulte donc pas d'un critere de precision etabli.
    mur : epaisseur d'un mur, en pixels. Deux pixels suffisent a interdire le
        passage.
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
    le repertoire d'appel du script.

    Le suffixe "_maze" est necessaire : l'appel compute(image, nom_sortie) du
    solveur ecrit un fichier <nom_sortie>.png contenant la figure du
    potentiel. Sans ce suffixe, l'image d'entree portant le meme nom serait
    ecrasee.
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
    print("Etape suivante, avec le solveur fourni, sans le modifier :")
    print("    import laplace_only")
    print("    laplace_only.compute('../data/laby14_maze.png', 'laby14')")
    print("Placer laby14_pot.npy et laby14_domain.txt dans ../data/ ,")
    print("puis lancer :  python simulation_labyrinthe.py")
