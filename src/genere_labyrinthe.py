"""
genere_labyrinthe.py
--------------------
Fabrique une image de labyrinthe au format attendu par le code du professeur.

POURQUOI CE FICHIER
===================
Le solveur du professeur prend une image : murs en noir, zone de recherche en
blanc, entree sur la ligne du haut, sortie sur la ligne du bas. Rien n'oblige
a utiliser le labyrinthe fourni en exemple. Generer le sien permet :

    - de choisir la taille, donc le temps de calcul ;
    - de verifier que le moteur ne depend pas d'un labyrinthe particulier ;
    - de disposer de plusieurs jeux de donnees, comme l'enonce le demande.

COMMENT ON CHOISIT LA TAILLE
============================
Le cout total du calcul varie a peu pres comme le CUBE du cote de l'image :

    - le nombre de mailles croit comme le carre du cote ;
    - le nombre de pas de temps croit comme la longueur du chemin.

Mesure sur la machine de developpement, pour les deux versions reunies :

    10 x 10 cases, 153 px  ->    8 s        16 x 16, 243 px ->  42 s
    12 x 12 cases, 183 px  ->   15 s        18 x 18, 273 px ->  94 s
    14 x 14 cases, 213 px  ->   22 s        20 x 20, 303 px -> 140 s

Le 14 x 14 a ete retenu : assez de couloirs pour que le cas soit riche, assez
petit pour tourner en quelques minutes et pouvoir etre relance souvent.

L'ALGORITHME
============
"Recursive backtracker" : on part d'une case, on creuse vers une voisine non
encore visitee choisie au hasard, et on recule quand on est bloque. Il produit
un labyrinthe PARFAIT : un seul chemin entre deux cases quelconques, aucune
boucle, et beaucoup d'impasses.

Les impasses sont voulues. Dans un cul-de-sac, l'eau est a l'arret : elle
devrait en ressortir par la seule ouverture, ce qui est impossible en regime
permanent incompressible. Comme le constituant est purement advecte, aucune
encre ne devrait s'y trouver, et toute encre qu'on y mesure est de la
diffusion numerique. Les impasses servent donc d'instrument de mesure.
"""

import numpy as np
from PIL import Image


def labyrinthe_parfait(largeur, hauteur, graine=11):
    """
    Creuse un labyrinthe parfait par "recursive backtracker".

    Retourne
    --------
    ouvert_bas[i, j]    : True s'il y a un passage vers la case du dessous
    ouvert_droite[i, j] : True s'il y a un passage vers la case de droite
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

    couloir : largeur d'un couloir, en pixels. En dessous de 8 environ,
        l'ecoulement n'est plus assez bien resolu dans la largeur.
    mur : epaisseur d'un mur, en pixels. Deux suffisent a bloquer, trois
        donnent une image plus nette.

    L'entree est ouverte sur la ligne du haut et la sortie sur la ligne du
    bas, comme le code du professeur l'exige :
        "L'entree du labyrinthe doit etre situee sur la ligne superieure
         et la sortie sur la ligne inferieure."
    """
    pas = couloir + mur
    hauteur_px = hauteur * pas + mur
    largeur_px = largeur * pas + mur
    image = np.zeros((hauteur_px, largeur_px), np.uint8)    # tout noir

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
    blanc(hauteur_px - mur, hauteur_px, mur + colonne_sortie * pas,
          mur + colonne_sortie * pas + couloir)                     # sortie
    return image


def fabrique(nom="laby14", cases=14, couloir=12, mur=3, graine=11,
             colonne_entree=3, colonne_sortie=10):
    """Genere l'image et l'enregistre. Retourne le chemin du fichier."""
    ob, od = labyrinthe_parfait(cases, cases, graine)
    image = dessine(cases, cases, ob, od, couloir, mur,
                    colonne_entree, colonne_sortie)
    fichier = nom + "_maze.png"
    Image.fromarray(np.dstack([image, image, image])).save(fichier)
    print("%s : %d x %d pixels   (%d x %d cases, couloir %d px, mur %d px)"
          % (fichier, image.shape[1], image.shape[0], cases, cases, couloir, mur))
    return fichier


# ATTENTION, PIEGE A CONNAITRE
# ============================
# La fonction compute(chemin_image, nom_sortie) du professeur ecrit un fichier
# <nom_sortie>.png : la FIGURE du potentiel. Si l'image d'entree porte le meme
# nom, elle est ECRASEE. C'est pour cela que l'image du labyrinthe s'appelle
# ici "<nom>_maze.png" et les sorties "<nom>_..." : aucun nom ne se recouvre.

if __name__ == "__main__":
    fichier = fabrique()
    print()
    print("Etape suivante, avec le code du professeur, sans le modifier :")
    print("    import laplace_only")
    print("    laplace_only.compute('%s', 'laby14')" % fichier)
    print()
    print("Cela produit laby14_pot.npy et laby14_domain.txt, a placer dans")
    print("../data/ . Le module simulation_labyrinthe les lira avec")
    print("    etude_complete(nom_laby='laby14')")
