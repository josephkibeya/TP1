"""
advection.py
------------
Deuxieme module du projet "Volumes Finis - Labyrinthe".

Objectif : transporter un constituant (de l'encre) dans le champ de vitesse
calcule par `vitesses.py`, en resolvant par VOLUMES FINIS l'equation de
l'enonce :

        dc/dt + u dc/dx + v dc/dy = 0        (forme non conservative)
        dc/dt + d(uc)/dx + d(vc)/dy = 0      (forme CONSERVATIVE, celle qu'on code)

Les deux formes sont equivalentes parce que l'ecoulement est incompressible
(div V = 0). C'est le NB de l'enonce. On code la forme conservative parce
qu'elle garantit la conservation exacte de la masse (sommes telescopiques).

LA CHAINE DE CALCUL
-------------------
Ce fichier ne contient QUE le moteur de calcul. Les verifications sont dans
`test_advection_canal_rectiligne.py`, a cote.

    1. calcule_dt           -> le pas de temps, impose par le nombre de Courant
    2. pentes               -> les deux gradients par maille (0 si constante)
    3. valeurs_aux_faces    -> les DEUX candidats de chaque face
    4. flux_upwind          -> UN SEUL flux par face (on garde l'amont)
    5. bilan                -> un nombre par maille : dc/dt
    6. pas_euler            -> la nouvelle concentration

Les etapes 2 a 5 forment une "machine a bilan". Euler l'appelle une fois ;
Runge-Kutta l'appellera deux fois (module suivant).

CONVENTION DES TABLEAUX (la meme que vitesses.py)
--------------------------------------------------
    c       : (nI, nJ)      une valeur par maille, au centre de gravite
    u_bord  : (nI, nJ-1)    u_bord[i, j] = face entre [i, j] et [i, j+1]
    v_bord  : (nI-1, nJ)    v_bord[i, j] = face entre [i, j] et [i+1, j]
    domain  : (nI, nJ)      0 = mur, 1 = ecoulement, 2 = entree/sortie

    u > 0 : vers la DROITE      v > 0 : vers le BAS (les lignes i augmentent
                                        vers le bas)

LES DEUX VERSIONS DEMANDEES PAR LE PROFESSEUR
----------------------------------------------
    Version A : reconstruction CONSTANTE + Euler explicite  -> ordre 1
    Version B : reconstruction LINEAIRE  + Runge-Kutta 2    -> ordre 2

On les apparie parce que l'erreur totale vaut min(ordre en temps, ordre en
espace) : melanger les deux coutera plus cher sans rien gagner.
Dans ce module, l'interrupteur `lineaire` fait passer de l'une a l'autre.
"""

import numpy as np


# =====================================================================
#  ETAPE 1 : le pas de temps
# =====================================================================

def calcule_dt(u_bord, v_bord, domain, dx=1.0, dy=1.0, marge=0.8):
    """
    Calcule le pas de temps le plus grand qui respecte la condition de
    Courant, puis lui applique une marge de securite.

    La condition (schema explicite decentre amont, en 2D) s'ecrit :

        |u| dt / dx  +  |v| dt / dy  <=  1

    On la retourne pour en tirer dt :

        dt_max = 1 / ( |u|max / dx  +  |v|max / dy )

    POURQUOI LE MAXIMUM
    -------------------
    Dans le cours, "a" est UNE vitesse, la meme partout. Ici chaque face a
    la sienne (plus de 400 000 faces). La condition doit tenir sur CHAQUE
    face : une seule qui deborde fait exploser tout le calcul. On prend donc
    la pire, c'est-a-dire la plus rapide. Si elle passe, toutes passent.

    PIEGE IMPORTANT
    ---------------
    On ne regarde que les faces VALIDES, celles qui ont du fluide des deux
    cotes. Les 42 faces d'entree/sortie portent encore la valeur fictive
    -100 (voir vitesses.py). Si on les laissait entrer dans le maximum, on
    obtiendrait dt = 0.005 au lieu de 1.38, soit 276 fois plus de pas de
    calcul pour une valeur qui n'a aucun sens physique.

    Parametres
    ----------
    u_bord, v_bord : sortie de vitesses.vitesses_aux_bords (murs deja annules)
    domain : 0 = mur, 1 = ecoulement, 2 = entree/sortie
    dx, dy : taille d'une maille (1 pixel par defaut)
    marge : fraction de dt_max que l'on garde. 0.8 = 20 % de securite.
            CE CHIFFRE EST UN CHOIX, pas une regle du cours : a justifier
            dans le rapport.

    Retourne
    --------
    dt : float, le pas de temps a utiliser
    infos : dict, pour l'afficher et le mettre dans le rapport
    """
    fluide = domain > 0

    # Une face est valide si ses DEUX mailles voisines sont du fluide.
    # (meme logique que annule_vitesses_murs, voir vitesses.py)
    valide_u = fluide[:, 1:] & fluide[:, :-1]
    valide_v = fluide[1:, :] & fluide[:-1, :]

    u_max = np.abs(u_bord[valide_u]).max() if valide_u.any() else 0.0
    v_max = np.abs(v_bord[valide_v]).max() if valide_v.any() else 0.0

    denominateur = u_max / dx + v_max / dy
    if denominateur == 0.0:
        raise ValueError("Toutes les vitesses sont nulles : dt serait infini.")

    dt_max = 1.0 / denominateur
    dt = marge * dt_max

    infos = {
        "u_max": u_max,
        "v_max": v_max,
        "dt_max": dt_max,
        "marge": marge,
        "dt": dt,
        "courant": (u_max / dx + v_max / dy) * dt,   # doit valoir `marge`
        "faces_u_valides": int(valide_u.sum()),
        "faces_v_valides": int(valide_v.sum()),
    }
    return dt, infos


# =====================================================================
#  ETAPE 2 : les pentes (c'est ici que constante et lineaire different)
# =====================================================================

def pentes(c, domain, lineaire, dx=1.0, dy=1.0):
    """
    Calcule les deux gradients directionnels de chaque maille.

    RECONSTRUCTION CONSTANTE  (lineaire = False)
        On suppose la concentration PLATE dans la maille. Il n'y a pas de
        pente : on renvoie deux tableaux de zeros.

    RECONSTRUCTION LINEAIRE  (lineaire = True)
        On suppose une variation "bi-lineaire" : une droite dans chaque
        direction, separement. Le professeur ecrit "gradients directionnels
        a evaluer sur base des 2 mailles voisines" (diapo 10) :

            sx[i,j] = ( c[i, j+1] - c[i, j-1] ) / (2 dx)
            sy[i,j] = ( c[i+1, j] - c[i-1, j] ) / (2 dy)

        Le 2 au denominateur vient de ce que les deux voisines sont separees
        de DEUX mailles.

    LES BORNES : TOUJOURS EN CONSTANTE
    ----------------------------------
    "Domaine limite -> on traite les bornes en reconstruction constante"
    (diapo 15). Une maille qui n'a pas ses DEUX voisines dans le fluide ne
    peut pas avoir de pente : on la met a zero dans cette direction-la.
    Les deux directions sont independantes : une maille collee a un mur a
    gauche garde sa pente verticale si ses voisines du haut et du bas
    existent.

    Retourne
    --------
    sx, sy : deux tableaux de la taille de c
    """
    sx = np.zeros_like(c)
    sy = np.zeros_like(c)

    if not lineaire:
        return sx, sy          # version constante : pentes nulles, c'est tout

    fluide = domain > 0

    # --- pente horizontale, colonnes 1 .. nJ-2 ---
    # Les colonnes 0 et nJ-1 restent a zero : ce sont les bornes du tableau.
    voisines_ok = fluide[:, 2:] & fluide[:, :-2]       # les DEUX voisines fluides ?
    sx[:, 1:-1] = np.where(voisines_ok, (c[:, 2:] - c[:, :-2]) / (2.0 * dx), 0.0)

    # --- pente verticale, lignes 1 .. nI-2 ---
    voisines_ok = fluide[2:, :] & fluide[:-2, :]
    sy[1:-1, :] = np.where(voisines_ok, (c[2:, :] - c[:-2, :]) / (2.0 * dy), 0.0)

    return sx, sy


# =====================================================================
#  ETAPE 3 : les DEUX candidats de chaque face
# =====================================================================

def valeurs_aux_faces(c, sx, sy, dx=1.0, dy=1.0):
    """
    Pour chaque face, calcule les DEUX valeurs proposees par ses deux
    mailles voisines.

    "Un bord dispose de 2 informations discontinues (une en provenance de la
    maille situee a sa gauche, l'autre de la maille situee a sa droite)"
    (diapo 3 du professeur).

    On part du centre et on avance d'un DEMI-pas jusqu'au bord :

        bord droit  d'une maille : c + (dx/2) sx
        bord gauche d'une maille : c - (dx/2) sx
        bord bas    d'une maille : c + (dy/2) sy     (i+1 est en bas)
        bord haut   d'une maille : c - (dy/2) sy

    Avec sx = sy = 0 (reconstruction constante), les quatre valent c : on
    recopie simplement la valeur de la maille.

    Retourne
    --------
    cL, cR : les deux candidats de chaque face VERTICALE,  taille (nI, nJ-1)
             cL vient de la maille de GAUCHE, cR de celle de DROITE
    cH, cB : les deux candidats de chaque face HORIZONTALE, taille (nI-1, nJ)
             cH vient de la maille du HAUT,   cB de celle du BAS
    """
    bord_droit  = c + (dx / 2.0) * sx
    bord_gauche = c - (dx / 2.0) * sx
    bord_bas    = c + (dy / 2.0) * sy
    bord_haut   = c - (dy / 2.0) * sy

    # Face verticale j : entre la maille j (a gauche) et la maille j+1 (a droite).
    # Le candidat de gauche est le BORD DROIT de la maille de gauche.
    cL = bord_droit[:, :-1]
    cR = bord_gauche[:, 1:]

    # Face horizontale i : entre la maille i (en haut) et la maille i+1 (en bas).
    cH = bord_bas[:-1, :]
    cB = bord_haut[1:, :]

    return cL, cR, cH, cB


# =====================================================================
#  ETAPE 4 : UN SEUL flux par face (decentrage amont)
# =====================================================================

def flux_upwind(cL, cR, cH, cB, u_bord, v_bord):
    """
    Choisit, pour chaque face, le candidat d'AMONT, puis le multiplie par la
    vitesse de la face.

    "Le flux a un bord doit pourtant etre unique" et "la stabilite du schema
    est assuree si le flux de bord est decentre vers l'amont" (diapo 3).

    LA REGLE (Partie 10, diapo 25)
    ------------------------------
        u > 0 (l'eau va vers la droite) -> on lit la maille de GAUCHE
        u < 0 (l'eau va vers la gauche) -> on lit la maille de DROITE

    L'ECRITURE SANS "if"
    --------------------
        F = 1/2 (u + |u|) c_gauche  +  1/2 (u - |u|) c_droite

    Verification :
        si u > 0 : u + |u| = 2u  et  u - |u| = 0   -> F = u c_gauche
        si u < 0 : u + |u| = 0   et  u - |u| = 2u  -> F = u c_droite

    Un seul terme survit a chaque fois. Cette ecriture traite les deux sens
    d'un coup, sur tout le tableau, sans boucle.

    POURQUOI PAS L'AUTRE COTE
    -------------------------
    Prendre le mauvais cote (decentrage AVAL) donne un facteur
    d'amplification de 1 + 2*nu au mode zigzag, strictement superieur a 1
    pour tout nu > 0 : le schema explose quel que soit le pas de temps.
    Le schema centre est instable aussi (max |G| = racine(1 + nu^2) > 1).

    Retourne
    --------
    Fu : flux sur chaque face verticale,   taille (nI, nJ-1)
    Fv : flux sur chaque face horizontale, taille (nI-1, nJ)
    """
    Fu = 0.5 * (u_bord + np.abs(u_bord)) * cL + 0.5 * (u_bord - np.abs(u_bord)) * cR
    Fv = 0.5 * (v_bord + np.abs(v_bord)) * cH + 0.5 * (v_bord - np.abs(v_bord)) * cB
    return Fu, Fv


# =====================================================================
#  ETAPE 5 : le bilan des quatre faces
# =====================================================================

def bilan(c, domain, u_bord, v_bord, lineaire=False, dx=1.0, dy=1.0):
    """
    Calcule dc/dt pour chaque maille : c'est le L(c) du document sur Euler.

    LA FORMULE (Partie 2, diapo 7)
    ------------------------------
        L[i,j] = - ( F_droite - F_gauche ) / dx  - ( F_bas - F_haut ) / dy

    Le signe moins vient de ce que le bilan compte "entree - sortie" : ce qui
    SORT fait BAISSER la concentration.

    LE REMPLISSAGE DE ZEROS AU POURTOUR
    -----------------------------------
    Les mailles du bord du tableau n'ont pas de face a l'exterieur. On ajoute
    donc une colonne (resp. une ligne) de flux nuls de chaque cote : rien
    n'entre ni ne sort par le pourtour du tableau. Dans le labyrinthe, ce
    pourtour est de toute facon un mur ajoute par le solveur du prof.

    LES MURS
    --------
    Il n'y a rien de special a faire : les faces de mur ont deja une vitesse
    nulle (annule_vitesses_murs), donc leur flux vaut zero et elles ne
    comptent pas dans le bilan.

    Retourne
    --------
    L : tableau de la taille de c
    """
    sx, sy = pentes(c, domain, lineaire, dx, dy)
    cL, cR, cH, cB = valeurs_aux_faces(c, sx, sy, dx, dy)
    Fu, Fv = flux_upwind(cL, cR, cH, cB, u_bord, v_bord)

    nI, nJ = c.shape

    # On entoure les flux d'une bande de zeros pour pouvoir soustraire
    # "face de droite moins face de gauche" sur TOUTES les mailles d'un coup.
    Fu_etendu = np.zeros((nI, nJ + 1))
    Fu_etendu[:, 1:-1] = Fu
    Fv_etendu = np.zeros((nI + 1, nJ))
    Fv_etendu[1:-1, :] = Fv

    #  Fu_etendu[:, 1:]  = flux de la face de DROITE de chaque maille
    #  Fu_etendu[:, :-1] = flux de la face de GAUCHE de chaque maille
    L = -(Fu_etendu[:, 1:] - Fu_etendu[:, :-1]) / dx \
        - (Fv_etendu[1:, :] - Fv_etendu[:-1, :]) / dy
    return L


# =====================================================================
#  ETAPE 6 : avancer d'un pas de temps
# =====================================================================

def pas_euler(c, domain, u_bord, v_bord, dt, lineaire=False, dx=1.0, dy=1.0):
    """
    Un pas d'Euler explicite :

        c_nouveau = c + dt * L(c)

    C'est la serie de Taylor en temps, coupee apres le terme en dt. Une seule
    evaluation du bilan par pas. Ordre 1 en temps.

    ATTENTION : a n'utiliser qu'avec la reconstruction CONSTANTE.
    Avec la reconstruction lineaire, ce schema est INSTABLE pour tout nu > 0
    (voir la note sur les quatre combinaisons, plus bas).
    """
    return c + dt * bilan(c, domain, u_bord, v_bord, lineaire, dx, dy)


def pas_rk2(c, domain, u_bord, v_bord, dt, lineaire=True, dx=1.0, dy=1.0):
    """
    Un pas de Runge-Kutta a 2 etapes (methode de Heun), en quatre gestes :

        1. k1 = L(c)                      la pente au point de depart
        2. c* = c + dt k1                 un pas d'essai (un BROUILLON)
        3. k2 = L(c*)                     la pente au point d'essai
        4. c_nouveau = c + dt (k1+k2)/2   le vrai pas, avec la moyenne

    "Un schema Runge-Kutta peut etre vu comme une double application d'un
    schema de type Euler" (diapo 17 du professeur).

    Deux evaluations du bilan par pas, pour l'ordre 2 en temps. Le pas
    d'essai c* n'est jamais conserve : il sert uniquement a aller lire une
    seconde pente.
    """
    k1 = bilan(c, domain, u_bord, v_bord, lineaire, dx, dy)
    essai = c + dt * k1
    k2 = bilan(essai, domain, u_bord, v_bord, lineaire, dx, dy)
    return c + dt * (k1 + k2) / 2.0


# =====================================================================
#  LES CONDITIONS AUX LIMITES "FAIBLES"
# =====================================================================

def applique_conditions_limites(c, domain, c_exterieur=0.0):
    """
    Impose la concentration dans toutes les mailles HORS domaine
    (domain == 0 : les murs et le contour ajoute par le solveur du prof).

    POURQUOI C'EST NECESSAIRE
    -------------------------
    Le bilan de la derniere maille du domaine utilise sa face de sortie. Le
    flux de cette face est lu en amont, donc dans la maille de fluide : la
    matiere s'en va. Mais dans le tableau, elle atterrit dans la maille du
    contour, qui n'appartient pas au domaine. Si on ne la remet pas a zero,
    elle s'y accumule et le bilan de masse devient faux.

    De la meme facon, a l'entree, la maille du contour est l'AMONT de la
    premiere face : sa valeur est ce qui entre dans le domaine. La mettre a
    zero, c'est faire entrer de l'eau PROPRE.

    C'est ce que le professeur appelle une condition limite "faible"
    (diapos 8 et 10) : on n'impose pas la valeur d'une maille du domaine, on
    impose ce qui traverse sa frontiere.

    Parametres
    ----------
    c_exterieur : la concentration de ce qui entre. 0.0 = eau propre.

    Retourne
    --------
    c : le champ corrige (une copie, l'original n'est pas modifie)
    masse_retiree : la quantite qui se trouvait hors domaine, c'est-a-dire
                    ce qui est SORTI pendant ce pas. Sert au controle de la
                    conservation.
    """
    c = c.copy()
    dehors = (domain == 0)
    masse_retiree = c[dehors].sum() - c_exterieur * dehors.sum()
    c[dehors] = c_exterieur
    return c, masse_retiree
