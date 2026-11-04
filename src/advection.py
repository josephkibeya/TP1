"""
advection.py
------------
Deuxieme module du projet "Volumes Finis - Labyrinthe".

Transport d'un constituant dans le champ de vitesse calcule par
`vitesses.py`, par resolution en volumes finis de l'equation de l'enonce :

        dc/dt + u dc/dx + v dc/dy = 0        (forme non conservative)
        dc/dt + d(uc)/dx + d(vc)/dy = 0      (forme conservative)

Les deux formes sont equivalentes ici parce que l'ecoulement est
incompressible (div V = 0). La forme conservative est retenue : elle
garantit la conservation exacte de la masse discrete, les flux internes
s'annulant deux a deux dans la somme des bilans.

CHAINE DE CALCUL
----------------
Ce module contient le moteur de calcul seul ; les verifications sont
regroupees dans `test_advection_canal_rectiligne.py`.

    1. calcule_dt           pas de temps admissible (condition de Courant)
    2. pentes               les deux gradients par maille (nuls si constante)
    3. valeurs_aux_faces    les deux valeurs candidates de chaque face
    4. flux_upwind          un flux unique par face, decentre vers l'amont
    5. bilan                dc/dt pour chaque maille
    6. pas_euler / pas_rk2  avancement d'un pas de temps

Les etapes 2 a 5 constituent l'evaluation du bilan L(c). Euler l'appelle
une fois par pas de temps, Runge-Kutta deux fois.

CONVENTION DES TABLEAUX (identique a vitesses.py)
--------------------------------------------------
    c       : (nI, nJ)      une valeur par maille, au centre de gravite
    u_bord  : (nI, nJ-1)    u_bord[i, j] = face entre [i, j] et [i, j+1]
    v_bord  : (nI-1, nJ)    v_bord[i, j] = face entre [i, j] et [i+1, j]
    domain  : (nI, nJ)      0 = mur, 1 = ecoulement, 2 = entree/sortie

    u > 0 : vers la droite      v > 0 : vers le bas, l'indice de ligne i
                                croissant vers le bas

VERSIONS ETUDIEES
-----------------
    Version A : reconstruction constante + Euler explicite   ordre 1
    Version B : reconstruction lineaire  + Runge-Kutta 2      ordre 2

Les ordres en temps et en espace sont apparies : l'erreur globale est
gouvernee par le plus faible des deux, de sorte qu'un appariement croise
augmenterait le cout de calcul sans ameliorer la precision. L'argument
`lineaire` commande le choix de la reconstruction.
"""

import numpy as np


# =====================================================================
#  ETAPE 1 : le pas de temps
# =====================================================================

def calcule_dt(u_bord, v_bord, domain, dx=1.0, dy=1.0, marge=0.8):
    """
    Determine le pas de temps le plus grand compatible avec la condition de
    Courant, puis lui applique une marge de securite.

    La condition de stabilite du schema explicite decentre amont s'ecrit, en
    deux dimensions,

        |u| dt / dx  +  |v| dt / dy  <=  1

    d'ou le pas de temps limite

        dt_max = 1 / ( |u|max / dx  +  |v|max / dy )

    Les deux contributions directionnelles s'additionnent : la condition est
    donc plus restrictive qu'en une dimension.

    Le maximum porte sur l'ensemble des faces. La vitesse n'est pas uniforme
    ici, chaque face ayant la sienne, et la condition doit etre satisfaite
    partout : une seule face qui la viole suffit a faire diverger le calcul.
    Le pas de temps est donc dicte par la face la plus rapide.

    Seules les faces valides, celles qui separent deux mailles fluides, sont
    retenues. Les faces d'entree et de sortie portent, avant correction, la
    valeur fictive -100 issue d'une difference entre un potentiel reel et le
    zero de remplissage du contour (voir vitesses.py). Les inclure ramenerait
    dt de 0.693 a 0.008 sur le labyrinthe laby14, soit 87 fois plus de pas de
    calcul pour une valeur depourvue de sens physique.

    Parametres
    ----------
    u_bord, v_bord : sortie de vitesses.vitesses_aux_bords (murs deja annules)
    domain : 0 = mur, 1 = ecoulement, 2 = entree/sortie
    dx, dy : dimensions d'une maille (un pixel par defaut)
    marge : fraction de dt_max conservee. La condition de stabilite impose
            seulement nu < 1 ; la valeur 0.8 retenue ici releve d'un choix,
            qui laisse 20 % de securite.

    Retourne
    --------
    dt : float, le pas de temps a utiliser
    infos : dict des grandeurs caracteristiques du pas de temps
    """
    fluide = domain > 0

    # Une face est valide si ses deux mailles voisines sont fluides
    # (meme critere que annule_vitesses_murs, dans vitesses.py).
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
        "courant": (u_max / dx + v_max / dy) * dt,   # vaut `marge` par construction
        "faces_u_valides": int(valide_u.sum()),
        "faces_v_valides": int(valide_v.sum()),
    }
    return dt, infos


# =====================================================================
#  ETAPE 2 : les pentes (reconstruction constante ou lineaire)
# =====================================================================

def pentes(c, domain, lineaire, dx=1.0, dy=1.0):
    """
    Calcule les deux gradients directionnels de chaque maille.

    RECONSTRUCTION CONSTANTE  (lineaire = False)
        La concentration est supposee uniforme dans la maille. Les pentes
        sont nulles : deux tableaux de zeros sont renvoyes.

    RECONSTRUCTION LINEAIRE  (lineaire = True)
        La concentration varie lineairement dans chaque direction, prise
        separement. Chaque gradient est estime par difference centree sur les
        deux mailles voisines :

            sx[i,j] = ( c[i, j+1] - c[i, j-1] ) / (2 dx)
            sy[i,j] = ( c[i+1, j] - c[i-1, j] ) / (2 dy)

        Le facteur 2 au denominateur traduit l'ecart de deux mailles entre
        les deux voisines.

    TRAITEMENT DES BORDS
    --------------------
    Le domaine etant borne, les mailles qui ne disposent pas de leurs deux
    voisines fluides sont traitees en reconstruction constante : leur pente
    est annulee dans la direction concernee. Les deux directions restent
    independantes, de sorte qu'une maille bordee d'un mur a sa gauche
    conserve sa pente verticale si ses voisines du haut et du bas existent.

    Retourne
    --------
    sx, sy : deux tableaux de la taille de c
    """
    sx = np.zeros_like(c)
    sy = np.zeros_like(c)

    if not lineaire:
        return sx, sy          # reconstruction constante : pentes nulles

    fluide = domain > 0

    # --- pente horizontale, colonnes 1 a nJ-2 ---
    # Les colonnes 0 et nJ-1 restent nulles : la difference centree y est
    # impossible, faute d'une voisine de chaque cote.
    voisines_ok = fluide[:, 2:] & fluide[:, :-2]       # les deux voisines fluides ?
    sx[:, 1:-1] = np.where(voisines_ok, (c[:, 2:] - c[:, :-2]) / (2.0 * dx), 0.0)

    # --- pente verticale, lignes 1 .. nI-2 ---
    voisines_ok = fluide[2:, :] & fluide[:-2, :]
    sy[1:-1, :] = np.where(voisines_ok, (c[2:, :] - c[:-2, :]) / (2.0 * dy), 0.0)

    return sx, sy


# =====================================================================
#  ETAPE 3 : les deux valeurs candidates de chaque face
# =====================================================================

def valeurs_aux_faces(c, sx, sy, dx=1.0, dy=1.0):
    """
    Evalue, pour chaque face, les deux valeurs proposees par les mailles qui
    l'encadrent.

    La reconstruction etant propre a chaque maille, un bord dispose de deux
    informations en general discontinues : l'une provenant de la maille
    situee a sa gauche, l'autre de la maille situee a sa droite. Chacune
    s'obtient en partant du centre de gravite et en avancant d'un demi-pas
    jusqu'au bord :

        bord droit  d'une maille : c + (dx/2) sx
        bord gauche d'une maille : c - (dx/2) sx
        bord bas    d'une maille : c + (dy/2) sy     (l'indice i+1 est en bas)
        bord haut   d'une maille : c - (dy/2) sy

    En reconstruction constante (sx = sy = 0) les quatre valeurs se reduisent
    a c : la valeur de la maille est simplement recopiee sur ses bords.

    Retourne
    --------
    cL, cR : les deux candidats de chaque face verticale,   taille (nI, nJ-1)
             cL provient de la maille de gauche, cR de celle de droite
    cH, cB : les deux candidats de chaque face horizontale, taille (nI-1, nJ)
             cH provient de la maille du haut,   cB de celle du bas
    """
    bord_droit  = c + (dx / 2.0) * sx
    bord_gauche = c - (dx / 2.0) * sx
    bord_bas    = c + (dy / 2.0) * sy
    bord_haut   = c - (dy / 2.0) * sy

    # Face verticale j : entre la maille j (a gauche) et la maille j+1 (a
    # droite). Le candidat de gauche est le bord droit de la maille de gauche.
    cL = bord_droit[:, :-1]
    cR = bord_gauche[:, 1:]

    # Face horizontale i : entre la maille i (en haut) et la maille i+1 (en bas).
    cH = bord_bas[:-1, :]
    cB = bord_haut[1:, :]

    return cL, cR, cH, cB


# =====================================================================
#  ETAPE 4 : le flux unique de chaque face (decentrage amont)
# =====================================================================

def flux_upwind(cL, cR, cH, cB, u_bord, v_bord):
    """
    Retient, pour chaque face, la valeur reconstruite du cote amont, et la
    multiplie par la vitesse de la face.

    Le flux a un bord doit etre unique, sous peine de rompre la conservation,
    et la stabilite du schema est assuree si ce flux de bord est decentre
    vers l'amont :

        u > 0, l'ecoulement va vers la droite -> maille de gauche
        u < 0, l'ecoulement va vers la gauche -> maille de droite

    Ce choix s'ecrit sans branchement conditionnel :

        F = 1/2 (u + |u|) c_gauche  +  1/2 (u - |u|) c_droite

    En effet, si u > 0 : u + |u| = 2u et u - |u| = 0, d'ou F = u c_gauche ;
    si u < 0 : u + |u| = 0 et u - |u| = 2u, d'ou F = u c_droite. Un seul
    terme subsiste dans chaque cas. Cette forme traite les deux sens
    d'ecoulement simultanement, sur l'ensemble du tableau et sans boucle.

    Les deux autres decentrages sont instables. Le decentrage aval conduit a
    un facteur d'amplification de 1 + 2 nu sur le mode de plus courte
    longueur d'onde, strictement superieur a 1 pour tout nu > 0 : le schema
    diverge quel que soit le pas de temps. Le schema centre donne de meme
    max |G| = racine(1 + nu^2), superieur a 1.

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
    Evalue dc/dt pour chaque maille, c'est-a-dire l'operateur L(c) auquel les
    schemas temporels sont appliques :

        L[i,j] = - ( F_droite - F_gauche ) / dx  - ( F_bas - F_haut ) / dy

    Le signe negatif traduit le bilan "entrees moins sorties" : un flux
    sortant fait decroitre la concentration de la maille.

    Les mailles du pourtour du tableau n'ont pas de face exterieure. Une
    colonne, respectivement une ligne, de flux nuls est donc ajoutee de
    chaque cote, ce qui revient a interdire tout echange par le pourtour.
    Dans le labyrinthe, ce pourtour est de toute facon un mur ajoute par le
    solveur fourni.

    Les faces de mur ne demandent aucun traitement particulier : leur vitesse
    a deja ete annulee par annule_vitesses_murs, leur flux est donc nul et
    elles ne contribuent pas au bilan.

    Retourne
    --------
    L : tableau de la taille de c
    """
    sx, sy = pentes(c, domain, lineaire, dx, dy)
    cL, cR, cH, cB = valeurs_aux_faces(c, sx, sy, dx, dy)
    Fu, Fv = flux_upwind(cL, cR, cH, cB, u_bord, v_bord)

    nI, nJ = c.shape

    # Les flux sont entoures d'une bande de zeros, ce qui permet de former la
    # difference "face de droite moins face de gauche" sur toutes les mailles
    # a la fois.
    Fu_etendu = np.zeros((nI, nJ + 1))
    Fu_etendu[:, 1:-1] = Fu
    Fv_etendu = np.zeros((nI + 1, nJ))
    Fv_etendu[1:-1, :] = Fv

    #  Fu_etendu[:, 1:]  : flux de la face de droite de chaque maille
    #  Fu_etendu[:, :-1] : flux de la face de gauche de chaque maille
    L = -(Fu_etendu[:, 1:] - Fu_etendu[:, :-1]) / dx \
        - (Fv_etendu[1:, :] - Fv_etendu[:-1, :]) / dy
    return L


# =====================================================================
#  ETAPE 6 : avancer d'un pas de temps
# =====================================================================

def pas_euler(c, domain, u_bord, v_bord, dt, lineaire=False, dx=1.0, dy=1.0):
    """
    Avancement d'un pas par la methode d'Euler explicite :

        c_nouveau = c + dt * L(c)

    soit le developpement de Taylor en temps tronque apres le terme en dt.
    Une seule evaluation du bilan par pas, pour un ordre 1 en temps.

    Ce schema est a employer avec la reconstruction constante. Associe a la
    reconstruction lineaire, il est instable pour tout nu > 0, ce qui motive
    le recours a Runge-Kutta dans la version B.
    """
    return c + dt * bilan(c, domain, u_bord, v_bord, lineaire, dx, dy)


def pas_rk2(c, domain, u_bord, v_bord, dt, lineaire=True, dx=1.0, dy=1.0,
            conditions=None):
    """
    Avancement d'un pas par la methode de Runge-Kutta a deux etapes (methode
    de Heun) :

        1. k1 = L(c)                      pente au point de depart
        2. c* = c + dt k1                 etat intermediaire
        3. k2 = L(c*)                     pente a l'etat intermediaire
        4. c_nouveau = c + dt (k1+k2)/2   pas effectif, par la moyenne

    Un schema de Runge-Kutta se lit ainsi comme une double application d'un
    schema de type Euler. Deux evaluations du bilan par pas donnent l'ordre 2
    en temps. L'etat intermediaire c* n'est pas conserve : il ne sert qu'a
    evaluer une seconde pente.

    ROLE DE L'ARGUMENT `conditions`
    -------------------------------
    L'etape 2 fait avancer l'ensemble du tableau, y compris les mailles
    situees hors domaine (domain == 0) qui fournissent la valeur amont a
    l'entree. Or la concentration de ces mailles n'est pas une inconnue mais
    une donnee du probleme : eau propre, ou eau chargee a 100 dans le cas
    d'une injection continue. Laisser l'etat intermediaire la modifier
    conduit l'etape 3 a lire une frontiere erronee. L'etat intermediaire peut
    s'ecarter de la solution a l'interieur du domaine, puisqu'il est ensuite
    abandonne, mais il ne peut pas alterer une donnee imposee.

    MESURE DU DEFAUT (labyrinthe out, injection continue a 100)
    -----------------------------------------------------------
    Sans reimposition de la condition sur l'etat intermediaire, la maille
    nourriciere tombe de 100 a 100 - dt*v*100 pendant l'etape 2, et l'etape 3
    lit cette valeur appauvrie. La masse injectee par pas vaut alors

        dm = 100 dt (somme des v_j)  -  50 dt^2 (somme des v_j au carre)

    soit, avec dt = 1.1046, somme v_j = 2.574996 et somme v_j^2 = 0.371299 :

        dm = 261.77   au lieu de   284.43        deficit de 7.97 %

    Avec reimposition : 284.43, soit la totalite. Le resultat est identique
    en reconstruction constante et en reconstruction lineaire, ce qui designe
    le schema temporel, et non la reconstruction, comme origine du defaut.
    `pas_euler` ne comporte qu'une seule etape, donc aucun etat
    intermediaire, et n'est pas concerne.

    Dans un schema a plusieurs etapes, les conditions aux limites doivent
    ainsi etre appliquees a chaque etape, chacune constituant un calcul
    complet qui requiert une frontiere correcte.

    Parametres
    ----------
    conditions : fonction c -> c, ou None.
        Remet la frontiere a sa valeur imposee. Pour une injection continue,
        `dehors` designant (domain == 0) et `alimente` les mailles du contour
        situees juste au-dessus de l'entree :

            def conditions(c):
                c = c.copy()
                c[dehors] = 0.0        # eau propre a l'exterieur
                c[alimente] = 100.0    # sauf aux mailles d'injection
                return c

        La valeur None ne se justifie que si rien n'entre par la frontiere,
        cas d'une tache initiale transportee sans apport : la valeur de
        l'etat intermediaire hors domaine n'est alors jamais lue en amont.
    """
    k1 = bilan(c, domain, u_bord, v_bord, lineaire, dx, dy)
    essai = c + dt * k1
    if conditions is not None:
        essai = conditions(essai)   # la frontiere est une donnee imposee
    k2 = bilan(essai, domain, u_bord, v_bord, lineaire, dx, dy)
    return c + dt * (k1 + k2) / 2.0


# =====================================================================
#  LES CONDITIONS AUX LIMITES FAIBLES
# =====================================================================

def applique_conditions_limites(c, domain, c_exterieur=0.0):
    """
    Impose la concentration dans toutes les mailles HORS domaine
    (domain == 0 : les murs et le contour ajoute par le solveur fourni).

    Le bilan de la derniere maille du domaine fait intervenir sa face de
    sortie, dont le flux est lu en amont, donc dans la maille fluide : la
    matiere quitte le domaine. Dans le tableau, elle est cependant deposee
    dans la maille de contour, qui n'appartient pas au domaine. Sans remise a
    zero, elle s'y accumule et le bilan de masse devient faux.

    A l'entree, symetriquement, la maille de contour constitue l'amont de la
    premiere face : sa valeur est celle du constituant qui penetre dans le
    domaine. La fixer a zero revient a faire entrer de l'eau propre.

    Il s'agit d'une condition limite faible : ce n'est pas la valeur d'une
    maille du domaine qui est imposee, mais le flux qui traverse sa
    frontiere.

    Parametres
    ----------
    c_exterieur : concentration de ce qui entre. 0.0 = eau propre.

    Retourne
    --------
    c : le champ corrige (une copie ; l'original n'est pas modifie)
    masse_retiree : quantite presente hors domaine, c'est-a-dire la masse
                    sortie pendant le pas. Sert au controle de la
                    conservation.
    """
    c = c.copy()
    dehors = (domain == 0)
    masse_retiree = c[dehors].sum() - c_exterieur * dehors.sum()
    c[dehors] = c_exterieur
    return c, masse_retiree
