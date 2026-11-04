"""
vitesses.py
-----------
Premier module du projet "Volumes Finis - Labyrinthe".

A partir du champ de potentiel de vitesse calcule par le solveur fourni
`laplace_only.py`, ce module determine les vitesses normales a chaque bord
des mailles de la grille, puis verifie que le champ obtenu est a divergence
nulle.

L'ecoulement etant irrotationnel, la vitesse derive d'un potentiel phi :

        u_i = d(phi)/d(x_i)

u designant la composante selon x et v la composante selon y. Le potentiel
n'est connu qu'aux centres des mailles, a raison d'une valeur par case de la
grille, alors que la formulation en volumes finis requiert les vitesses
normales aux bords. Entre deux mailles voisines separees de dx, la derivee
au bord qui les separe est approchee par

        d(phi)/dx  =  (phi[voisin] - phi[maille]) / dx

Cette expression a la forme d'une difference avant, mais le bord se situe
exactement a mi-distance des deux centres : rapportee a ce bord, elle
constitue en realite un schema centre, donc precis au second ordre. C'est le
cas i+1/2 de l'analyse de l'erreur de troncature.

SENS DE LA VITESSE
----------------------------------------------------------------------
Avec V = +grad(phi), le fluide se dirige vers les potentiels croissants. Or
le solveur impose phi = +100 a l'entree, sur la ligne du haut, et
phi = -100 a la sortie, sur la ligne du bas : l'ecoulement irait alors de la
sortie vers l'entree. Le constat se verifie sur laby.png, ou v_bord est
negatif, donc oriente vers le haut, sous l'entree.

La convention V = -grad(phi) est donc retenue : toutes les vitesses sont
multipliees par -1, ce qui oriente l'ecoulement de l'entree vers la sortie.
Le champ reste a divergence nulle, un changement de signe laissant un zero
inchange, le module de la vitesse est conserve, et le solveur fourni n'est
pas modifie.

Convention matricielle du projet :
    - [0,0] est en haut a gauche
    - voisin de gauche de [i,j] : [i, j-1]      -> direction x (colonnes)
    - voisin de droite  de [i,j] : [i, j+1]      -> direction x (colonnes)
    - voisin du haut    de [i,j] : [i-1, j]      -> direction y (lignes)
    - voisin du bas     de [i,j] : [i+1, j]      -> direction y (lignes)

DOMAINE DE VALIDITE DE LA FORMULE
----------------------------------------------------------------------
La formule ci-dessus n'a de sens que si les deux mailles situees de part et
d'autre du bord appartiennent au domaine d'ecoulement (domain == 1). Si l'une
des deux est un mur (domain == 0), le potentiel y vaut zero par defaut, par
simple remplissage et non par resolution, et la formule produit alors une
vitesse fictive, comme si le fluide traversait le mur. Le controle en est
fait par `verifie_divergence_nulle`.
"""

import numpy as np


def vitesses_aux_bords(pot, dx=1.0, dy=1.0):
    """
    Calcule la vitesse normale a chaque bord de la grille, y compris les
    bords qui touchent un mur ; ces derniers sont annules par la suite.

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
        Vitesse normale a chaque bord vertical, c'est-a-dire a chaque bord
        entre une maille et sa voisine de droite.
        u_bord[i, j] = bord entre [i, j] et [i, j+1].
    v_bord : numpy.ndarray, shape (nI-1, nJ)
        Vitesse normale a chaque bord horizontal, c'est-a-dire a chaque bord
        entre une maille et sa voisine du bas.
        v_bord[i, j] = bord entre [i, j] et [i+1, j].

    Mise en oeuvre
    --------------
    Les deux differences sont formees par decalage de tableaux :
    `pot[:, 1:] - pot[:, :-1]` soustrait, colonne par colonne, chaque maille
    de sa voisine de droite, et fournit donc l'ensemble des bords verticaux
    en une seule operation, sans boucle. Le meme procede s'applique aux
    lignes avec `pot[1:, :]` et `pot[:-1, :]`.
    """
    # Signe moins : V = -grad(phi), de maniere que l'ecoulement aille de
    # l'entree (phi = +100, en haut) vers la sortie (phi = -100, en bas).
    # Convention : u_bord > 0 vers la droite ; v_bord > 0 vers le bas, les
    # lignes de la matrice croissant vers le bas.
    u_bord = -(pot[:, 1:] - pot[:, :-1]) / dx
    v_bord = -(pot[1:, :] - pot[:-1, :]) / dy
    return u_bord, v_bord


def annule_vitesses_murs(u_bord, v_bord, domain):
    """
    Annule la vitesse normale sur toute face dont l'un des deux cotes au
    moins est un mur (domain == 0).

    Dans `pot`, un mur porte la valeur zero, par simple remplissage et non
    comme potentiel resolu. Entre une maille fluide et un mur, la formule de
    `vitesses_aux_bords` produit donc une vitesse fictive, de l'ordre de 68 en
    moyenne contre 0.014 pour une veritable face de fluide. Or rien ne
    traverse un mur. Le solveur fourni retient deja cette hypothese, puisqu'il
    exclut les voisins murs de son calcul, ce qui correspond a la condition
    d(phi)/dn = 0 : annuler ces faces est donc coherent avec lui.

    CAS DE L'ENTREE ET DE LA SORTIE (domain == 2)
    ---------------------------------------------
    Le solveur entoure le domaine d'un contour de valeurs nulles. La face
    entre ce contour et une maille d'entree ou de sortie n'est pas un mur :
    le fluide y passe. Elle est donc laissee inchangee a ce stade. Sa valeur
    courante, -100 a l'entree, est elle aussi fictive et sera traitee par
    `vitesses_entree_sortie`.

    Parametres
    ----------
    u_bord, v_bord : sortie de `vitesses_aux_bords` (modifiees sur place).
    domain : numpy.ndarray, 0 = mur, 1 = ecoulement, 2 = entree/sortie.

    Retourne
    --------
    u_bord, v_bord : memes tableaux, avec 0 sur les faces de mur.
    """
    fluide = domain > 0    # vrai pour 1 (ecoulement) et 2 (entree/sortie)

    # Une face est une face de mur des que l'une de ses deux mailles est un mur.
    # u_bord[i, j] : entre [i, j] et [i, j+1] ; v_bord[i, j] : entre [i, j] et [i+1, j]
    face_mur_u = ~(fluide[:, 1:] & fluide[:, :-1])
    face_mur_v = ~(fluide[1:, :] & fluide[:-1, :])

    # Faces entre une maille d'entree/sortie et le contour : non annulees.
    face_es_v = ((domain[1:, :] == 2) & (domain[:-1, :] == 0)) | \
                ((domain[:-1, :] == 2) & (domain[1:, :] == 0))

    u_bord[face_mur_u] = 0.0
    v_bord[face_mur_v & ~face_es_v] = 0.0
    return u_bord, v_bord


def vitesses_entree_sortie(u_bord, v_bord, domain, dx=1.0, dy=1.0):
    """
    Determine la vitesse normale effective sur les faces d'entree et de
    sortie, c'est-a-dire les faces separant une maille `domain == 2` du
    contour exterieur. Leur nombre vaut deux fois la largeur de l'ouverture :
    24 faces sur le labyrinthe laby14, 42 sur le grand labyrinthe out fourni
    en exemple.

    ORIGINE DU PROBLEME
    -------------------
    La formule v = -(phi[bas] - phi[haut]) / dy n'a de sens que si les deux
    mailles portent un potentiel resolu. Or le contour ajoute par le solveur
    est un mur, sur lequel l'equation de Laplace n'a jamais ete resolue : il y
    subsiste un zero de remplissage. A l'entree, le calcul donne donc

        v = -(100 - 0) / 1 = -100

    alors que les faces voisines valent environ 0.12, soit un rapport de
    l'ordre de 800. Cette valeur n'est pas une vitesse : elle resulte d'une
    soustraction entre une donnee reelle, 100, et un remplissage, 0.

    A la difference d'une face de mur, elle ne peut pas etre annulee :
    l'entree et la sortie sont precisement les deux seuls endroits ou le
    fluide traverse la frontiere, et les mettre a zero reviendrait a obturer
    le labyrinthe.

    SOLUTION RETENUE : IMPOSER LA CONSERVATION DE LA MASSE
    ------------------------------------------------------
    L'ecoulement est incompressible : aucune maille ne peut ni creer ni
    detruire du fluide. Pour une maille [i, j], ce qui entre par ses 4 faces
    doit donc exactement ressortir par les autres. En Volumes Finis, cela
    s'ecrit (divergence discrete nulle) :

        (u_droite - u_gauche) / dx  +  (v_bas - v_haut) / dy  =  0

    avec, pour la maille [i, j] :
        u_gauche = u_bord[i, j-1]      u_droite = u_bord[i, j]
        v_haut   = v_bord[i-1, j]      v_bas    = v_bord[i, j]

    Pour une maille d'entree, les trois autres faces sont connues et
    correctes : elles separent deux mailles resolues, ou bien elles touchent
    un mur et sont nulles. Il ne subsiste donc qu'une seule inconnue, la face
    du haut, et l'equation precedente se resout directement :

        v_haut = v_bas + (dy / dx) * (u_droite - u_gauche)

    Pour une maille de sortie, l'inconnue est la face du bas :

        v_bas = v_haut - (dy / dx) * (u_droite - u_gauche)

    La vitesse d'entree n'est donc pas estimee, mais deduite de
    l'incompressibilite : il entre par le haut exactement ce qui ressort par
    les trois autres cotes.

    Apres cette correction, la divergence est nulle sur l'ensemble des mailles
    fluides, mailles d'entree et de sortie comprises, et le debit total
    entrant egale le debit total sortant. Ce dernier resultat decoule de la
    somme des bilans de toutes les mailles, ou les flux interieurs
    s'annulent deux a deux. C'est ce qui garantit la conservation de la masse
    de constituant par le schema d'advection.

    Parametres
    ----------
    u_bord, v_bord : sortie de `annule_vitesses_murs` (modifiees sur place).
        Cette fonction doit etre appelee apres `annule_vitesses_murs` : dans
        l'ordre inverse, les faces laterales des mailles d'entree et de sortie
        porteraient encore des vitesses fictives et le bilan serait faux.
    domain : numpy.ndarray, 0 = mur, 1 = ecoulement, 2 = entree/sortie.
    dx, dy : dimensions d'une maille.

    Retourne
    --------
    u_bord, v_bord : memes tableaux, faces d'entree/sortie corrigees.
    infos : dict
        Nombre de faces corrigees, debit entrant, debit sortant et ecart
        entre les deux.
    """
    nI, nJ = domain.shape

    # Les mailles d'entree et de sortie se comptent par dizaines, contre
    # plusieurs dizaines de milliers pour le domaine. Une boucle Python y est
    # donc d'un cout negligeable, et plus lisible qu'une forme vectorisee.
    mailles_es = np.argwhere(domain == 2)

    debit_entrant = 0.0
    debit_sortant = 0.0
    nb_faces = 0

    for i, j in mailles_es:
        # Les faces laterales et la face opposee a l'exterieur sont connues.
        # Les tests sur j evitent un debordement de tableau dans le cas ou une
        # maille d'entree toucherait le bord gauche ou droit de l'image.
        u_gauche = u_bord[i, j - 1] if j > 0      else 0.0
        u_droite = u_bord[i, j]     if j < nJ - 1 else 0.0

        # Cote ou se trouve l'exterieur, c'est-a-dire le contour (domain == 0).
        exterieur_en_haut = (i > 0)      and (domain[i - 1, j] == 0)
        exterieur_en_bas  = (i < nI - 1) and (domain[i + 1, j] == 0)

        if exterieur_en_haut:
            # L'inconnue est la face du haut, v_bord[i-1, j].
            v_bas = v_bord[i, j] if i < nI - 1 else 0.0
            v_haut = v_bas + (dy / dx) * (u_droite - u_gauche)
            v_bord[i - 1, j] = v_haut
            # v > 0 etant oriente vers le bas, une entree admet du fluide
            # lorsque v_haut > 0. Le debit a travers une face vaut v * dx.
            debit_entrant += v_haut * dx
            nb_faces += 1

        elif exterieur_en_bas:
            # L'inconnue est la face du bas, v_bord[i, j].
            v_haut = v_bord[i - 1, j] if i > 0 else 0.0
            v_bas = v_haut - (dy / dx) * (u_droite - u_gauche)
            v_bord[i, j] = v_bas
            debit_sortant += v_bas * dx
            nb_faces += 1

    infos = {
        "nb_mailles_entree_sortie": len(mailles_es),
        "nb_faces_corrigees": nb_faces,
        "debit_entrant": debit_entrant,
        "debit_sortant": debit_sortant,
        "ecart_entrant_sortant": abs(debit_entrant - debit_sortant),
    }
    return u_bord, v_bord, infos


def verifie_divergence_nulle(u_bord, v_bord, domain, dx=1.0, dy=1.0, tol=1e-9):
    """
    Verifie que du/dx + dv/dy = 0, traduisant l'incompressibilite, sur les
    seules mailles pleinement interieures : celles qui appartiennent au
    domaine (domain == 1) et dont les quatre voisines y appartiennent aussi.

    RESTRICTION DU CONTROLE
    -----------------------
    Comme indique en tete de module, la formule de vitesse aux bords n'a de
    sens que si les deux mailles encadrant le bord appartiennent au domaine.
    Une maille qui touche un mur possede donc au moins un bord dont la vitesse
    est fictive, et la divergence qui en resulte n'a aucune raison d'etre
    nulle : il ne s'agit pas d'un defaut du calcul, mais d'une maille hors du
    champ de validite du controle. Seules sont donc retenues les mailles
    entourees de quatre voisines d'ecoulement, ou la formule est valide des
    deux cotes.

    Parametres
    ----------
    u_bord, v_bord : sortie de `vitesses_aux_bords`
    domain : numpy.ndarray, meme forme que pot (0 = mur, 1 = ecoulement,
        2 = condition limite entree/sortie)
    tol : float
        Seuil en dessous duquel la divergence est consideree comme nulle.

    Retourne
    --------
    divergence : numpy.ndarray
        Divergence calculee sur chaque maille interieure, y compris hors du
        masque, ce qui permet de localiser les mailles ou elle ne s'annule
        pas.
    masque_interieur : numpy.ndarray (bool)
        Vrai la ou la maille est pleinement interieure, donc la ou la
        divergence doit etre nulle.
    ok : bool
        Vrai si la divergence maximale en valeur absolue, restreinte au
        masque, est inferieure a `tol`.
    """
    # Divergence brute, calculee partout, y compris pres des murs ou elle n'a
    # pas de sens physique.
    du_dx = (u_bord[1:-1, 1:] - u_bord[1:-1, :-1]) / dx
    dv_dy = (v_bord[1:, 1:-1] - v_bord[:-1, 1:-1]) / dy
    divergence = du_dx + dv_dy

    # La maille elle-meme et ses quatre voisines doivent valoir 1.
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
    # Ce bloc ne s'execute qu'a l'appel direct du fichier, et non lors d'un
    # import du module depuis un autre script.
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

    # --- Avant correction : les faces de mur portent des vitesses fictives ---
    divergence, masque, ok = verifie_divergence_nulle(u_bord, v_bord, domain)
    print("Nombre de mailles verifiees (pleinement interieures) :", masque.sum())
    print("Divergence max (valeur absolue) sur ces mailles       :", np.max(np.abs(divergence[masque])))
    print("Divergence nulle partout ou ca compte ? ->", ok)

    touche_mur = (domain[1:-1, 1:-1] == 1) & ~masque
    print()
    print("AVANT correction, sur les", touche_mur.sum(), "mailles qui touchent un mur :")
    print("Divergence max (valeur absolue) :", np.max(np.abs(divergence[touche_mur])))
    print("-> grand et normal : la formule brute invente une vitesse a travers le mur.")

    # --- Correction 1 : annulation des faces de mur ---
    u_bord, v_bord = annule_vitesses_murs(u_bord, v_bord, domain)
    divergence, _, _ = verifie_divergence_nulle(u_bord, v_bord, domain)
    tout_fluide = (domain[1:-1, 1:-1] == 1)   # mailles d'ecoulement, murs voisins compris
    div_max = np.max(np.abs(divergence[tout_fluide]))
    print()
    print("APRES correction des murs, sur les", tout_fluide.sum(), "mailles d'ecoulement :")
    print("Divergence max (valeur absolue) :", div_max)
    print("Divergence nulle partout ? ->", bool(div_max < 1e-9))

    # --- Correction 2 : faces d'entree et de sortie ---
    print()
    # Masque des faces d'entree et de sortie, de meme forme que v_bord :
    faces_es = ((domain[1:, :] == 2) & (domain[:-1, :] == 0)) | \
               ((domain[:-1, :] == 2) & (domain[1:, :] == 0))
    print("Les", faces_es.sum(), "faces entree/sortie AVANT correction : v =",
          np.unique(v_bord[faces_es]))

    u_bord, v_bord, infos = vitesses_entree_sortie(u_bord, v_bord, domain)
    print("Mailles entree+sortie :", infos["nb_mailles_entree_sortie"],
          "-> faces corrigees :", infos["nb_faces_corrigees"])
    print("Debit entrant :", infos["debit_entrant"])
    print("Debit sortant :", infos["debit_sortant"])
    print("Ecart entrant/sortant :", infos["ecart_entrant_sortant"],
          "-> la masse est conservee a l'echelle de tout le labyrinthe.")

    divergence, _, _ = verifie_divergence_nulle(u_bord, v_bord, domain)
    # Le controle porte cette fois sur toutes les mailles fluides, mailles
    # d'entree et de sortie comprises.
    tout_fluide = (domain[1:-1, 1:-1] >= 1)
    div_max = np.max(np.abs(divergence[tout_fluide]))
    print()
    print("APRES les deux corrections, sur les", tout_fluide.sum(),
          "mailles fluides (entree/sortie comprises) :")
    print("Divergence max (valeur absolue) :", div_max)
    print("Divergence nulle partout ? ->", bool(div_max < 1e-9))

    # Le pas de temps admissible depend de la vitesse maximale : on verifie
    # que les faces d'entree et de sortie ne la majorent plus artificiellement.
    print()
    print("|u| max :", np.max(np.abs(u_bord)))
    print("|v| max :", np.max(np.abs(v_bord)))
