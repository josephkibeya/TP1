"""
test_advection_canal_rectiligne.py
-----------------------------------
Verification du moteur `advection.py` sur un CANAL RECTILIGNE.

C'est la consigne de la diapo 2 du professeur :

    "Travailler initialement dans un canal rectiligne, champ de vitesse
     uniforme. Solution exacte == translation de la donnee"

POURQUOI UN CAS INVENTE
-----------------------
Les chiffres de ce couloir (7 cellules, u = 1, encre 100 dans les cellules
2 et 3, dt = 0.5) ne viennent PAS du labyrinthe : ils sont choisis pour que
la reponse exacte soit connue d'avance et que tous les nombres tombent sur
des moities, verifiables a la main en deux minutes.

Une fonction fausse dans le labyrinthe complet est introuvable. La meme
fonction, testee seule sur un cas dont on connait la reponse, se repere
tout de suite. On ne touche au vrai labyrinthe qu'une fois ces tests passes.

LES SIX TESTS
-------------
    1. la masse est-elle conservee ?
    2. a Courant = 1, la translation est-elle EXACTE ?
    3. les murs sont-ils impermeables ?
    4. pourquoi le professeur apparie les deux versions
    5. les deux versions cote a cote
    6. le pas de temps sur le vrai labyrinthe (preparation de l'etape suivante)

CE FICHIER N'ECRIT RIEN SUR LE DISQUE. Il lit out_pot.npy et out_domain.txt
(test 6) et affiche du texte. Rien d'autre.

Pour le lancer :   python src/test_advection_canal_rectiligne.py
"""

import numpy as np

from advection import (calcule_dt, pentes, valeurs_aux_faces, flux_upwind,
                       bilan, pas_euler, pas_rk2, applique_conditions_limites)


# =====================================================================
#  LE DOMAINE DE TEST
# ===================================================================== (consigne de la diapo 2 du professeur)
# =====================================================================

def canal_rectiligne(n_cellules=7, u_valeur=1.0):
    """
    Fabrique un petit domaine de test : un couloir droit horizontal, avec des
    murs au-dessus et au-dessous, et un champ de vitesse uniforme.

    "Travailler initialement dans un canal rectiligne, champ de vitesse
    uniforme. Solution exacte == translation de la donnee" (diapo 2).

    On connait donc la reponse exacte, ce qui permet de detecter un bug tout
    de suite, au lieu de le chercher dans le labyrinthe complet.

        colonne :  0    1   ...   n    n+1
        ligne 0 :  MUR  MUR  ...  MUR  MUR
        ligne 1 :  MUR  c1   ...  cn   MUR     <- le couloir
        ligne 2 :  MUR  MUR  ...  MUR  MUR

    Les faces d'entree (entre la colonne 0 et la colonne 1) et de sortie
    portent la vitesse u_valeur : l'eau traverse. La maille 0 reste a c = 0,
    donc c'est de l'eau PROPRE qui entre.

    Retourne
    --------
    domain, u_bord, v_bord
    """
    nI, nJ = 3, n_cellules + 2
    domain = np.zeros((nI, nJ), dtype=int)
    domain[1, 1:-1] = 1                      # le couloir

    u_bord = np.zeros((nI, nJ - 1))
    u_bord[1, :] = u_valeur                  # toutes les faces de la ligne 1,
                                             # entree et sortie comprises
    v_bord = np.zeros((nI - 1, nJ))          # aucun ecoulement vertical
    return domain, u_bord, v_bord


def _bloc_encre(domain, colonnes, valeur=100.0):
    """Pose un bloc d'encre dans le couloir, aux colonnes demandees."""
    c = np.zeros(domain.shape)
    c[1, colonnes] = valeur
    return c


if __name__ == "__main__":
    import os

    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    ligne = "=" * 78

    def boucle(c, domain, u_bord, v_bord, dt, n, lineaire, schema):
        """Avance de n pas et renvoie (champ final, masse sortie)."""
        avance = pas_euler if schema == "euler" else pas_rk2
        sorti = 0.0
        for _ in range(n):
            c = avance(c, domain, u_bord, v_bord, dt, lineaire)
            c, retire = applique_conditions_limites(c, domain)
            sorti += retire
        return c, sorti

    # -----------------------------------------------------------------
    print(ligne)
    print("TEST 1 - CANAL RECTILIGNE : la masse est-elle conservee ?")
    print(ligne)
    domain, u_bord, v_bord = canal_rectiligne(7, u_valeur=1.0)
    fluide = domain > 0
    c = _bloc_encre(domain, [2, 3])
    dt = 0.5                                   # nu = 0.5
    masse_depart = c[fluide].sum()
    sorti = 0.0
    print(f"  depart : {c[1, 1:-1]}   masse = {masse_depart:.2f}")
    for k in range(6):
        c = pas_euler(c, domain, u_bord, v_bord, dt, lineaire=False)
        c, retire = applique_conditions_limites(c, domain)
        sorti += retire
        print(f"  pas {k+1} : {c[1, 1:-1]}   dans={c[fluide].sum():7.2f}"
              f"  sorti={sorti:6.2f}  TOTAL={c[fluide].sum()+sorti:7.2f}")
    ecart = abs(c[fluide].sum() + sorti - masse_depart)
    print(f"  -> ecart sur la masse totale : {ecart:.2e}   "
          f"{'OK' if ecart < 1e-10 else 'ECHEC'}")

    # -----------------------------------------------------------------
    print()
    print(ligne)
    print("TEST 2 - CANAL RECTILIGNE : translation EXACTE a Courant = 1")
    print(ligne)
    print("  A nu = 1, la dissipation numerique eps = u dx (1 - nu) / 2 vaut")
    print("  exactement ZERO : le bloc doit se deplacer sans se deformer.")
    domain, u_bord, v_bord = canal_rectiligne(9, u_valeur=1.0)
    c = _bloc_encre(domain, [1, 2])
    print(f"  depart : {c[1, 1:-1]}")
    c, _ = boucle(c, domain, u_bord, v_bord, 1.0, 3, False, "euler")
    attendu = np.zeros(9)
    attendu[3:5] = 100.0
    print(f"  apres 3 pas : {c[1, 1:-1]}")
    print(f"  attendu     : {attendu}")
    ecart = np.abs(c[1, 1:-1] - attendu).max()
    print(f"  -> ecart maximal : {ecart:.2e}   {'OK' if ecart < 1e-10 else 'ECHEC'}")

    # -----------------------------------------------------------------
    print()
    print(ligne)
    print("TEST 3 - CANAL RECTILIGNE : les murs sont-ils impermeables ?")
    print(ligne)
    domain, u_bord, v_bord = canal_rectiligne(7, u_valeur=1.0)
    c = _bloc_encre(domain, [2, 3])
    c, _ = boucle(c, domain, u_bord, v_bord, 0.5, 10, False, "euler")
    fuite = max(np.abs(c[0, :]).max(), np.abs(c[2, :]).max())
    print(f"  encre dans les lignes de MUR apres 10 pas : {fuite:.2e}   "
          f"{'OK' if fuite < 1e-12 else 'ECHEC'}")

    # -----------------------------------------------------------------
    print()
    print(ligne)
    print("TEST 4 - POURQUOI LE PROFESSEUR APPARIE LES DEUX VERSIONS")
    print(ligne)
    print("  On lance les QUATRE combinaisons sur 40 pas, a nu = 0.5,")
    print("  et on regarde si les valeurs restent raisonnables.")
    print()
    domain, u_bord, v_bord = canal_rectiligne(20, u_valeur=1.0)
    print(f"  {'reconstruction':>15} {'temps':>7} | {'max |c| apres 40 pas':>22} | verdict")
    print("  " + "-" * 70)
    for nom_r, lin in (("constante", False), ("lineaire", True)):
        for nom_t, sch in (("Euler", "euler"), ("RK2", "rk2")):
            c = _bloc_encre(domain, [3, 4, 5])
            c, _ = boucle(c, domain, u_bord, v_bord, 0.5, 40, lin, sch)
            m = np.abs(c).max()
            verdict = "STABLE" if m < 200 else "EXPLOSE"
            marque = "   <- demande" if (nom_r, nom_t) in (("constante", "Euler"),
                                                           ("lineaire", "RK2")) else ""
            print(f"  {nom_r:>15} {nom_t:>7} | {m:>22.2f} | {verdict}{marque}")
    print()
    print("  Conclusion : LINEAIRE + EULER est instable pour tout nu > 0.")
    print("  Ce n'est donc pas seulement que l'ordre 2 en espace serait")
    print("  gaspille : cette combinaison ne fonctionne tout simplement pas.")

    # -----------------------------------------------------------------
    print()
    print(ligne)
    print("TEST 5 - LES DEUX VERSIONS DU PROFESSEUR, COTE A COTE")
    print(ligne)
    domain, u_bord, v_bord = canal_rectiligne(7, u_valeur=1.0)
    for nom, lin, sch in (("A : constante + Euler", False, "euler"),
                          ("B : lineaire  + RK2  ", True, "rk2")):
        c = _bloc_encre(domain, [2, 3])
        c, _ = boucle(c, domain, u_bord, v_bord, 0.5, 6, lin, sch)
        print(f"  {nom} : {c[1, 1:-1]}")
        print(f"  {'':23}  maximum = {c[1, 1:-1].max():6.2f}"
              f"   minimum = {c[1, 1:-1].min():6.2f}")
    print()
    print("  Le maximum exact vaut 100 : plus il est haut, moins le schema a")
    print("  dissipe. Un minimum negatif signale un depassement, que les")
    print("  limiteurs de la Partie 8 corrigent.")

    # -----------------------------------------------------------------
    print()
    print(ligne)
    print("TEST 6 - LE PAS DE TEMPS SUR LE VRAI LABYRINTHE")
    print(ligne)
    ici = os.path.dirname(os.path.abspath(__file__))
    chemin_pot = os.path.join(ici, "..", "data", "out_pot.npy")
    chemin_dom = os.path.join(ici, "..", "data", "out_domain.txt")

    if os.path.exists(chemin_pot):
        from vitesses import vitesses_aux_bords, annule_vitesses_murs

        pot = np.load(chemin_pot)
        dom = np.loadtxt(chemin_dom)
        uu, vv = vitesses_aux_bords(pot)
        uu, vv = annule_vitesses_murs(uu, vv, dom)

        dt_laby, infos = calcule_dt(uu, vv, dom, marge=0.8)
        print(f"  faces valides            : {infos['faces_u_valides']} verticales,"
              f" {infos['faces_v_valides']} horizontales")
        print(f"  |u| max sur ces faces    : {infos['u_max']:.4f}")
        print(f"  |v| max sur ces faces    : {infos['v_max']:.4f}")
        print(f"  dt_max (Courant = 1)     : {infos['dt_max']:.4f}")
        print(f"  marge retenue            : {infos['marge']}")
        print(f"  dt utilise               : {infos['dt']:.4f}")
        print(f"  Courant obtenu           : {infos['courant']:.4f}"
              f"   (doit valoir la marge)")
        print()
        print("  Pour comparaison, SI on oubliait d'exclure les faces")
        print("  d'entree/sortie fictives :")
        um, vm = np.abs(uu).max(), np.abs(vv).max()
        print(f"    |u|max = {um:.4f}, |v|max = {vm:.1f}"
              f"  ->  dt_max = {1/(um+vm):.5f}")
        print(f"    soit {infos['dt_max']/(1/(um+vm)):.0f} fois plus petit,"
              f" donc {infos['dt_max']/(1/(um+vm)):.0f} fois plus de pas pour rien.")
    else:
        print("  (donnees du labyrinthe introuvables, test saute)")
