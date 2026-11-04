"""
test_advection_canal_rectiligne.py
-----------------------------------
Verification du moteur `advection.py` sur un canal rectiligne.

Le cas de reference est celui propose par l'enonce : un canal rectiligne a
champ de vitesse uniforme, dont la solution exacte est la translation de la
donnee initiale.

CHOIX DES VALEURS NUMERIQUES
----------------------------
Les valeurs retenues pour ce couloir, sept cellules, u = 1, concentration 100
dans les cellules 2 et 3 et dt = 0.5, ne proviennent pas du labyrinthe. Elles
sont choisies de maniere que la solution exacte soit connue a l'avance et que
les resultats intermediaires tombent sur des demi-entiers, donc verifiables
a la main.

Une fonction erronee est difficilement decelable au sein du labyrinthe
complet, alors qu'elle se repere immediatement sur un cas de solution connue.
Le labyrinthe n'est donc aborde qu'une fois ces controles satisfaits.

LES SIX CONTROLES
-----------------
    1. conservation de la masse
    2. exactitude de la translation a Courant = 1
    3. impermeabilite des murs
    4. appariement des schemas en temps et en espace
    5. comparaison des deux versions
    6. pas de temps admissible sur le labyrinthe

Ce module n'ecrit aucun fichier. Il lit out_pot.npy et out_domain.txt pour le
sixieme controle et produit un compte rendu a l'ecran.

Execution :   python src/test_advection_canal_rectiligne.py
"""

import numpy as np

from advection import (calcule_dt, pentes, valeurs_aux_faces, flux_upwind,
                       bilan, pas_euler, pas_rk2, applique_conditions_limites)


# =====================================================================
#  LE DOMAINE DE TEST
# =====================================================================
# =====================================================================

def canal_rectiligne(n_cellules=7, u_valeur=1.0):
    """
    Construit un domaine de controle : un couloir rectiligne horizontal borde
    de murs, parcouru par un champ de vitesse uniforme.

    Conformement au cas de reference de l'enonce, la solution exacte y est la
    translation de la donnee initiale. Elle est donc connue, ce qui permet de
    deceler immediatement une erreur, au lieu de la rechercher dans le
    labyrinthe complet.

        colonne :  0    1   ...   n    n+1
        ligne 0 :  MUR  MUR  ...  MUR  MUR
        ligne 1 :  MUR  c1   ...  cn   MUR     <- le couloir
        ligne 2 :  MUR  MUR  ...  MUR  MUR

    Les faces d'entree, entre les colonnes 0 et 1, et de sortie portent la
    vitesse u_valeur : le fluide les traverse. La maille 0 restant a c = 0,
    c'est de l'eau propre qui penetre dans le couloir.

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
    v_bord = np.zeros((nI - 1, nJ))          # aucune composante verticale
    return domain, u_bord, v_bord


def _bloc_encre(domain, colonnes, valeur=100.0):
    """Place un bloc de constituant dans le couloir, aux colonnes indiquees."""
    c = np.zeros(domain.shape)
    c[1, colonnes] = valeur
    return c


if __name__ == "__main__":
    import os

    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    ligne = "=" * 78

    def boucle(c, domain, u_bord, v_bord, dt, n, lineaire, schema):
        """Avance de n pas et renvoie le champ final et la masse sortie."""
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
    dt = 0.5                                   # soit nu = 0.5
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
    print("TEST 4 - APPARIEMENT DES SCHEMAS EN TEMPS ET EN ESPACE")
    print(ligne)
    print("  Les quatre combinaisons sont comparees sur 40 pas, a nu = 0.5,")
    print("  en controlant que les valeurs obtenues restent bornees.")
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
            marque = "   <- retenue" if (nom_r, nom_t) in (("constante", "Euler"),
                                                           ("lineaire", "RK2")) else ""
            print(f"  {nom_r:>15} {nom_t:>7} | {m:>22.2f} | {verdict}{marque}")
    print()
    print("  La combinaison lineaire + Euler est instable pour tout nu > 0.")
    print("  L'appariement ne vise donc pas seulement a eviter de gaspiller")
    print("  l'ordre 2 en espace : cette combinaison est inutilisable.")

    # -----------------------------------------------------------------
    print()
    print(ligne)
    print("TEST 5 - COMPARAISON DES DEUX VERSIONS ETUDIEES")
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
    print("  Le maximum exact vaut 100 : plus la valeur obtenue en est proche,")
    print("  moins le schema a dissipe. Un minimum negatif signale un")
    print("  depassement, qu'un limiteur de pente permettrait de corriger.")

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
        print("  A titre de comparaison, sans exclusion des faces d'entree")
        print("  et de sortie portant encore une vitesse fictive :")
        um, vm = np.abs(uu).max(), np.abs(vv).max()
        print(f"    |u|max = {um:.4f}, |v|max = {vm:.1f}"
              f"  ->  dt_max = {1/(um+vm):.5f}")
        print(f"    soit {infos['dt_max']/(1/(um+vm)):.0f} fois plus petit,"
              f" donc {infos['dt_max']/(1/(um+vm)):.0f} fois plus de pas de calcul.")
    else:
        print("  (donnees du labyrinthe introuvables, test saute)")
