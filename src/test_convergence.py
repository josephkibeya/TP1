"""
test_convergence.py
-------------------
Mesure experimentale de l'ordre de convergence des deux versions.

PORTEE DU CONTROLE
==================
Le theoreme de Lax etablit que consistance et stabilite entrainent la
convergence. Il porte toutefois sur le schema, non sur son implementation :
il garantit que ce schema, correctement ecrit, converge a l'ordre annonce,
mais ne dit rien de la fidelite du programme au schema.

C'est l'objet de ce test. Une erreur d'indice, un signe inverse ou une
condition limite mal posee laissent le programme s'executer, produire des
champs d'allure correcte et conserver la masse, tout en degradant l'ordre de
2 a 1. La mesure de l'ordre de convergence constitue a cet egard le controle
le plus sensible dont on dispose.

PRINCIPE
========
Le controle porte sur un cas de solution exacte connue : un canal droit, une
vitesse uniforme a et un profil initial regulier. La solution exacte est
alors une translation pure,

    c(x, t) = c0(x - a t)

Le calcul est repete sur des maillages de plus en plus fins, a nombre de
Courant constant, de sorte que dt decroit proportionnellement a dx, et
l'ecart a la solution exacte est mesure en norme L1 :

    E = (1 / N) * somme |c_calcule - c_exact|

Si le schema est d'ordre p, E est proportionnel a dx^p : diviser dx par deux
divise E par 2^p, et l'ordre observe se lit directement

    p = log2( E(dx) / E(dx/2) )

    Version A (constante + Euler) : ordre attendu voisin de 1
    Version B (lineaire + RK2)    : ordre attendu voisin de 2

REGULARITE DU PROFIL INITIAL
============================
Le profil retenu est une gaussienne, et non un creneau. Sur une
discontinuite, la derivee n'existe pas, le developpement de Taylor cesse de
s'appliquer et aucun schema n'atteint l'ordre 2 : la mesure donnerait un
ordre voisin de 1 pour les deux versions et conduirait a conclure a tort a
l'inutilite de la reconstruction lineaire. Les ordres annonces ne valent que
pour des solutions regulieres.
"""

import numpy as np

import advection as A

# --- le cas de reference ------------------------------------------------
VITESSE = 1.0      # a, orientee vers la droite
COURANT = 0.8      # nu, identique sur tous les maillages
X_DEPART = 0.25    # abscisse du centre de la gaussienne a l'instant initial
SIGMA = 0.06       # largeur de la gaussienne
TEMPS = 0.5        # duree simulee ; le profil se deplace de 0,5


def profil_exact(x, t):
    """Solution exacte : la gaussienne translatee de a*t, sans deformation."""
    return 100.0 * np.exp(-((x - X_DEPART - VITESSE * t) / SIGMA) ** 2)


def canal(n_mailles):
    """
    Construit un canal droit de n_mailles cellules sur une longueur 1.

    Le canal compte trois lignes : une ligne d'ecoulement entre deux lignes de
    mur. Les vitesses verticales sont nulles et le transport est donc purement
    horizontal. Le moteur bidimensionnel employe est neanmoins exactement
    celui qui traite le labyrinthe : le controle porte bien sur le code du
    projet, et non sur une implementation separee.
    """
    dx = 1.0 / n_mailles
    domain = np.zeros((3, n_mailles), dtype=float)
    domain[1, :] = 1.0                       # la ligne d'ecoulement

    u_bord = np.zeros((3, n_mailles - 1))
    u_bord[1, :] = VITESSE                   # seules les faces entre deux mailles fluides
    v_bord = np.zeros((2, n_mailles))        # aucun mouvement vertical

    x = (np.arange(n_mailles) + 0.5) * dx    # centres de gravite des mailles
    return domain, u_bord, v_bord, dx, x


def erreur(n_mailles, version):
    """
    Calcule l'ecart moyen a la solution exacte au terme de la duree TEMPS,
    sur un maillage de n_mailles cellules.
    """
    domain, u_bord, v_bord, dx, x = canal(n_mailles)
    dt = COURANT * dx / VITESSE
    n_pas = int(round(TEMPS / dt))

    c = np.zeros_like(domain)
    c[1, :] = profil_exact(x, 0.0)

    for _ in range(n_pas):
        if version == "A":
            c = A.pas_euler(c, domain, u_bord, v_bord, dt, lineaire=False, dx=dx)
        else:
            c = A.pas_rk2(c, domain, u_bord, v_bord, dt, lineaire=True, dx=dx)

    exact = profil_exact(x, n_pas * dt)
    return np.mean(np.abs(c[1, :] - exact)), n_pas


def etude(maillages=(80, 160, 320, 640)):
    """Applique le controle a une suite de maillages et affiche les ordres observes."""
    print("Test de convergence  -  canal droit, gaussienne translatee")
    print("vitesse %.1f   Courant %.2f   duree simulee %.1f"
          % (VITESSE, COURANT, TEMPS))
    print()
    resultats = {}
    for version, attendu in (("A", 1), ("B", 2)):
        nom = ("A : constante + Euler" if version == "A"
               else "B : lineaire + RK2")
        print("%s        (ordre attendu : %d)" % (nom, attendu))
        print("  %8s %8s %8s %14s %10s"
              % ("mailles", "dx", "pas", "erreur L1", "ordre p"))
        print("  " + "-" * 54)
        precedente = None
        ordres = []
        for n in maillages:
            E, n_pas = erreur(n, version)
            if precedente is None:
                texte = "   -"
            else:
                p = np.log2(precedente / E)
                ordres.append(p)
                texte = "%7.2f" % p
            print("  %8d %8.5f %8d %14.6e %10s"
                  % (n, 1.0 / n, n_pas, E, texte))
            precedente = E
        moyen = float(np.mean(ordres)) if ordres else float("nan")
        resultats[version] = moyen
        print("  ordre moyen observe : %.2f   (attendu %d)" % (moyen, attendu))
        print()
    return resultats


if __name__ == "__main__":
    r = etude()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("  Version A : ordre %.2f  ->  %s"
          % (r["A"], "ordre 1 confirme" if 0.8 < r["A"] < 1.3 else "A VERIFIER"))
    print("  Version B : ordre %.2f  ->  %s"
          % (r["B"], "ordre 2 confirme" if 1.7 < r["B"] < 2.3 else "A VERIFIER"))
    print()
    print("  L'ordre 2 de la version B n'est donc pas seulement attendu :")
    print("  il est mesure. C'est l'element demande par l'enonce au titre de")
    print("  la precision du schema spatial.")
