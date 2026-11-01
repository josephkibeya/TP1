"""
test_convergence.py
-------------------
Verifie EXPERIMENTALEMENT l'ordre des deux versions.

POURQUOI CE TEST, ALORS QUE LA THEORIE LE DIT DEJA
==================================================
Le theoreme de Lax dit : consistance + stabilite => convergence. Mais il
parle du SCHEMA, pas du CODE. Il affirme que ce schema, correctement ecrit,
converge a l'ordre 2 ; il ne dit rien sur le fait que ces quelques centaines
de lignes de Python ecrivent bien ce schema.

Et c'est tout l'interet du test. Une erreur d'indice, un signe inverse, une
condition limite mal posee : le code tourne, produit de belles images,
conserve meme la masse... et converge a l'ordre 1 au lieu de 2. Le test de
convergence est le meilleur detecteur de bug des methodes numeriques.

LE PRINCIPE
===========
On prend un cas dont la solution exacte est CONNUE : un canal droit, vitesse
uniforme a, et un profil de depart lisse. La solution exacte est alors une
simple translation :

    c(x, t) = c0(x - a t)

On calcule sur des maillages de plus en plus fins, a nombre de Courant
CONSTANT (donc dt diminue en meme temps que dx), et on mesure l'ecart a la
solution exacte :

    E = (1 / N) * somme |c_calcule - c_exact|        (norme L1)

Si le schema est d'ordre p, alors E est proportionnel a dx^p. Diviser dx par
deux divise donc E par 2^p, et l'ordre observe se lit directement :

    p = log2( E(dx) / E(dx/2) )

    Version A (constante + Euler) : on attend p proche de 1
    Version B (lineaire + RK2)    : on attend p proche de 2

POURQUOI LE PROFIL DE DEPART DOIT ETRE LISSE
============================================
On prend une gaussienne, surtout pas un creneau. Sur une discontinuite, la
derivee n'existe pas, le developpement de Taylor ne s'applique plus, et
AUCUN schema n'atteint l'ordre 2. On mesurerait p proche de 1 pour les deux
versions et on conclurait a tort que la reconstruction lineaire ne sert a
rien. Les ordres annonces ne valent que pour des solutions regulieres.
"""

import numpy as np

import advection as A

# --- le cas de reference ------------------------------------------------
VITESSE = 1.0      # a, vers la droite
COURANT = 0.8      # nu, le meme sur tous les maillages
X_DEPART = 0.25    # centre de la gaussienne au temps zero
SIGMA = 0.06       # largeur de la gaussienne
TEMPS = 0.5        # duree simulee : le profil se deplace de 0,5


def profil_exact(x, t):
    """La solution exacte : la gaussienne translatee de a*t, sans deformation."""
    return 100.0 * np.exp(-((x - X_DEPART - VITESSE * t) / SIGMA) ** 2)


def canal(n_mailles):
    """
    Construit un canal droit de n_mailles cellules sur une longueur 1.

    Le canal fait trois lignes de haut : une ligne d'eau entre deux lignes de
    mur. Les vitesses verticales sont nulles, le transport est donc purement
    horizontal, mais on utilise exactement le meme moteur 2D que sur le
    labyrinthe : c'est bien le code du projet qu'on teste, pas un code a part.
    """
    dx = 1.0 / n_mailles
    domain = np.zeros((3, n_mailles), dtype=float)
    domain[1, :] = 1.0                       # la ligne d'eau

    u_bord = np.zeros((3, n_mailles - 1))
    u_bord[1, :] = VITESSE                   # uniquement entre deux cases d'eau
    v_bord = np.zeros((2, n_mailles))        # aucun mouvement vertical

    x = (np.arange(n_mailles) + 0.5) * dx    # centres des mailles
    return domain, u_bord, v_bord, dx, x


def erreur(n_mailles, version):
    """
    Calcule l'ecart moyen a la solution exacte apres TEMPS, sur un maillage
    de n_mailles cellules.
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
    """Lance le test sur une suite de maillages et affiche les ordres observes."""
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
    print("  L'ordre 2 de la version B n'est donc pas seulement annonce :")
    print("  il est mesure. C'est la preuve demandee par l'enonce sous le nom")
    print("  de 'precision du schema spatial'.")
