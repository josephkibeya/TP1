"""
simulation_labyrinthe.py
------------------------
Troisieme module du projet "Volumes Finis - Labyrinthe" : l'ETUDE.

Ce fichier ne contient AUCUNE physique nouvelle. Toute la physique est deja
dans les deux modules precedents :

    vitesses.py    transforme le potentiel phi en vitesses aux faces
    advection.py   transporte une concentration c sur une grille quelconque

Ni l'un ni l'autre ne sait qu'il travaille sur un labyrinthe : ils prennent
des tableaux et rendent des tableaux. C'est volontaire, et c'est ce qui rend
ces modules reutilisables.

Ce fichier-ci ne fait donc que de l'assemblage :
    1. charger les donnees du professeur (phi et le domaine) ;
    2. en deduire le champ de vitesse et le pas de temps ;
    3. poser une condition initiale et une condition limite ;
    4. boucler en temps ;
    5. enregistrer les resultats et tracer les figures.

POURQUOI UN FICHIER SEPARE DU FICHIER DE TEST
=============================================
`test_advection_canal_rectiligne.py` fait tourner le MEME moteur sur un canal
droit de 7 cellules. La difference n'est pas dans le code, elle est dans ce
que l'on connait :

                            canal rectiligne        labyrinthe
    connait-on la solution ?      OUI                   NON

Dans le canal a Courant = 1, la solution exacte est connue (translation pure),
donc on peut verifier le code au chiffre pres. Dans le labyrinthe, personne ne
connait la solution : si le code etait faux, le resultat ressemblerait quand
meme a quelque chose.

    REGLE : on ne valide jamais un code sur le cas qu'on cherche a resoudre.
    On le valide la ou l'on connait la reponse, puis on l'applique la ou on
    ne la connait pas.

LES DEUX VERSIONS DEMANDEES PAR LE PROFESSEUR
=============================================
    Version A : reconstruction CONSTANTE + EULER explicite   -> ordre 1
    Version B : reconstruction LINEAIRE  + RK2 (Heun)        -> ordre 2

Elles ne sont pas appariees au hasard : la combinaison lineaire + Euler est
instable pour tout nu > 0 (voir la note dans advection.py et le test 4).

LES DEUX FACONS D'INTRODUIRE LE CONSTITUANT
===========================================
    - TACHE INITIALE  : une goutte posee au depart, plus aucun apport ensuite.
      Elle sert de TEST : la masse totale doit rester constante, donc
          masse initiale = masse encore dedans + masse sortie.
    - INJECTION CONTINUE : on maintient de l'encre a 100 juste au-dessus de
      l'entree. Elle sert d'ETUDE : au bout d'un moment, le chemin complet de
      l'entree vers la sortie est trace sur une seule image.

Les deux doivent faire apparaitre le MEME chemin : c'est une verification
croisee.
"""

import ctypes
import os
import sys
import numpy as np

import vitesses as V
import advection as A

# tqdm affiche une barre de progression. Il fait partie des dependances du
# professeur (voir laplace_prof/requirements.txt), il est donc deja installe.
# S'il manquait malgre tout, le module continue de fonctionner sans barre.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DATA = os.path.join(ICI, "..", "data")
DOSSIER_RESULTATS = os.path.join(ICI, "..", "resultats")


# =====================================================================
#  1. PREPARATION : des donnees du prof au champ de vitesse
# =====================================================================

def empeche_la_veille(actif=True):
    """
    Demande a Windows de ne pas se mettre en veille pendant le calcul.

    Sans cela, un portable qui s'endort GELE le processus Python : le calcul
    ne progresse plus, et on croit a tort que le code est lent. La duree
    mesuree est alors le temps de calcul PLUS toute la duree de la veille.

    Ce reglage n'est valable que pendant l'execution : rien n'est modifie de
    facon permanente dans les options de Windows. L'ecran a le droit de
    s'eteindre (on ne demande pas ES_DISPLAY_REQUIRED), seule la mise en
    veille du systeme est empechee.

    Limite connue : rabattre l'ecran d'un portable endort la machine malgre
    tout, ce reglage-la est separe. Sur macOS et Linux, la fonction ne fait
    rien et ne provoque pas d'erreur.
    """
    if sys.platform != "win32":
        return
    ES_CONTINUOUS = 0x80000000        # ce reglage reste actif
    ES_SYSTEM_REQUIRED = 0x00000001   # le systeme doit rester eveille
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if actif else 0))
    except Exception:
        pass                          # jamais bloquant


def pas_necessaires(domain, u_bord, v_bord, dt, marge=1.5, seuil=1e-3):
    """
    Estime combien de pas de temps il faut pour que le constituant ait
    parcouru tout le labyrinthe.

    L'idee est un simple temps de remplissage : le volume a remplir divise
    par le debit, puis divise par la duree d'un pas.

        nombre de pas = (mailles actives / debit entrant) / dt * marge

    "Mailles actives" = celles ou l'eau circule vraiment. Les impasses sont
    exclues : l'eau y est a l'arret, elles ne se rempliront jamais.

    La marge compense le fait que le front n'est pas net : il s'etale, donc
    les dernieres mailles mettent plus longtemps que ce calcul ne le dit.
    Mesure sur le labyrinthe 14 x 14, part du chemin coloree a plus de 50 :

        marge 1,20 -> 97,1 %     marge 1,70 -> 98,3 %
        marge 1,35 -> 97,6 %     marge 1,90 -> 98,6 %
        marge 1,50 -> 98,0 %     marge 2,40 -> 99,1 %

    Le remplissage sature : passer de 1,50 a 2,40 double presque le temps de
    calcul pour gagner un point. On retient donc 1,50.
    """
    uc = np.zeros_like(domain, dtype=float)
    vc = np.zeros_like(domain, dtype=float)
    uc[:, 1:-1] = 0.5 * (u_bord[:, 1:] + u_bord[:, :-1])
    vc[1:-1, :] = 0.5 * (v_bord[1:, :] + v_bord[:-1, :])
    actives = (domain > 0) & (np.hypot(uc, vc) > seuil)

    debit = 0.0
    for i, j in np.argwhere(domain == 2):
        if i > 0 and domain[i - 1, j] == 0:
            debit += abs(v_bord[i - 1, j])
    if debit <= 0:
        return 20000
    return int(actives.sum() / debit / dt * marge)


def prepare(nom="laby14", marge=0.8):
    """
    Charge le potentiel et le domaine, en deduit les vitesses aux faces et le
    pas de temps admissible.

    Les trois appels a `vitesses` DOIVENT se faire dans cet ordre :
        1. vitesses_aux_bords      la formule brute, partout
        2. annule_vitesses_murs    met 0 sur les faces de mur
        3. vitesses_entree_sortie  corrige les 42 faces d'entree/sortie
    Si on inverse 2 et 3, les faces laterales des mailles d'entree contiennent
    encore des vitesses fictives et le bilan de la maille est faux.

    Les fichiers du professeur sont ouverts en LECTURE SEULE : ce module ne
    les modifie jamais.

    Retourne
    --------
    domain, u_bord, v_bord, dt, infos
    """
    # Le professeur enregistre ses fichiers sous la forme <nom>_pot.npy et
    # <nom>_domain.txt. On suit la meme convention, ce qui permet de changer
    # de labyrinthe sans toucher une ligne de code :
    #     prepare("laby14")   le labyrinthe de travail
    #     prepare("out")      le grand labyrinthe fourni par le professeur
    pot = np.load(os.path.join(DOSSIER_DATA, nom + "_pot.npy"))
    domain = np.loadtxt(os.path.join(DOSSIER_DATA, nom + "_domain.txt"))

    u_bord, v_bord = V.vitesses_aux_bords(pot)
    u_bord, v_bord = V.annule_vitesses_murs(u_bord, v_bord, domain)
    u_bord, v_bord, infos_es = V.vitesses_entree_sortie(u_bord, v_bord, domain)

    dt, infos_dt = A.calcule_dt(u_bord, v_bord, domain, marge=marge)

    infos = {}
    infos.update(infos_es)
    infos.update(infos_dt)
    infos["mailles_fluides"] = int((domain > 0).sum())
    return domain, u_bord, v_bord, dt, infos


def reperage(domain):
    """
    Trouve, une fois pour toutes, les trois familles de mailles dont on aura
    besoin pour les conditions aux limites.

    Rappel : le solveur du professeur marque
        domain == 0   mur, ou contour ajoute autour de l'image
        domain == 1   ecoulement
        domain == 2   entree et sortie (condition forte sur phi)

    Retourne
    --------
    dehors    : toutes les mailles hors domaine (domain == 0)
    alimente  : les mailles du contour situees JUSTE AU-DESSUS de l'entree.
                Ce sont elles qui servent de valeur amont aux 21 faces
                d'entree : c'est la "citerne".
    evacue    : le reste de `dehors`. Tout ce qui y atterrit est sorti du
                domaine pour de bon.
    """
    dehors = (domain == 0)
    alimente = np.zeros_like(dehors)

    for i, j in np.argwhere(domain == 2):
        # une maille d'entree a le contour AU-DESSUS d'elle,
        # une maille de sortie a le contour EN DESSOUS.
        if i > 0 and domain[i - 1, j] == 0:
            alimente[i - 1, j] = True

    evacue = dehors & ~alimente
    return dehors, alimente, evacue


# =====================================================================
#  2. CONDITIONS AUX LIMITES ET CONDITION INITIALE
# =====================================================================

def fabrique_conditions(domain, c_injection=0.0):
    """
    Fabrique la fonction qui remet la frontiere a sa valeur imposee.

    C'est une condition limite FAIBLE (diapos 8 et 10 du professeur) : on
    n'ecrit JAMAIS dans une maille du domaine. On impose seulement la valeur
    des mailles exterieures, qui servent de valeur amont aux faces de la
    frontiere. L'encre entre donc uniquement PAR LE FLUX, et la quantite qui
    entre a chaque pas est calculable d'avance :

        dm = dt * dx * c_injection * (somme des 21 vitesses d'entree)

    Parametres
    ----------
    c_injection : 0.0  -> de l'eau propre entre (cas de la tache initiale)
                  100. -> injection continue d'encre

    Retourne
    --------
    conditions : fonction c -> c, a appliquer apres chaque pas ET a passer
                 a `pas_rk2` pour qu'elle soit aussi appliquee au brouillon.
    c_impose   : le champ exterieur impose (sert au bilan de masse)
    """
    dehors, alimente, evacue = reperage(domain)

    c_impose = np.zeros_like(domain, dtype=float)
    c_impose[alimente] = c_injection

    def conditions(c):
        c = c.copy()
        c[dehors] = c_impose[dehors]
        return c

    return conditions, c_impose


def tache_initiale(domain, hauteur=15, valeur=100.0):
    """
    Pose une tache d'encre juste sous l'entree, sur toute sa largeur.

    On la met la pour qu'elle parcoure le chemin complet : c'est ce qui permet
    de comparer le trajet de la tache et celui de l'injection continue.

    Parametres
    ----------
    hauteur : nombre de lignes occupees par la tache.
    """
    entree = [(i, j) for i, j in np.argwhere(domain == 2)
              if i > 0 and domain[i - 1, j] == 0]
    i0 = min(i for i, j in entree)
    j_min = min(j for i, j in entree)
    j_max = max(j for i, j in entree)

    c = np.zeros_like(domain, dtype=float)
    bloc = np.zeros_like(domain, dtype=bool)
    bloc[i0:i0 + hauteur, j_min:j_max + 1] = True
    bloc &= (domain > 0)            # on ne met pas d'encre dans un mur
    c[bloc] = valeur
    return c


# =====================================================================
#  3. LA BOUCLE EN TEMPS
# =====================================================================

def simule(domain, u_bord, v_bord, dt, c_depart, conditions, c_impose,
           version="B", n_pas=21000, instants=None, bavard=True, nom="",
           position=None, n_instants=6):
    """
    Fait avancer la concentration de `n_pas` pas de temps.

    LA BARRE DE PROGRESSION
    -----------------------
    Un calcul de 21 000 pas peut durer plusieurs dizaines de minutes. Sans
    affichage, rien ne bouge a l'ecran et on croit que le programme est
    bloque. La barre affiche donc en permanence :

        injection_B :  34%|########       | 7203/21000 [02:31<04:49, 47.6 pas/s]
                        ^      ^              ^           ^      ^        ^
                        |      |              |           |      |     vitesse
                        |      |         pas faits    ecoule  restant
                        |   avancement
                    le cas en cours

    La VITESSE (pas/s) est la plus utile : elle dit si la machine calcule
    vraiment. Si elle est elevee mais que le total dure tres longtemps, c'est
    que le processus a ete mis en pause (mise en veille), pas qu'il est lent.

    LES DEUX VERSIONS
    -----------------
        version = "A"  -> reconstruction constante + Euler explicite
        version = "B"  -> reconstruction lineaire  + RK2 (Heun)

    La version B passe `conditions` a `pas_rk2`, pour que la frontiere soit
    reimposee sur le brouillon de l'etape intermediaire. Sans cela, la maille
    nourriciere se vide pendant le brouillon et il n'entre plus que 92 % de
    l'encre prevue. La version A n'a qu'une etape, donc pas ce probleme.

    LE BILAN DE MASSE
    -----------------
    Apres chaque pas, les mailles hors domaine contiennent ce que le schema y
    a depose. En comparant au champ impose, on separe deux choses :

        ce qui est SORTI    = ce qui s'est accumule dans les mailles d'evacuation
        ce qui est ENTRE    = ce qui a quitte les mailles nourricieres

    On peut alors verifier, a tout instant :

        masse initiale + masse entree  =  masse encore dedans + masse sortie

    C'est la seule verification qui prouve que le schema est conservatif sur
    le vrai labyrinthe.

    Retourne
    --------
    snaps  : dict {numero de pas -> champ c}
    trace  : le maximum atteint par c dans chaque maille depuis le debut.
             Pour la tache, c'est l'image du chemin parcouru.
    bilan  : dict des masses (initiale, entree, sortie, dedans, ecart)
    """
    if instants is None:
        # n_instants photos reparties regulierement, la derniere a la fin
        instants = [round(n_pas * k / n_instants) for k in range(1, n_instants + 1)]

    dehors, alimente, evacue = reperage(domain)
    fluide = (domain > 0)

    c = conditions(c_depart.copy())
    masse_initiale = c[fluide].sum()
    trace = c.copy()

    masse_entree = 0.0
    masse_sortie = 0.0
    snaps = {}

    # La barre remplace le compteur muet. `ecrire` sert a afficher les lignes
    # de bilan sans casser l'affichage de la barre.
    pas_a_faire = range(1, n_pas + 1)
    barre = None
    if tqdm is not None:
        barre = tqdm(pas_a_faire, desc="%-14s" % (nom or version),
                     unit=" pas", ncols=92, leave=True, position=position)
        pas_a_faire = barre
    ecrire = (tqdm.write if (barre is not None) else print)

    for n in pas_a_faire:
        if version == "A":
            c = A.pas_euler(c, domain, u_bord, v_bord, dt, lineaire=False)
        else:
            c = A.pas_rk2(c, domain, u_bord, v_bord, dt, lineaire=True,
                          conditions=conditions)

        # --- bilan de masse, AVANT de remettre la frontiere en place ---
        masse_sortie += (c[evacue] - c_impose[evacue]).sum()
        masse_entree += -(c[alimente] - c_impose[alimente]).sum()

        c = conditions(c)
        np.maximum(trace, c, out=trace)

        if n in instants:
            snaps[n] = c.copy()
            if bavard:
                dedans = c[fluide].sum()
                ecart = masse_initiale + masse_entree - dedans - masse_sortie
                ecrire("    pas %6d   dedans=%12.2f  sortie=%12.2f  "
                       "entree=%12.2f  ecart=%.2e"
                       % (n, dedans, masse_sortie, masse_entree, ecart))

    if barre is not None:
        barre.close()

    dedans = c[fluide].sum()
    bilan = {
        "masse_initiale": masse_initiale,
        "masse_entree": masse_entree,
        "masse_sortie": masse_sortie,
        "masse_dedans": dedans,
        "ecart": masse_initiale + masse_entree - dedans - masse_sortie,
        "c_min": c[fluide].min(),
        "c_max": c[fluide].max(),
    }
    return snaps, trace, bilan


def encre_dans_les_impasses(c, domain, u_bord, v_bord, seuil=1e-3):
    """
    Mesure la part d'encre qui se trouve dans les mailles OU L'EAU NE BOUGE PAS.

    Dans un cul-de-sac, l'eau est a l'arret : elle ne peut pas y entrer, car il
    faudrait qu'elle ressorte par la meme ouverture, ce qui est impossible en
    regime permanent incompressible. Comme le constituant est purement advecte
    (aucune diffusion physique), AUCUNE encre ne devrait s'y trouver.

    Tout ce qu'on mesure ici est donc de la DIFFUSION NUMERIQUE : c'est un
    chiffre qui mesure le defaut du schema, et il doit etre plus grand pour la
    version A (ordre 1) que pour la version B (ordre 2).

    Retourne
    --------
    pourcentage de la masse totale situee dans les mailles a l'arret.
    """
    uc = np.zeros_like(c)
    vc = np.zeros_like(c)
    uc[:, 1:-1] = 0.5 * (u_bord[:, 1:] + u_bord[:, :-1])
    vc[1:-1, :] = 0.5 * (v_bord[1:, :] + v_bord[:-1, :])
    vitesse = np.hypot(uc, vc)

    fluide = (domain > 0)
    impasse = fluide & (vitesse < seuil)
    total = c[fluide].sum()
    if total == 0.0:
        return 0.0
    return 100.0 * c[impasse].sum() / total


# =====================================================================
#  4. LE PROGRAMME PRINCIPAL
# =====================================================================

def _un_cas(parametres):
    """
    Execute UN cas complet. Fonction de premier niveau (pas une fonction
    imbriquee) pour que `multiprocessing` puisse la transmettre a un autre
    processus : seuls des objets simples circulent, les gros tableaux sont
    relus dans chaque processus.
    """
    nom_laby, cas, version, n_pas, marge, position = parametres

    domain, u_bord, v_bord, dt, infos = prepare(nom_laby, marge=marge)
    c_injection = {"injection": 100.0, "tache": 0.0}[cas]
    conditions, c_impose = fabrique_conditions(domain, c_injection)
    c0 = (tache_initiale(domain) if cas == "tache"
          else np.zeros_like(domain, dtype=float))

    nom = "%s_%s" % (cas, version)
    snaps, trace, bilan = simule(domain, u_bord, v_bord, dt, c0, conditions,
                                 c_impose, version=version, n_pas=n_pas,
                                 nom=nom, position=position, bavard=False)

    bilan["impasses_%"] = encre_dans_les_impasses(snaps[max(snaps)], domain,
                                                  u_bord, v_bord)
    debit_encre = infos["debit_entrant"] * c_injection * dt
    if debit_encre > 0:
        bilan["injection_%"] = 100.0 * bilan["masse_entree"] / (debit_encre * n_pas)

    np.savez_compressed(
        os.path.join(DOSSIER_RESULTATS, nom + ".npz"),
        trace=trace.astype(np.float32),
        **{"pas_%d" % n: s.astype(np.float32) for n, s in snaps.items()})
    return nom, bilan


def etude_complete(nom_laby="laby14", n_pas=None, marge=0.8,
                   cas=("injection", "tache"), versions=("A", "B"),
                   marge_pas=1.5, parallele=True, instants=6):
    """
    Lance l'etude et enregistre tout dans ../resultats/.

    PARAMETRES
    ----------
    nom_laby : le labyrinthe a traiter. Les fichiers lus sont
        ../data/<nom_laby>_pot.npy et ../data/<nom_laby>_domain.txt.
            "laby14" : le labyrinthe de travail, 213 x 213, rapide
            "out"    : le grand labyrinthe fourni par le professeur, 502 x 502
    n_pas : None -> calcule automatiquement par `pas_necessaires`.
    cas : "injection" (le resultat) et/ou "tache" (le test de conservation).
    parallele : True -> les cas tournent en meme temps, sur des coeurs
        differents. Ils sont totalement independants : aucun n'attend le
        resultat d'un autre. Le temps total devient celui du plus long au
        lieu de la somme.
    instants : nombre d'instantanes enregistres par cas.

    A QUOI SERT CHAQUE CAS
    ----------------------
        injection continue : LE RESULTAT. Le chemin trace, et la comparaison
            de la diffusion numerique entre les deux versions.
        tache initiale     : LE TEST. C'est le seul cas ou l'on peut ecrire
            en une ligne   masse initiale = masse restante + masse sortie,
            puisque rien n'entre apres le depart. C'est aussi celui qui
            separe le mieux les deux schemas.
    """
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
    empeche_la_veille(True)
    try:
        domain, u_bord, v_bord, dt, infos = prepare(nom_laby, marge=marge)
        if n_pas is None:
            n_pas = pas_necessaires(domain, u_bord, v_bord, dt, marge_pas)

        print("Labyrinthe %s : %d x %d, %d mailles fluides"
              % (nom_laby, domain.shape[0], domain.shape[1],
                 infos["mailles_fluides"]))
        print("dt = %.4f   (Courant = %.2f)   |u|max = %.4f   |v|max = %.4f"
              % (dt, infos["courant"], infos["u_max"], infos["v_max"]))
        print("debit entrant = %.6f   debit sortant = %.6f   ecart = %.1e"
              % (infos["debit_entrant"], infos["debit_sortant"],
                 infos["ecart_entrant_sortant"]))
        print("%d pas de temps   |   %s   |   versions %s   |   %s"
              % (n_pas, ", ".join(cas), ", ".join(versions),
                 "en parallele" if parallele else "en serie"))
        print()

        travaux = [(nom_laby, un_cas, version, n_pas, marge, k)
                   for k, (un_cas, version) in enumerate(
                       [(a, b) for a in cas for b in versions])]

        if parallele and len(travaux) > 1:
            from multiprocessing import Pool, RLock
            # Chaque processus ecrit sa barre de progression dans le MEME
            # terminal. Sans verrou partage, les barres se melangent et une
            # meme ligne s'affiche deux fois. `tqdm.set_lock` donne a tous
            # les processus le meme verrou : chacun attend son tour pour
            # ecrire, et chaque cas garde sa ligne.
            demarrage, arguments = (None, ())
            if tqdm is not None:
                demarrage, arguments = tqdm.set_lock, (RLock(),)
            with Pool(processes=min(len(travaux), os.cpu_count() or 1),
                      initializer=demarrage, initargs=arguments) as pool:
                resultats = pool.map(_un_cas, travaux)
            if tqdm is not None:
                print("\n" * len(travaux))   # on passe sous les barres
        else:
            resultats = [_un_cas(t) for t in travaux]

        np.save(os.path.join(DOSSIER_RESULTATS, "domaine_%s.npy" % nom_laby),
                domain.astype(np.int8))
        return dict(resultats), dt, n_pas
    finally:
        empeche_la_veille(False)


if __name__ == "__main__":
    import time
    t0 = time.time()
    resume, dt, n_pas = etude_complete()
    duree = time.time() - t0

    print()
    print("=" * 78)
    print("RESUME   (%d pas, %.0f s au total)" % (n_pas, duree))
    print("=" * 78)
    print("%-14s %13s %15s %10s %10s" %
          ("cas", "ecart bilan", "encre impasses", "c min", "c max"))
    print("-" * 78)
    for nom in sorted(resume):
        b = resume[nom]
        print("%-14s %13.2e %13.2f %% %10.2f %10.2f"
              % (nom, b["ecart"], b["impasses_%"], b["c_min"], b["c_max"]))
