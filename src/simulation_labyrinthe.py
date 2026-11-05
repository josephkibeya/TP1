"""
simulation_labyrinthe.py
------------------------
Troisieme module du projet "Volumes Finis - Labyrinthe" : l'application au
labyrinthe.

Ce module n'introduit aucune physique nouvelle. Celle-ci est entierement
contenue dans les deux modules precedents :

    vitesses.py    transforme le potentiel phi en vitesses aux faces
    advection.py   transporte une concentration c sur une grille quelconque

Aucun des deux ne suppose que le domaine est un labyrinthe : ils recoivent
des tableaux et en renvoient. Ce choix les rend reutilisables sur un autre
domaine sans modification.

Le present module se limite donc a l'assemblage :
    1. charger le potentiel et le domaine ;
    2. en deduire le champ de vitesse et le pas de temps ;
    3. poser une condition initiale et une condition limite ;
    4. boucler en temps ;
    5. enregistrer les resultats et tracer les figures.

SEPARATION DE L'APPLICATION ET DE LA VALIDATION
===============================================
`test_advection_canal_rectiligne.py` applique le meme moteur a un canal droit
de sept cellules. La difference entre les deux cas ne porte pas sur le code,
mais sur ce qui est connu :

                            canal rectiligne        labyrinthe
    solution analytique           oui                   non

Dans le canal a nombre de Courant unitaire, la solution exacte est une
translation pure, ce qui permet de controler le code au chiffre pres. Dans le
labyrinthe, aucune solution de reference n'existe : un code errone y
produirait neanmoins un resultat d'apparence plausible.

La validation est donc conduite la ou la reponse est connue, puis le code est
applique la ou elle ne l'est pas.

VERSIONS ETUDIEES
=================
    Version A : reconstruction constante + Euler explicite   ordre 1
    Version B : reconstruction lineaire  + RK2 (Heun)        ordre 2

L'appariement n'est pas arbitraire : la combinaison reconstruction lineaire
et Euler explicite est instable pour tout nu > 0, comme l'illustre le test 4
de `test_advection_canal_rectiligne.py`.

MODES D'INTRODUCTION DU CONSTITUANT
===================================
    - TACHE INITIALE : une quantite posee au depart, sans apport ulterieur.
      Ce cas sert de controle : la masse totale doit rester constante, soit
          masse initiale = masse encore presente + masse sortie.
    - INJECTION CONTINUE : la concentration est maintenue a 100 juste au-dessus
      de l'entree. Ce cas sert a l'etude : apres un temps suffisant, le chemin
      complet de l'entree vers la sortie apparait sur une seule image.

Les deux modes doivent faire apparaitre le meme chemin, ce qui constitue une
verification croisee.
"""

import ctypes
import os
import sys
import numpy as np

import vitesses as V
import advection as A

# tqdm affiche une barre de progression. Il figure parmi les dependances du
# solveur fourni (laplace_prof/requirements.txt) et est donc deja installe.
# En son absence, le module reste fonctionnel, sans barre de progression.
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DATA = os.path.join(ICI, "..", "data")
DOSSIER_RESULTATS = os.path.join(ICI, "..", "resultats")


# =====================================================================
#  1. PREPARATION : du potentiel au champ de vitesse
# =====================================================================

def empeche_la_veille(actif=True):
    """
    Demande a Windows de ne pas se mettre en veille pendant le calcul.

    En l'absence de ce reglage, la mise en veille suspend le processus
    Python : le calcul cesse de progresser et la duree mesuree devient la
    somme du temps de calcul et du temps de veille, ce qui fausse toute
    mesure de performance.

    Le reglage ne vaut que pour la duree de l'execution ; aucune option de
    Windows n'est modifiee de facon permanente. L'extinction de l'ecran reste
    autorisee, ES_DISPLAY_REQUIRED n'etant pas demande : seule la mise en
    veille du systeme est empechee.

    Limite connue : la fermeture de l'ecran d'un portable endort la machine
    malgre ce reglage, ce comportement relevant d'une option distincte. Sur
    macOS et Linux, la fonction est sans effet et ne provoque pas d'erreur.
    """
    if sys.platform != "win32":
        return
    ES_CONTINUOUS = 0x80000000        # le reglage reste actif
    ES_SYSTEM_REQUIRED = 0x00000001   # le systeme doit rester eveille
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if actif else 0))
    except Exception:
        pass                          # non bloquant


def pas_necessaires(domain, u_bord, v_bord, dt, marge=1.5, seuil=1e-3):
    """
    Estime combien de pas de temps il faut pour que le constituant ait
    parcouru tout le labyrinthe.

    L'estimation repose sur un temps de remplissage : le volume a remplir
    divise par le debit, puis par la duree d'un pas.

        nombre de pas = (mailles actives / debit entrant) / dt * marge

    Les mailles actives sont celles ou l'ecoulement est effectif. Les impasses
    en sont exclues : le fluide y est au repos et elles ne se remplissent
    jamais.

    La marge tient compte de l'etalement du front, qui n'est pas net : les
    dernieres mailles se remplissent plus tard que ne le predit ce calcul.
    Mesure sur le labyrinthe 14 x 14, part du chemin coloree a plus de 50 :

        marge 1,20 -> 97,1 %     marge 1,70 -> 98,3 %
        marge 1,35 -> 97,6 %     marge 1,90 -> 98,6 %
        marge 1,50 -> 98,0 %     marge 2,40 -> 99,1 %

    Le remplissage sature : passer de 1,50 a 2,40 double presque le temps de
    calcul pour un gain d'un point de pourcentage. La valeur 1,50 est donc
    retenue.
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

    Les trois appels au module `vitesses` doivent respecter cet ordre :
        1. vitesses_aux_bords      la formule brute, sur tout le domaine
        2. annule_vitesses_murs    annulation des faces de mur
        3. vitesses_entree_sortie  correction des faces d'entree et de sortie
    Dans l'ordre inverse des etapes 2 et 3, les faces laterales des mailles
    d'entree portent encore des vitesses fictives et le bilan de la maille est
    faux.

    Les fichiers fournis sont ouverts en lecture seule : ce module ne les
    modifie jamais.

    Retourne
    --------
    domain, u_bord, v_bord, dt, infos
    """
    # Le solveur enregistre ses fichiers sous la forme <nom>_pot.npy et
    # <nom>_domain.txt. La meme convention est reprise ici, ce qui permet de
    # changer de labyrinthe sans modifier le code :
    #     prepare("laby14")   le labyrinthe de travail
    #     prepare("out")      le grand labyrinthe fourni en exemple
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
    Identifie une fois pour toutes les trois familles de mailles necessaires
    aux conditions aux limites.

    Le solveur fourni marque
        domain == 0   mur, ou contour ajoute autour de l'image
        domain == 1   ecoulement
        domain == 2   entree et sortie (condition forte sur phi)

    Retourne
    --------
    dehors    : toutes les mailles hors domaine (domain == 0)
    alimente  : les mailles du contour situees immediatement au-dessus de
                l'entree. Elles fournissent la valeur amont des faces
                d'entree et constituent donc le reservoir d'alimentation.
    evacue    : le reste de `dehors`. Toute masse qui y parvient a
                definitivement quitte le domaine.
    """
    dehors = (domain == 0)
    alimente = np.zeros_like(dehors)

    for i, j in np.argwhere(domain == 2):
        # Une maille d'entree a le contour au-dessus d'elle, une maille de
        # sortie l'a en dessous.
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

    Il s'agit d'une condition limite faible : aucune maille du domaine n'est
    jamais ecrite. Seule la valeur des mailles exterieures est imposee, ces
    mailles fournissant la valeur amont des faces de la frontiere. Le
    constituant n'entre donc que par le flux, et la quantite introduite a
    chaque pas est connue a l'avance :

        dm = dt * dx * c_injection * (somme des vitesses des faces d'entree)

    Parametres
    ----------
    c_injection : 0.0  -> entree d'eau propre (cas de la tache initiale)
                  100. -> injection continue de constituant

    Retourne
    --------
    conditions : fonction c -> c, a appliquer apres chaque pas et a
                 transmettre a `pas_rk2`, afin qu'elle s'applique aussi a
                 l'etat intermediaire.
    c_impose   : champ exterieur impose, utilise pour le bilan de masse
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
    Place une tache de constituant immediatement sous l'entree, sur toute sa
    largeur.

    Cette position lui fait parcourir le chemin complet, ce qui permet de
    comparer son trajet a celui de l'injection continue.

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
    bloc &= (domain > 0)            # aucun constituant dans un mur
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

    SUIVI DE L'AVANCEMENT
    ---------------------
    Un calcul de plusieurs milliers de pas peut durer plusieurs dizaines de
    minutes. La barre de progression affiche en continu le cas traite,
    l'avancement, le nombre de pas effectues, le temps ecoule, le temps
    restant estime et la cadence en pas par seconde.

    Cette cadence est la grandeur la plus informative : une cadence elevee
    associee a une duree totale anormalement longue signale une suspension du
    processus, par mise en veille, et non une lenteur du calcul.

    VERSIONS
    --------
        version = "A"  -> reconstruction constante + Euler explicite
        version = "B"  -> reconstruction lineaire  + RK2 (Heun)

    La version B transmet `conditions` a `pas_rk2`, de maniere que la
    frontiere soit reimposee sur l'etat intermediaire. A defaut, la maille
    d'alimentation se vide pendant cette etape et seuls 90,63 % du constituant
    prevu penetrent dans le domaine : mesure sur laby14, injection continue a
    100, sur 300 pas. La version A ne comporte qu'une etape et n'est pas
    concernee.

    BILAN DE MASSE
    --------------
    Apres chaque pas, les mailles hors domaine contiennent ce que le schema y
    a depose. La comparaison au champ impose separe deux contributions :

        masse sortie   = accumulation dans les mailles d'evacuation
        masse entree   = quantite ayant quitte les mailles d'alimentation

    La relation suivante peut alors etre verifiee a tout instant :

        masse initiale + masse entree  =  masse encore presente + masse sortie

    C'est le seul controle qui etablisse le caractere conservatif du schema
    sur le labyrinthe lui-meme.

    Retourne
    --------
    snaps  : dict {numero de pas -> champ c}
    trace  : maximum atteint par c dans chaque maille depuis le debut. Pour la
             tache, il donne l'image du chemin parcouru.
    bilan  : dict des masses (initiale, entree, sortie, presente, ecart)
    """
    if instants is None:
        # n_instants releves regulierement espaces, le dernier en fin de calcul
        instants = [round(n_pas * k / n_instants) for k in range(1, n_instants + 1)]

    dehors, alimente, evacue = reperage(domain)
    fluide = (domain > 0)

    c = conditions(c_depart.copy())
    masse_initiale = c[fluide].sum()
    trace = c.copy()

    masse_entree = 0.0
    masse_sortie = 0.0
    snaps = {}

    # `ecrire` permet d'afficher les lignes de bilan sans perturber le rendu
    # de la barre de progression.
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

        # --- bilan de masse, avant reimposition de la frontiere ---
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
    Mesure la part du constituant situee dans les mailles ou le fluide est au
    repos.

    Dans une impasse, le debit net par la seule ouverture est necessairement
    nul en regime permanent incompressible : le fluide y est donc a l'arret.
    Le constituant etant purement advecte, en l'absence de tout terme de
    diffusion physique, aucune masse ne devrait s'y trouver.

    La quantite mesuree ici est par consequent entierement imputable a la
    diffusion numerique. Elle constitue une mesure du defaut du schema et doit
    etre plus elevee pour la version A, d'ordre 1, que pour la version B,
    d'ordre 2.

    Retourne
    --------
    pourcentage de la masse totale situee dans les mailles au repos.
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
    Execute un cas complet. La fonction est definie au premier niveau, et non
    imbriquee, afin que `multiprocessing` puisse la transmettre a un autre
    processus : seuls des objets simples circulent, les tableaux volumineux
    etant relus dans chaque processus.
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
            "out"    : le grand labyrinthe fourni en exemple, 502 x 502
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
