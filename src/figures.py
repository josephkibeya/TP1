"""
figures.py
----------
Produit TOUTES les images et les deux videos du projet, a partir des fichiers
que `simulation_labyrinthe.py` a enregistres dans ../resultats/.

LA REGLE DU DEPOT
=================
Tout ce qui se trouve dans ../resultats/ est produit par du code de ../src/ :
les .npz et le .npy par simulation_labyrinthe.py, les .png et les .mp4 par ce
fichier. Aucune image n'est fabriquee a la main, aucun recadrage n'est choisi
en dehors du code. Un resultat qu'on ne sait pas refaire n'est pas un resultat.

AUCUN CHIFFRE N'EST ECRIT EN DUR
================================
Les legendes affichent dt, le nombre de pas, les bornes de c et le pourcentage
d'encre dans les impasses. Tous ces nombres sont RECALCULES ici, jamais recopies.
Si on change de labyrinthe, les legendes suivent toutes seules.

DEUX PORTES D'ENTREE
====================
    figures()   relit les .npz deja calcules et ecrit les 2 .png     < 5 s
    videos()    relance les deux schemas et ecrit les 2 .mp4         ~2 min

Pourquoi videos() est obligee de recalculer : un .npz ne garde que 6 instants,
une video fluide en demande 240. Il n'y a pas de raccourci.

Mais recalculer n'est pas risque, parce qu'il n'y a AUCUN hasard dans le
moteur : ni random, ni rng, ni seed dans advection.py, vitesses.py et
simulation_labyrinthe.py. Le seul tirage du projet est dans
genere_labyrinthe.py, et sa graine est fixee a 11.

videos() le VERIFIE quand meme : aux 6 pas que le .npz a enregistres, elle
compare son propre etat a celui qui est sur le disque et affiche l'ecart.

    pas   1625 : ecart max au .npz   A = 0.000e+00   B = 0.000e+00

Si une ligne n'affiche pas zero, quelque chose a bouge, et on le sait AVANT
de regarder la video.

COMMENT LE TITRE "VERSION A" ARRIVE SUR LA BONNE MOITIE
=======================================================
Le programme ne le devine pas : c'est lui qui a fait le calcul. Dans la boucle,

    ca = conditions(A.pas_euler(ca, ...))     <- ceci est la version A
    cb = conditions(A.pas_rk2  (cb, ...))     <- ceci est la version B

l'etiquette est sur la meme ligne que le calcul. Elles ne peuvent pas se
contredire.

ASSEMBLAGE DES VIDEOS
=====================
Il faut ffmpeg. Le plus simple est le paquet pip qui le contient deja :

    pip install imageio-ffmpeg

Ce module le cherche d'abord dans le PATH, puis dans ce paquet. S'il ne le
trouve nulle part, il GARDE les images et affiche la commande a lancer, au
lieu de planter.
"""

import os
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")           # pas de fenetre : on ecrit des fichiers
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import advection as A
import simulation_labyrinthe as S

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_RESULTATS = os.path.join(ICI, "..", "resultats")

COULEUR_MUR = (0.16, 0.17, 0.21)
FOND_BANDEAU = (27, 32, 42)
CARTE = matplotlib.colormaps["turbo"]


# =====================================================================
#  1. PEINDRE UN CHAMP DE CONCENTRATION
# =====================================================================

def peint(c, domain, vmax=100.0):
    """
    Transforme un champ de concentration en image couleur.

    Les murs sont gris fonce, l'eau propre est blanche, l'encre suit la
    palette "turbo". La puissance 0,33 sur la concentration etire le bas de
    l'echelle : sans elle, les tres faibles concentrations laissees par la
    diffusion numerique seraient invisibles, et c'est precisement ce qu'on
    cherche a comparer entre les deux versions.

    Retourne un tableau de flottants entre 0 et 1, de forme (ni, nj, 3).
    """
    mur = (domain == 0)
    img = np.empty(domain.shape + (3,))
    img[...] = [1.0, 1.0, 1.0]
    img[mur] = COULEUR_MUR

    f = np.clip(c / vmax, 0.0, 1.0)
    g = f ** 0.33
    encre = CARTE(np.clip(0.30 + 0.70 * g, 0.0, 1.0))[..., :3]
    opacite = np.clip(g * 1.6, 0.0, 1.0)[..., None]

    visible = (~mur)[..., None] & (f[..., None] > 1e-4)
    return np.where(visible, img * (1 - opacite) + encre * opacite, img)


def _en_octets(image_flottante):
    """Passe d'une image de flottants 0..1 a une image d'octets 0..255."""
    return (image_flottante * 255).astype(np.uint8)


def _agrandi(image, k):
    """
    Agrandit l'image d'un facteur entier k, sans lissage.

    On duplique les pixels (np.kron) au lieu d'interpoler : une maille doit
    rester une maille. Un lissage inventerait des valeurs intermediaires qui
    n'existent pas dans le calcul.
    """
    return np.kron(image, np.ones((k, k, 1), np.uint8))


# =====================================================================
#  2. ECRIRE DU TEXTE SUR UNE IMAGE
# =====================================================================

def _police(taille, gras=False):
    """
    Charge DejaVu Sans depuis matplotlib.

    On ne met pas de chemin en dur vers une police du systeme : matplotlib
    est deja une dependance du projet et embarque cette police, donc le code
    trouve la meme sur n'importe quelle machine.
    """
    nom = "DejaVuSans-Bold.ttf" if gras else "DejaVuSans.ttf"
    chemin = os.path.join(os.path.dirname(matplotlib.__file__),
                          "mpl-data", "fonts", "ttf", nom)
    try:
        return ImageFont.truetype(chemin, taille)
    except OSError:
        return ImageFont.load_default()


def _bande(largeur, hauteur):
    """Fabrique une bande de couleur unie, de la largeur de l'image."""
    bande = np.empty((hauteur, largeur, 3), np.uint8)
    bande[...] = FOND_BANDEAU
    return bande


def _habille(panneaux, titres, bas, largeur_titre=None):
    """
    Assemble les panneaux cote a cote et ajoute les bandeaux de texte.

    panneaux : liste d'images d'octets, toutes de meme hauteur.
    titres   : liste de (titre, sous_titre), un par panneau.
    bas      : la ligne de texte du bandeau inferieur, centree.

    Retourne une image PIL prete a etre enregistree.
    """
    separateur = np.full((panneaux[0].shape[0], 6, 3), 255, np.uint8)
    corps = panneaux[0]
    for suivant in panneaux[1:]:
        corps = np.hstack([corps, separateur, suivant])

    largeur = corps.shape[1]
    haut = _bande(largeur, 54)
    bas_bande = _bande(largeur, 36)
    image = Image.fromarray(np.vstack([haut, corps, bas_bande]))

    dessin = ImageDraw.Draw(image)
    grande, petite = _police(19, gras=True), _police(13)

    # chaque titre est pose au-dessus de SON panneau
    pas_colonne = largeur_titre if largeur_titre else (panneaux[0].shape[1] + 6)
    for k, (titre, sous_titre) in enumerate(titres):
        x = 15 + k * pas_colonne
        dessin.text((x, 8), titre, font=grande, fill=(255, 255, 255))
        dessin.text((x, 32), sous_titre, font=petite, fill=(158, 170, 190))

    # le bandeau du bas, centre
    moyenne = _police(15)
    l_texte = dessin.textlength(bas, font=moyenne)
    dessin.text(((largeur - l_texte) / 2, image.height - 27), bas,
                font=moyenne, fill=(222, 228, 238))
    return image


# =====================================================================
#  3. LES DEUX FIGURES FIXES
# =====================================================================

def _contexte(nom_laby="laby14"):
    """
    Recharge ce qu'il faut pour legender : le domaine, les vitesses, dt.

    C'est ce qui permet de ne RIEN ecrire en dur dans les legendes.
    """
    domain, u_bord, v_bord, dt, infos = S.prepare(nom_laby)
    return domain, u_bord, v_bord, dt, infos


def _charge(nom):
    return np.load(os.path.join(DOSSIER_RESULTATS, nom + ".npz"))


def _instants(archive):
    """Retrouve les numeros de pas enregistres, lus dans les cles du .npz."""
    return sorted(int(k.split("_")[1]) for k in archive.files
                  if k.startswith("pas_"))


def _cadre(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for cote in ax.spines.values():
        cote.set_color("#c9ced6")


DESCRIPTION = {"A": "constante + Euler", "B": "lineaire + RK2"}
ORDRE = {"A": 1, "B": 2}


def _echelle(cas, domain):
    """
    Choisit le haut de l'echelle de couleur, COMMUN aux deux versions d'un
    meme cas.

    Pour l'injection, c'est 100 : la valeur injectee.

    Pour la tache, c'est autre chose. Rien n'entre apres le depart, donc la
    tache se dilue en avancant et son maximum tombe bien en dessous de 100 :
    a echelle fixe, les dernieres images seraient presque blanches. On prend
    donc le maximum atteint, tous instants et LES DEUX VERSIONS confondues.

    Le "les deux versions confondues" est essentiel. Si chaque figure avait sa
    propre echelle, la meme couleur ne representerait pas la meme
    concentration d'une figure a l'autre, et les comparer serait trompeur.
    """
    if cas == "injection":
        return 100.0
    fluide = (domain > 0)
    return max(_charge("%s_%s" % (cas, v))["pas_%d" % n][fluide].max()
               for v in ("A", "B")
               for n in _instants(_charge("%s_%s" % (cas, v))))


def figure_six_instants(cas="injection", version="B", nom_laby="laby14",
                        sortie=None):
    """
    Les six instants d'UN cas, dans UNE version. Une figure par cas.

    POURQUOI UNE FIGURE PAR CAS, ET PAS LES DEUX VERSIONS SUPERPOSEES
    ----------------------------------------------------------------
    La comparaison simultanee des deux schemas est deja faite, et bien mieux,
    par la video A_contre_B.mp4 : elle les montre cote a cote, au meme
    instant, sur 240 images au lieu de 6. Entasser les deux versions dans une
    figure fixe referait mal ce travail, et rendrait chaque labyrinthe deux
    fois plus petit.

    Ces figures ont donc un autre role : documenter chaque cas en grand, pour
    qu'on puisse y lire le detail d'un couloir. Quatre cas, quatre figures,
    toutes au meme format et a la meme echelle de couleur.

    cas     : "injection" (l'encre entre en continu) ou "tache" (une tache
              posee au depart, et plus rien n'entre ensuite).
    version : "A" (constante + Euler) ou "B" (lineaire + RK2).
    """
    if sortie is None:
        sortie = "fig1_%s_%s_6_instants.png" % (cas, version)

    domain, u_bord, v_bord, dt, infos = _contexte(nom_laby)
    archive = _charge("%s_%s" % (cas, version))
    instants = _instants(archive)
    n_pas = max(instants)
    vmax = _echelle(cas, domain)

    fig = plt.figure(figsize=(13.2, 9.4), facecolor="white")
    for k, n in enumerate(instants):
        ax = fig.add_subplot(2, 3, k + 1)
        ax.imshow(peint(archive["pas_%d" % n], domain, vmax),
                  interpolation="nearest")
        ax.set_title("pas %d   -   t = %.0f" % (n, n * dt),
                     fontsize=11, color="#222", pad=7)
        _cadre(ax)

    titre = ("Injection continue" if cas == "injection"
             else "Tache initiale (rien n'entre apres le depart)")
    fig.suptitle("%s, Version %s (%s)  -  six instants"
                 % (titre, version, DESCRIPTION[version]),
                 fontsize=17, fontweight="bold", color="#111", y=0.982)
    fig.text(0.5, 0.928,
             "%s, %d x %d px   |   Courant = %.2f   |   %d pas   |   "
             "dt = %.4f   |   echelle 0 a %.1f"
             % (nom_laby, domain.shape[1], domain.shape[0],
                infos["courant"], n_pas, dt, vmax),
             ha="center", fontsize=11.5, color="#5a6068")
    fig.subplots_adjust(left=.02, right=.98, top=.885, bottom=.02,
                        wspace=.05, hspace=.12)

    chemin = os.path.join(DOSSIER_RESULTATS, sortie)
    fig.savefig(chemin, dpi=115, facecolor="white")
    plt.close(fig)
    print("ecrit : resultats/%s" % sortie)
    return chemin


def figure_quatre_cas(nom_laby="laby14", sortie="fig2_quatre_cas.png"):
    """
    Figure 2 : les quatre cas, A contre B, injection et tache.

    En haut l'etat FINAL de l'injection : on y lit les depassements de la
    version B (c sort de [0 ; 100], ce qu'aucune concentration physique ne
    fait). En bas la TRACE de la tache : toutes les mailles ou l'encre est
    passee, donc l'etendue de la diffusion numerique, mesuree par le
    pourcentage d'encre arrivee dans les impasses.

    Les quatre pourcentages et les quatre couples (c_min, c_max) sont
    RECALCULES ici a partir des .npz.
    """
    domain, u_bord, v_bord, dt, infos = _contexte(nom_laby)
    fluide = (domain > 0)
    blocs = []

    for cas in ("injection", "tache"):
        for version in ("A", "B"):
            archive = _charge("%s_%s" % (cas, version))
            n_pas = max(_instants(archive))
            final = archive["pas_%d" % n_pas]
            impasses = S.encre_dans_les_impasses(final, domain, u_bord, v_bord)

            if cas == "injection":
                champ = final
                legende = ("INJECTION  -  Version %s\n%s   -   c va de %.2f a %.2f"
                           % (version,
                              "constante + Euler" if version == "A"
                              else "lineaire + RK2",
                              final[fluide].min(), final[fluide].max()))
            else:
                champ = archive["trace"]
                legende = ("TACHE, trace du parcours  -  Version %s\n"
                           "%.2f %% d'encre dans les impasses"
                           % (version, impasses))
            blocs.append((champ, legende))

    fig = plt.figure(figsize=(12.4, 13.2), facecolor="white")
    for k, (champ, legende) in enumerate(blocs):
        ax = fig.add_subplot(2, 2, k + 1)
        ax.imshow(peint(champ, domain), interpolation="nearest")
        ax.set_title(legende, fontsize=11.6, color="#1b2a41", pad=9,
                     linespacing=1.6)
        _cadre(ax)

    fig.suptitle("%s  -  les quatre cas" % nom_laby,
                 fontsize=17.5, fontweight="bold", color="#111", y=0.988)
    fig.text(0.5, 0.952,
             "en haut l'etat final de l'injection  |  en bas la TRACE de la "
             "tache : toutes les cases ou l'encre est passee",
             ha="center", fontsize=11.5, color="#5a6068")
    fig.subplots_adjust(left=.02, right=.98, top=.875, bottom=.015,
                        wspace=.05, hspace=.14)

    chemin = os.path.join(DOSSIER_RESULTATS, sortie)
    fig.savefig(chemin, dpi=115, facecolor="white")
    plt.close(fig)
    print("ecrit : resultats/%s" % sortie)
    return chemin


def figures(nom_laby="laby14"):
    """
    Ecrit les trois figures fixes. Ne recalcule aucune simulation.

    La regle est la symetrie : ce qui est montre pour une version est montre
    pour l'autre, au meme format et a la meme echelle de couleur. Sans cela,
    la comparaison A contre B ne vaut rien.

        fig1_injection_A_6_instants     fig1_injection_B_6_instants
        fig1_tache_A_6_instants         fig1_tache_B_6_instants
        fig2_quatre_cas                 etat final et trace, les 4 reunis

    La comparaison simultanee des deux schemas, elle, est le role de la video
    A_contre_B.mp4, pas celui des figures.
    """
    for cas in ("injection", "tache"):
        for version in ("A", "B"):
            figure_six_instants(cas, version, nom_laby)
    figure_quatre_cas(nom_laby)


# =====================================================================
#  4. LES DEUX VIDEOS
# =====================================================================

def _ffmpeg():
    """
    Trouve ffmpeg : d'abord dans le PATH, puis dans le paquet imageio-ffmpeg.

    Retourne le chemin de l'executable, ou None.
    """
    trouve = shutil.which("ffmpeg")
    if trouve:
        return trouve
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _assemble(dossier_images, sortie, images_par_seconde=24):
    """
    Assemble une suite de PNG numerotes en un MP4.

    Si ffmpeg est introuvable, on GARDE les images et on affiche la commande
    a lancer. Mieux vaut un resultat incomplet et un message clair qu'une
    trace d'erreur illisible.
    """
    exe = _ffmpeg()
    motif = os.path.join(dossier_images, "f%04d.png")
    chemin = os.path.join(DOSSIER_RESULTATS, sortie)

    if exe is None:
        print()
        print("ffmpeg introuvable : les images sont gardees dans")
        print("    %s" % dossier_images)
        print("Installer le paquet qui le contient :")
        print("    pip install imageio-ffmpeg")
        print("ou assembler a la main :")
        print('    ffmpeg -framerate %d -i "%s" -c:v libx264 -pix_fmt yuv420p "%s"'
              % (images_par_seconde, motif, chemin))
        return None

    subprocess.run([exe, "-y", "-loglevel", "error",
                    "-framerate", str(images_par_seconde), "-i", motif,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", chemin], check=True)
    print("ecrit : resultats/%-22s %6.0f Ko"
          % (sortie, os.path.getsize(chemin) / 1024))
    return chemin


def videos(nom_laby="laby14", n_images=240, images_par_seconde=24,
           verifie=True):
    """
    Refait tourner l'injection continue dans les DEUX versions en meme temps,
    et ecrit deux videos :

        injection_B.mp4    la version B seule, en grand
        A_contre_B.mp4     les deux cote a cote, chacune titree

    Les deux schemas avancent dans la MEME boucle, avec le meme dt et le meme
    nombre de pas. La comparaison est donc faite a conditions rigoureusement
    identiques : a chaque image, les deux moities montrent le meme instant.

    verifie : aux pas enregistres dans les .npz, compare l'etat courant a
        celui du disque et affiche l'ecart. Il doit valoir exactement zero.
    """
    domain, u_bord, v_bord, dt, infos = S.prepare(nom_laby)
    n_pas = S.pas_necessaires(domain, u_bord, v_bord, dt, 1.5)
    conditions, c_impose = S.fabrique_conditions(domain, 100.0)

    # les images a rendre : n_images instants repartis regulierement
    a_rendre = {round(n_pas * k / n_images) for k in range(1, n_images + 1)}

    # les pas de controle, lus dans les .npz deja sur le disque
    reference = {}
    if verifie:
        try:
            reference = {v: _charge("injection_%s" % v) for v in ("A", "B")}
            controle = set(_instants(reference["B"]))
        except FileNotFoundError:
            controle = set()
    else:
        controle = set()

    dossier = os.path.join(DOSSIER_RESULTATS, "_images_temporaires")
    shutil.rmtree(dossier, ignore_errors=True)
    os.makedirs(os.path.join(dossier, "b"))
    os.makedirs(os.path.join(dossier, "ab"))

    print("Video : %s, %d pas, dt = %.4f, %d images"
          % (nom_laby, n_pas, dt, n_images))
    print("Les deux versions avancent dans la meme boucle.")
    print()

    ca = conditions(np.zeros_like(domain, dtype=float))
    cb = ca.copy()
    numero = 0

    pas_a_faire = range(1, n_pas + 1)
    if tqdm is not None:
        pas_a_faire = tqdm(pas_a_faire, desc="%-14s" % "A et B",
                           unit=" pas", ncols=92)

    for n in pas_a_faire:
        # --- l'etiquette est sur la meme ligne que le calcul ---------------
        ca = conditions(A.pas_euler(ca, domain, u_bord, v_bord, dt,
                                    lineaire=False))                # version A
        cb = conditions(A.pas_rk2(cb, domain, u_bord, v_bord, dt,
                                  lineaire=True, conditions=conditions))  # B

        if n in controle:
            # les .npz sont enregistres en float32 : on compare dans ce type
            ecart_a = np.abs(ca.astype(np.float32)
                             - reference["A"]["pas_%d" % n]).max()
            ecart_b = np.abs(cb.astype(np.float32)
                             - reference["B"]["pas_%d" % n]).max()
            ligne = ("pas %6d : ecart max au .npz   A = %.3e   B = %.3e"
                     % (n, ecart_a, ecart_b))
            (tqdm.write if tqdm is not None else print)(ligne)

        if n in a_rendre:
            numero += 1
            panneau_a = _en_octets(peint(ca, domain))
            panneau_b = _en_octets(peint(cb, domain))
            bas = "pas %d / %d          t = %.0f" % (n, n_pas, n * dt)

            seule = _habille([_agrandi(panneau_b, 3)],
                             [("VERSION B", "reconstruction lineaire + Runge-Kutta 2"
                                            "   -   ordre 2")],
                             bas)
            seule.save(os.path.join(dossier, "b", "f%04d.png" % numero))

            cote = _agrandi(panneau_a, 2).shape[1] + 6
            duo = _habille([_agrandi(panneau_a, 2), _agrandi(panneau_b, 2)],
                           [("VERSION A", "constante + Euler   -   ordre 1"),
                            ("VERSION B", "lineaire + RK2   -   ordre 2")],
                           bas, largeur_titre=cote)
            duo.save(os.path.join(dossier, "ab", "f%04d.png" % numero))

    if tqdm is not None:
        pas_a_faire.close()
    print("%d images rendues" % numero)
    print()

    faits = [_assemble(os.path.join(dossier, "b"), "injection_B.mp4",
                       images_par_seconde),
             _assemble(os.path.join(dossier, "ab"), "A_contre_B.mp4",
                       images_par_seconde)]

    if all(faits):
        shutil.rmtree(dossier, ignore_errors=True)
    return faits


# =====================================================================
#  5. TOUT
# =====================================================================

def tout(nom_laby="laby14"):
    """Les deux figures puis les deux videos."""
    figures(nom_laby)
    print()
    videos(nom_laby)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        figures()                 # les 2 .png seulement, quelques secondes
    elif len(sys.argv) > 1 and sys.argv[1] == "videos":
        videos()                  # les 2 .mp4 seulement, environ 2 minutes
    else:
        tout()
