# Feuille de route — Projet Volumes Finis (Labyrinthe)

Cours : GCVIV0185-7, fin du cours = décembre. On est début octobre : il y a
le temps, à condition d'avancer régulièrement (petites étapes, testées à
chaque fois).

## Où on en est

- [x] Python installé, dépôt du prof récupéré et déjà exécuté avec succès
      (`laplace_only.py` → `out_pot.npy`, `out_domain.txt`, images)
- [x] Repo GitHub créé sur github.com, prof invité en collaborateur
- [ ] Repo GitHub pas encore relié à ce PC (à faire : `git init` +
      remote + premier push, une fois qu'on a l'URL)
- [ ] Étape 1 : vitesses aux bords + vérification divergence nulle — **en cours**
- [ ] Étape 2 : reconstruction constante + flux upwind + Euler explicite,
      testés sur un canal rectiligne (vitesse uniforme → la solution exacte
      est juste une translation de la donnée initiale)
- [ ] Étape 3 : condition CFL / stabilité (nombre de Courant < 1)
- [ ] Étape 4 : reconstruction linéaire (gradients par maille)
- [ ] Étape 5 : schéma temporel Runge-Kutta à 2 pas
- [ ] Étape 6 : application au labyrinthe complet + conditions limites
      "faibles" aux entrée/sortie
- [ ] Rapport : lier chaque résultat à la théorie (pas de rappel
      théorique brut)
- [ ] Notebook Jupyter qui exploite le code
- [ ] Démo privée du code le 5 novembre

## Pourquoi cet ordre

On code d'abord des morceaux SIMPLES et VÉRIFIABLES avant de les combiner :
une fonction fausse dans un système complexe est presque impossible à
retrouver ; une fonction fausse testée seule, sur un cas simple, se repère
en 2 minutes. C'est pour ça qu'on commence par la vérification de
divergence nulle, puis un canal rectiligne (où on CONNAÎT la réponse
exacte) avant de toucher au vrai labyrinthe.

## Rappel Python accumulé au fil des étapes

(ajouté au fur et à mesure — relis cette liste à chaque fois que t'as un
doute, ça va s'étoffer)

- **numpy.ndarray** : un tableau de nombres à N dimensions. `pot.shape`
  donne ses dimensions, ex. `(502, 502)` = 502 lignes x 502 colonnes.
- **slicing** `tableau[a:b]` : sous-partie du tableau entre l'indice a
  (inclus) et b (exclu). `[1:]` = à partir du 2e élément. `[:-1]` = tout
  sauf le dernier.
- **vectorisation** : faire une opération sur TOUT un tableau d'un coup
  (`a - b`) plutôt qu'avec une boucle `for` élément par élément — plus
  rapide et plus lisible.
- **def ma_fonction(params):** définit une fonction. `return` renvoie le(s)
  résultat(s). `a, b = ma_fonction(...)` récupère plusieurs valeurs
  renvoyées d'un coup (on dit qu'on "déballe" un tuple).
- **masque booléen** : `tableau[condition]` où `condition` est un tableau
  de True/False de même taille — sélectionne uniquement les éléments où
  c'est True. Très utilisé pour "ne garder que certaines mailles".
- **`if __name__ == "__main__":`** : le code dans ce bloc ne s'exécute que
  si on lance CE fichier directement (`python vitesses.py`), pas si on
  l'importe depuis un autre script (`import vitesses`).
