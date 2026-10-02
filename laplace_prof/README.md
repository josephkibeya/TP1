# Labyrinthe

Résolution d'un labyrinthe sur base d'un écoulement irrotationnel (différences finies).

# Donnée de base

Le labyrinthe est fourni sous la forme d'une image RGB, exploitation de la composante R. Les murs sont associés au noir et la zone de rechreche au blanc.
Si un bord blanc existe sur le pourtour de l'image, il est supprimé par recherche des éléments noirs externes.

# Conditions aux limites

L'entrée du labyrinthe doit être située sur la ligne supérieure et la sortie sur la ligne inférieure.

L'entrée et la sortie sont soumises à condition forte. Le différentiel de potentiel génèrera l'écoulement.

# Convention

La convention de position est **matricielle**. L'élément [0,0] est situé en haut à gauche. Le voisin de gauche de l'élément [i,j] est [i,j-1], de droite [i,j+1], du haut [i-1,j], du bas [i+1,j].
# Solution

La routine écrit sur disque :
  - la matrice de domaine finale (éventuellement réduite en taille par rapport à l'image de départ) au format .txt et image
  - la solution du potentiel de vitesse (même taille que la domaine) au format .txt, .npy (numpy binaire) et image

# Exemple

Un exemple est fourni via le fichier image "laby.png".
L'exécution du script "laplace_only.py" fournira les résultats.

# Utilisation comme module

Il est possible d'importer le module "laplace_only" dans un autre script via "import laplace_only".

On peut ensuite appeler la fonction "compute" sur base d'un chemin d'accès vers un fichier image et un nom de sortie (optionnel - défaut = out).

Les extensions "_pot.txt", "_pot.npy", "_pot.png", "_domain.txt" et "_domain.png" seront automatiquement ajoutées au nom de sortie fourni. Il ne doit donc pas contenir d'extension.

# Dépendances

Ce module utilise les paquets :
  - numpy
  - pillow
  - matplotlib
  - tqdm
  - scipy

Il est possible de les installer via "pip install -r requirements.txt"



Enjoy!
