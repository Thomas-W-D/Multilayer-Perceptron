"""
Module pour les fonctions importantes :
- fonctions d'activation des neurones.
- fonctions de génération et traitement des données.
"""

import numpy as np
import scipy.stats as sc


### Fonctions d'activation ####################################################
# RELU
def ReLU(x):
    """Fonction ReLU"""
    return np.maximum(0,x)
def dReLU(x):
    """Derivée de la fonction ReLU (=0 si x<0 et =1 si x>0)"""
    return (x > 0) * np.ones(x.shape)

# IDENTITÉ
def Id(x):
    """Renvoie son entrée"""
    return x
def dId(x):
    """Dérivée de x par rapport à x, soit 1"""
    return np.ones(x.shape)

# TANH
def tanh(x):
    """Fonction tangente hyperbolique"""
    return np.tanh(x)
def dtanh(x):
    """Dérivée de la tangente hyperbolique"""
    return 1-np.tanh(x)**2
###############################################################################


### Fonctions de génération et traitement des données #########################
def generation(x, erreur=0, fonction=np.sin):
    """
    Génère des données suivant la fonction 'fonction' avec le bruit 'erreur'
    """
    Y = fonction(x)
    y = np.empty(x.shape)
    for i in range(y.size):
        y[i] = sc.norm.rvs()*erreur + Y[i]
    return y


def lots_donnees_reg(prop_test, taille_de_batch, *donnees):
    """
    Renvoie un tableau de taille (nb_batch, taille_de_batch, nb_entrees).

    Sépare les données de test et d'entraînement en fonction de :
    - 'prop_test' qui indique la proportion (minimale) de données de test dans
    les données totales.
    - 'taille_de_batch' qui indique la taille du batch.
    - '*donnees' qui est l'ensemble des données.

    La fonction attend de recevoir un tableau de données par entrée du réseau.

    Classe les données d'entraînement régulièrement par batch.
    Exemple : données = [1,2,3,4,5,6] -> [[1,3,5], [2,4,6]] (taille de batch 3)
    """
    donnees = list(donnees) # liste des tableaux d'entrées
    nb_entrees = len(donnees) # nombre de neurones de la première couche
    # mise en forme des entrées
    for i in range(len(donnees)):
        donnees[i] = np.reshape(donnees[i], (donnees[0].size,1))
    donnees = np.concatenate(donnees, axis=1) # un tableau pour les données

    # tirage aléatoire des données de test et d'entraînement
    # On prend une proportion prop_test de données de test dans les
    # données, et on y ajoute les données en trop dans ce qui reste.
    test_indices = np.random.choice(
        donnees.shape[0],
        size=(int(prop_test*donnees.shape[0]+((1-prop_test)*donnees.shape[0])%taille_de_batch)),
        replace=False
    )
    lot_ent_indices = np.setdiff1d(np.arange(donnees.shape[0]), test_indices)
    # séparation des données de test et d'entraînement
    test = donnees[test_indices]
    lot_ent = donnees[lot_ent_indices]

    if lot_ent.size == 0: # on vérifie qu'on ne renverra pas de tableau vide
        raise ValueError(
            "Pas assez de données envoyées par rapport à la taille de batch."
        )

    # mise en forme des données d'entraînement
    nb_batch = int(lot_ent.shape[0]/taille_de_batch)
    lot_ent = np.reshape(lot_ent.T, (nb_entrees, taille_de_batch, nb_batch)).T

    return lot_ent, test


def lots_donnees_aleat(prop_test, taille_de_batch, *donnees):
    """
    Renvoie un tableau de taille (nb_batch, taille_de_batch, nb_entrees).

    Sépare les données de test et d'entraînement en fonction de :
    - 'prop_test' qui indique la proportion (minimale) de données de test dans
    les données totales.
    - 'taille_de_batch' qui indique la taille du batch.
    - '*donnees' qui est l'ensemble des données.

    La fonction attend de recevoir un tableau de données par entrée du réseau.

    Classe les données d'entraînement aléatoirement.
    """
    donnees = list(donnees) # liste des tableaux d'entrées
    nb_entrees = len(donnees) # nombre de neurones de la première couche
    # mise en forme des entrées
    for i in range(len(donnees)):
        donnees[i] = np.reshape(donnees[i], (donnees[0].size,1))
    donnees = np.concatenate(donnees, axis=1) # un tableau pour les données

    # tirage aléatoire des données de test et d'entraînement
    # On prend une proportion prop_test de données de test dans les
    # données, et on y ajoute les données en trop dans ce qui reste.
    np.random.shuffle(donnees) # on classe les données aléatoirement
    taille_test = int(prop_test*donnees.shape[0]+((1-prop_test)*donnees.shape[0])%taille_de_batch)
    test = donnees[:taille_test] # les premières valeurs correspondent au test
    lot_ent = donnees[taille_test:] # le reste, aux données d'entraînement

    if lot_ent.size == 0: # on vérifie qu'on ne renverra pas de tableau vide
        raise ValueError(
            "Pas assez de données envoyées par rapport à la taille de batch."
        )

    # mise en forme des données d'entraînement
    nb_batch = int(lot_ent.shape[0]/taille_de_batch)
    lot_ent = np.reshape(lot_ent, (nb_batch, taille_de_batch, nb_entrees))

    return lot_ent, test
###############################################################################