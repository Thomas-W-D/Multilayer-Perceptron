"""
Module pour la classe de réseau.
"""

import numpy as np


class Reseau:
    """
    Classe de réseau qui contient plusieurs couches de neurones.

    Attend qu'on lui donne des objets de type 'Couche' du module 'couche' en
    entrée (sauf la première). Les fonctions de la classe ne fonctionneront pas
    si on lui donne des objets différents.

    'opt_type' définit l'optimizer utilisé (par défaut BGD) :
    - 0 pour Batch Gradient Descent (BGD).
    - 1 pour Adaptative Moment Estimation (Adam).
    """
    def __init__(self, opt_type, *couches):
        # regroupe les couches de neurones sous forme de liste
        self.couches = list(couches)
        self.nb_couches = len(self.couches) # nombre de couches
        self.N_batch = 1 # taille de batch
        self.lr = 1 # learning rate

        self.opt_type = opt_type # type d'optimizer
        self.learn_number = 0 # indique le nombre de fois qu'on a appris

        self.entries = None # entrées du réseau
        self.output = None # sortie du réseau
        self.loss_value = None # résultat de la loss function

        if type(self.opt_type) != int:
            raise ValueError("Préciser l'optimizer à utiliser.")


    def pass_forward(self, entries):
        """
        Fonction de pass forward qui prend les entrées du
        réseau et renvoie les prédictions de celui-ci.
        - 'self.output' correspond finalement à la prédiction du réseau.
        - Fonctionne avec plusieurs données a la fois :
        'entries' doit être de la forme [[donnees1], [donnees2], ...]
        (liste ou tableau numpy)
        """
        self.entries = np.array(entries) # sauvegarde les entrées
        self.N_batch = self.entries.shape[0] # actualise la taille de batch
        # initialise les données qui passent dans le réseau
        self.output = self.entries
        # fait "avancer" les données dans le réseau
        for couche in self.couches:
            self.output = couche.pass_forward(self.output)
        return self.output


    def loss(self, valeur_att):
        """
        Renvoie le résultat de la loss function (ici MSE) calculée sur le batch
        """
        valeur_att = np.array(valeur_att)
        self.loss_value = (1/self.N_batch)*np.sum((self.output-valeur_att)**2)
        return self.loss_value


    def back_propagation(self, valeur_att):
        """Effectue la back propagation dans le réseau : modifie les poids"""
        self.learn_number += 1 # on apprend une fois de plus

        # Calcul des gradients dans les différentes couches
        if self.nb_couches > 1:
            # calcul des gradients et modification des poids (dernière couche)
            self.couches[-1].bp_init(
                valeur_att, 
                self.N_batch, 
                self.couches[self.nb_couches -2].u
            )
            # calcul des gradients (hidden layers en partant de la fin)
            for i in range(self.nb_couches -2):
                self.couches[self.nb_couches -i -2].back_propagation(
                    self.N_batch,
                    self.couches[self.nb_couches -i -2 -1].u, 
                    self.couches[self.nb_couches -i -2 +1].dL_dz, 
                    self.couches[self.nb_couches -i -2 +1].W
                )
            # calcul des gradients (première couche)
            self.couches[0].back_propagation(
                self.N_batch,
                self.entries, 
                self.couches[1].dL_dz, 
                self.couches[1].W
            )
        else:
            self.couches[-1].bp_init(
                valeur_att, 
                self.N_batch, 
                self.entries
            )

        # Modification des poids
        for couche in self.couches:
            couche.update_weights(self.opt_type, self.lr, self.learn_number)