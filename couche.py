"""
Module pour la classe de couche de neurones.
"""

import numpy as np


class Couche:
    """
    Classe de couches de neurones
    """
    def __init__(self, n_neurones_coucheprec, n_neurones, f_activ, df_activ):
        self.n_neu = n_neurones # nb de neurones de la couche
        self.n_neup = n_neurones_coucheprec # nb de neurones de la couche précédente
        self.f = f_activ # fonction d'activation
        self.df = df_activ # derivée de la fonction d'activation

        # génération aléatoire de la matrice de poids (biais inclus)
        self.W = np.random.uniform(-1,1, (self.n_neu,self.n_neup+1))
        self.z = None # somme pondérée des sorties précédentes par les poids
        self.u = None # sorties de la couche actuelle
        self.u_n_1 = None # sorties de la couche précédente

        self.dL_dz = None # gradient par rapport à z
        self.dL_dW = np.zeros(self.W.shape) # gradient par rapport à W

        # paramètres pour l'optimizer
        self.m_t_1 = np.zeros(self.W.shape) # mémoire pour m_t
        self.v_t_1 = np.zeros(self.W.shape) # mémoire pour v_t
        # paramètres biaisés
        self.m_t = None
        self.v_t = None
        # paramètres corrigés
        self.mt = None
        self.vt = None


    def pass_forward(self, Yp):
        """
        Prend les sorties de la couche précédente, effectue leur somme pondérée
        par les poids (pour chaque neurone de la couche) et applique la
        fonction d'activation pour renvoyer les résultats.
        """
        self.z = np.matmul(
            self.W,
            np.concatenate([Yp.T, np.array([[1]*Yp.shape[0]])], axis=0)
        ).T
        self.u = self.f(self.z)
        return self.u


    def bp_init(self, valeur_att, N_batch, u_n_1):
        """
        Calcule les gradients pour la dernière couche :
        - calcule les gradients par rapport à z
        - en déduit les gradients par rapports à W
        """
        self.u_n_1 = np.concatenate([u_n_1,np.array([[1]]*N_batch)], axis=1)
        self.dL_dz = (2/N_batch) * (self.u - valeur_att) * self.df(self.z)
        self.dL_dW = np.tensordot(self.dL_dz, self.u_n_1, axes=([0],[0]))


    def back_propagation(self, N_batch, u_n_1, dL_dZ_n1, W_n1):
        """
        Calcule les gradients pour les couches (sauf la dernière) :
        - calcule les gradients par rapport à z
        - en déduit les gradients par rapport à W
        """
        self.u_n_1 = np.concatenate([u_n_1,np.array([[1]]*N_batch)], axis=1)
        self.dL_dz = self.df(self.z) * np.matmul(dL_dZ_n1, W_n1[:,:-1])
        # on ne prend pas en compte les biais de la couche suivante car ils ne
        # dépendent pas de la couche actuelle
        self.dL_dW = np.tensordot(self.dL_dz, self.u_n_1, axes=([0],[0]))


    def update_weights(self, type, learning_rate=0.001, learn_number=1, B1=0.9,
                       B2=0.999, e=1e-8):
        """
        Modifie les poids à partir des gradients de la loss function.
        - 'type' indique le type d'optimizer utilisé.
        - 'learning_rate' indique le taux d'apprentissage.
        - 'learn_number' indique le numéro d'apprentissage.
        - 'B1' et 'B2' sont liés aux paramètres calculés (taux d''oubli').
        - 'e' est utilisé pour éviter une possible division par zéro.

        On a différents optimizer possibles :
        - 'type' = 0 -> Implémentation de la descente de gradient classique,
        Batch Gradient Descent (BGD).
        - 'type' = 1 -> Implémentation de l'optimize Adaptative Moment
        Estimation (Adam).
        """
        if type == 0:
            # BATCH GRADIENT DESCENT
            self.W -= learning_rate * self.dL_dW

        if type == 1:
            # ADAM OPTIMIZER
            # paramètres biaisés
            self.m_t = B1*self.m_t_1 + (1-B1)*self.dL_dW
            self.v_t = B2*self.v_t_1 + (1-B2)*self.dL_dW**2
            # correction des paramètres (suppression des biais)
            self.mt = self.m_t/(1-B1**learn_number)
            self.vt = self.v_t/(1-B2**learn_number)
            # modification des poids
            self.W -= learning_rate * self.mt/(np.sqrt(self.vt) + e)
            # changement des paramètres en mémoire
            self.m_t_1 = self.m_t
            self.v_t_1 = self.v_t