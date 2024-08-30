import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

import couche
import reseau
import fonctions


# paramètres d'entraînement
nb_epoch = 200
taille_de_batch = 100
prop_test = 0.3 # proportion de données d'entraînement dans les données complètes
erreur_ent = []
erreur_test = []
f_acti = fonctions.tanh
df_acti = fonctions.dtanh


# génération des données
nb_donnees = 1000
x = np.linspace(-10,10,nb_donnees) # ensemble des données
y = np.sin(x) # objectif

# séparation des lots d'entraînement et de test
lot_ent, test = fonctions.lots_donnees_aleat(prop_test, taille_de_batch, x, y)
lot_ent_x = lot_ent[:,:,0:1]
lot_ent_sinx = lot_ent[:,:,1:2]
test_x = test[:,0:1]
test_sinx = test[:,1:2]

# définition du réseau
net = reseau.Reseau(
    1,
    couche.Couche(1,20,f_acti, df_acti),
    couche.Couche(20,10,f_acti, df_acti),
    couche.Couche(10,1,fonctions.Id, fonctions.dId)
)
net.lr = 0.01


x = np.reshape(x, (x.size,1))
evolution = []


# Entraînement
for epoch in range(nb_epoch): # EPOCH
    evolution.append(net.pass_forward(x))
    for batch in range(lot_ent.shape[0]): # BATCH
        # test
        net.pass_forward(test_x)
        erreur_test.append(net.loss(test_sinx))
        # entraînement
        net.pass_forward(lot_ent_x[batch])
        erreur_ent.append(net.loss(lot_ent_sinx[batch]))
        net.back_propagation(lot_ent_sinx[batch])
evolution.append(net.pass_forward(x))


# calcul de l'erreur moyenne par epoch
erreur_test = np.array(erreur_test)
erreur_test = np.reshape(erreur_test, (nb_epoch, int(len(erreur_test)/nb_epoch)))
erreur_test = np.sum(erreur_test, axis=1)/nb_epoch
erreur_ent = np.array(erreur_ent)
erreur_ent = np.reshape(erreur_ent, (nb_epoch, int(len(erreur_ent)/nb_epoch)))
erreur_ent = np.sum(erreur_ent, axis=1)/nb_epoch



fig, ax = plt.subplots()
ax.set_xlim(-12,12)
ax.set_ylim(-1,1)

line, = ax.plot([], [])

def init():
    line.set_data([], [])
    return line,

def animate(i):
    return line.set_data(x, evolution[i])

anim = ani.FuncAnimation(fig, animate, init_func=init, frames=len(evolution), interval=50)
plt.show()