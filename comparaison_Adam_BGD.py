import numpy as np
import matplotlib.pyplot as plt

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


# Entraînement
for epoch in range(nb_epoch): # EPOCH
    for batch in range(lot_ent.shape[0]): # BATCH
        # test
        net.pass_forward(test_x)
        erreur_test.append(net.loss(test_sinx))
        # entraînement
        net.pass_forward(lot_ent_x[batch])
        erreur_ent.append(net.loss(lot_ent_sinx[batch]))
        net.back_propagation(lot_ent_sinx[batch])


# PLOT
# calcul du résultat attendu
x = np.reshape(x, (x.size,1))
y = np.reshape(y, (y.size,1)) # résultat attendu
plt.plot(x,y,"r-", label="objectif")


y_pred = net.pass_forward(x) # résultat du réseau

# Affichage des résultats
plt.plot(x,y_pred,"-", label="Adam")



# définition du réseau
net = reseau.Reseau(
    0,
    couche.Couche(1,20,f_acti, df_acti),
    couche.Couche(20,10,f_acti, df_acti),
    couche.Couche(10,1,fonctions.Id, fonctions.dId)
)
net.lr = 0.001


# Entraînement
for epoch in range(nb_epoch): # EPOCH
    for batch in range(lot_ent.shape[0]): # BATCH
        # test
        net.pass_forward(test_x)
        erreur_test.append(net.loss(test_sinx))
        # entraînement
        net.pass_forward(lot_ent_x[batch])
        erreur_ent.append(net.loss(lot_ent_sinx[batch]))
        net.back_propagation(lot_ent_sinx[batch])


# PLOT
# calcul du résultat attendu
x = np.reshape(x, (x.size,1))
y = np.reshape(y, (y.size,1)) # résultat attendu
y_pred = net.pass_forward(x) # résultat du réseau

# Affichage des résultats
plt.plot(x,y_pred,"-", label="SGD")



plt.legend()
plt.title("Données calculées par le MLP")
plt.show()