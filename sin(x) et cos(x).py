import numpy as np
import matplotlib.pyplot as plt

import couche
import reseau
import fonctions


# paramètres d'entraînement
nb_epoch = 500
taille_de_batch = 100
prop_test = 0 # proportion de données d'entraînement dans les données complètes
erreur_ent = []
erreur_test = []
f_acti = fonctions.tanh
df_acti = fonctions.dtanh


# génération des données
nb_donnees = 1000
x = np.linspace(-10,10,nb_donnees) # ensemble des données

# séparation des lots d'entraînement et de test
lot_ent, test = fonctions.lots_donnees_aleat(prop_test, taille_de_batch, x, np.sin(x), np.cos(x))
lot_ent_x = lot_ent[:,:,0:1]
lot_ent_sinx_cosx = lot_ent[:,:,1:3]
test_x = test[:,0:1]
test_sinx_cosx = test[:,1:3]

# définition du réseau
net = reseau.Reseau(
    1,
    couche.Couche(1,20,f_acti, df_acti),
    couche.Couche(20,10,f_acti, df_acti),
    couche.Couche(10,2,fonctions.Id, fonctions.dId)
)
net.lr = 0.01


# Entraînement
for epoch in range(nb_epoch): # EPOCH
    for batch in range(lot_ent.shape[0]): # BATCH
        # entraînement
        net.pass_forward(lot_ent_x[batch])
        erreur_ent.append(net.loss(lot_ent_sinx_cosx[batch]))
        net.back_propagation(lot_ent_sinx_cosx[batch])


# calcul de l'erreur moyenne par epoch
erreur_ent = np.array(erreur_ent)
erreur_ent = np.reshape(erreur_ent, (nb_epoch, int(len(erreur_ent)/nb_epoch)))
erreur_ent = np.sum(erreur_ent, axis=1)/nb_epoch


# PLOT
# calcul du résultat attendu
x = np.reshape(x, (x.size,1))
y_pred = net.pass_forward(x) # résultat du réseau
print(y_pred[:,0])
# Affichage des résultats
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    x.flatten(), y_pred[:,0], y_pred[:,1],
    marker="."
)
ax.set_xlabel("x")
ax.set_ylabel("sin(x)")
ax.set_zlabel("cos(x)")
plt.show()