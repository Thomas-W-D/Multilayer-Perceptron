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
y = fonctions.generation(x, erreur=0., fonction=np.sin) # objectif

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


# calcul de l'erreur moyenne par epoch
erreur_test = np.array(erreur_test)
erreur_test = np.reshape(erreur_test, (nb_epoch, int(len(erreur_test)/nb_epoch)))
erreur_test = np.sum(erreur_test, axis=1)/nb_epoch
erreur_ent = np.array(erreur_ent)
erreur_ent = np.reshape(erreur_ent, (nb_epoch, int(len(erreur_ent)/nb_epoch)))
erreur_ent = np.sum(erreur_ent, axis=1)/nb_epoch


# PLOT
# calcul du résultat attendu
x = np.reshape(x, (x.size,1))
y = np.reshape(y, (y.size,1)) # résultat attendu
y_pred = net.pass_forward(x) # résultat du réseau

# Affichage des résultats
plt.subplot(2,1,1)
plt.plot(x,y,"-", label="objectif")
plt.plot(x,y_pred,"-", label="prédiction")
#plt.plot(lot_ent, np.sin(lot_ent), "r.")
#plt.plot(test, np.sin(test), "b.")
plt.legend()
plt.title("Données calculées par le MLP")

plt.subplot(2,1,2)
plt.plot(erreur_test, "b", label="erreur des données de test")
plt.plot(erreur_ent, "r", label="erreur des données d'entraînement")
plt.legend()
plt.xlabel("Évolution de l'erreur au cours de l'entraînement")

plt.show()

print(lot_ent_x)