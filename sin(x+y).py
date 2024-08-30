import numpy as np
import matplotlib.pyplot as plt

import couche
import reseau
import fonctions


# paramètres d'entraînement
nb_epoch = 300
taille_de_batch = 1
prop_test = 0 # proportion de données d'entraînement dans les données complètes
erreur_ent = []
erreur_test = []
f_acti = fonctions.tanh
df_acti = fonctions.dtanh


# génération des données
nb_donnees = 100 # **2
x = np.linspace(-10,10,nb_donnees) # ensemble des données
y = np.linspace(-10,10,nb_donnees)
x1, y1 = np.meshgrid(x, y)
x = np.reshape(x1, (x1.size,))
y = np.reshape(y1, (y1.size,))



lot_ent, test = fonctions.lots_donnees_aleat(prop_test, taille_de_batch, x, y, np.sin(x+y))
lot_ent_xy = lot_ent[:,:,0:2]
lot_ent_sinx = lot_ent[:,:,2:3]
test_xy = test[:,0:2]
test_sinx = test[:,2:3]

# définition du réseau
net = reseau.Reseau(
    1,
    couche.Couche(2,20,f_acti, df_acti),
    couche.Couche(20,10,f_acti, df_acti),
    couche.Couche(10,1,fonctions.Id, fonctions.dId)
)
net.lr = 0.01


# Entraînement
for epoch in range(nb_epoch): # EPOCH
    for batch in range(lot_ent.shape[0]): # BATCH
        # entraînement
        net.pass_forward(lot_ent_xy[batch])
        print(lot_ent_xy[batch])
        erreur_ent.append(net.loss(lot_ent_sinx[batch]))
        net.back_propagation(np.sin(lot_ent_sinx[batch]))



z_pred = net.pass_forward(np.concatenate([x1.flatten().reshape((x1.size,1)),y1.flatten().reshape((y1.size,1))], axis=1)) # résultat du réseau

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    x1.flatten().reshape((x1.size,1)),y1.flatten().reshape((y1.size,1)),
    z_pred,
    marker="."
)

# Étiqueter les axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('sin(X+Y)')

# Afficher le graphique
plt.show()