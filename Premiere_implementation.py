import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Exemple de paramètres (à adapter en fonction de tes données)
timesteps = 10        # nombre de pas de temps par séquence
features = 4          # par exemple, commande + sortie pour 4 moteurs ou plus selon tes mesures
num_classes = 9

# Construction du modèle
model = Sequential()
# Première couche LSTM qui retourne la séquence complète
model.add(LSTM(100, input_shape=(timesteps, features), return_sequences=True))
# Dropout pour la régularisation
model.add(Dropout(0.1))
# Deuxième couche LSTM qui ne retourne que le dernier état caché
model.add(LSTM(100))
# Couche dense finale avec 9 neurones et activation softmax
model.add(Dense(num_classes, activation='softmax'))

# Compilation du modèle avec l'optimiseur Adam et la perte categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Affichage du résumé du modèle
model.summary()
