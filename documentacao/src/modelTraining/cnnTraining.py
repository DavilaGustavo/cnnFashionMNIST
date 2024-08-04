import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Carregar os dados
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# print("x_train.shape: ", x_train.shape)

# Expandir as dimensões para a convolução
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# print("x_train.shape nova forma: ", x_train.shape)

# Data Augmentation
dataAug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
dataAug.fit(x_train)

# Número de classes
K = len(set(y_train))
# print("número de classes: ", K)

# Construir o modelo usando a API funcional
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(i)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

# Compilar e ajustar
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks para Early Stopping e Redução do Learning Rate
pararAntes = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduzirTA = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Treinar o modelo
model.fit(dataAug.flow(x_train, y_train, batch_size=128), 
          validation_data=(x_test, y_test), 
          epochs=100, 
          callbacks=[pararAntes, reduzirTA])

# Salvar o modelo treinado
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.weights.h5')