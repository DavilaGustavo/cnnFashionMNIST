import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from functions import plot_confusion_matrix, show_misclassified_examples

# Carregar os dados
fashion_mnist = tf.keras.datasets.fashion_mnist
(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test / 255.0

# Os dados são apenas 2D, mas a convolução espera altura x largura x cor
x_test = np.expand_dims(x_test, -1)

# Carregar o modelo salvo
with open('modelTraining/model.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights('modelTraining/model.weights.h5')
# print("Modelo carregado do disco")

# Previsões e matriz de confusão
p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)

# Plotar a matriz de confusão
plot_confusion_matrix(cm)

# Mostrar um exemplo de classificação errado
show_misclassified_examples(x_test, y_test, p_test)
