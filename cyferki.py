import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def create_experiment_folder(base_folder):
    # Sprawdź, ile istniejących eksperymentów już istnieje w folderze
    existing_experiments = [name for name in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, name))]

    # Określ numer nowego eksperymentu
    new_experiment_num = len(existing_experiments)

    # Utwórz folder dla nowego eksperymentu
    new_experiment_folder = os.path.join(base_folder, f'exp{new_experiment_num}')
    os.makedirs(new_experiment_folder, exist_ok=True)

    return new_experiment_folder


# Sprawdzanie dostępności danych MNIST - jak nie to pobierz
try:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
except:
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizacja danych do zakresu [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definicja modelu
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Zmienić epoki do trenowania modelu
model.fit(train_images, train_labels, epochs=150)

# Ocena modelu na danych testowych
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nDokładność na danych testowych: {test_acc}")

# Przewidywanie etykiet dla danych testowych
predictions = model.predict(test_images)
predicted_labels = tf.argmax(predictions, axis=1)

# MACIEŻ POMYŁEK
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Normalizacja macierzy pomyłek
normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 1000

# Utwórz folder dla nowego eksperymentu
base_folder = r'C:\Users\mateu\Desktop\wykrywanie_rzeczy\macierz_pomyłek_cyfry'
output_folder = create_experiment_folder(base_folder)

# Określenie lokalizacji pliku do zapisu obrazu
output_image_path = os.path.join(output_folder, 'macierz_pomyłek_cyfry.png')

# Wyświetlanie macierzy pomyłek
plt.figure(figsize=(10, 8))
sns.heatmap(normalized_conf_matrix, annot=True, fmt='.0f', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Przewidywane etykiety')
plt.ylabel('Rzeczywiste etykiety')
plt.title('Znormalizowana macierz pomyłek (maksymalna wartość 1000)')

# Zapisywanie macierzy pomyłek jako obraz
plt.savefig(output_image_path)

# Wyświetlanie pierwszego obrazu treningowego
first_train_image = train_images[523]

plt.figure()
plt.imshow(first_train_image, cmap='gray')
plt.title('Obraz treningowy')
plt.show()
