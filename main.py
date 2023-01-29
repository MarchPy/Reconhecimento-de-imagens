import os
import tensorflow as tf
import numpy as np
import sys


class IA:
    batch_size = 32
    img_height = 180
    img_width = 180
    data_dir = "Imagens"

    def __init__(self):
        # Dados de treinamento
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

        # Dados de validação
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

        self.class_names = self.train_ds.class_names
        self.num_classes = len(self.class_names)
        
        # Otimização
        autotune = tf.data.AUTOTUNE
        train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
        val_ds = self.val_ds.cache().prefetch(buffer_size=autotune)
        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        
        # Modelo
        self.model = tf.keras.models.Sequential([
            # Camadas
            tf.keras.layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes)
        ])

        # Compilar Modelo
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy']
                          )

        # Treinar modelo
        epochs = 10
        history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs
        )

    def main(self, image):
        image_path = tf.keras.utils.get_file(origin=image)

        img = tf.keras.utils.load_img(
            image_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        os.system(command="cls")
        print(
            f"Essa imagem foi classificada como {self.class_names[np.argmax(score)]} com {100 * np.max(score)} porcetos de acerto."
        )


def main():
    ia = IA()
    args = sys.argv

    if args[1] == "--url":
        print("yes")
        ia.main(
            image=args[2]
        )

    elif args[1] == "--file":
        ia.main(image=args[2])


main()
