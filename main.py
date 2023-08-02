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
        # Carregar os dados de treinamento e validação, bem como as informações das classes
        self.train_ds, self.val_ds, self.class_names, self.num_classes = self.load_data()

        # Construir o modelo de rede neural
        self.model = self.build_model()

        # Treinar o modelo de rede neural usando os dados carregados
        self.train_model()

    def load_data(self):
        # Carregar os dados de treinamento e validação a partir do diretório especificado
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

        # Obter os nomes das classes a partir dos dados de treinamento
        class_names = train_ds.class_names

        # Determinar o número de classes
        num_classes = len(class_names)

        # Realizar pré-processamento e otimização dos datasets
        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)

        # Retornar os datasets e as informações das classes
        return train_ds, val_ds, class_names, num_classes

    def build_model(self):
        # Construir o modelo sequencial da rede neural
        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        
        model = tf.keras.models.Sequential([
            normalization_layer,
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

        # Compilar o modelo definindo a função de perda e métrica
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def train_model(self):
        # Treinar o modelo com os datasets de treinamento e validação
        epochs = 10
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )

    def main(self, image):
        if image.startswith("http://") or image.startswith("https://"):
            # Caso seja uma URL, carrega a imagem como antes
            image_path = tf.keras.utils.get_file(origin=image)
        else:
            # Caso seja um caminho de arquivo local, usa diretamente o caminho fornecido
            image_path = image

        img = tf.keras.utils.load_img(
            image_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Criar um batch de tamanho 1

        # Fazer previsões na imagem usando o modelo treinado
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Limpar a tela do terminal e exibir a classificação e a confiança do modelo
        os.system(command="cls")
        print(
            f"Essa imagem foi classificada como {self.class_names[np.argmax(score)]} com {100 * np.max(score)} porcento de acerto."
        )

def main():
    # Instanciar a classe IA e obter os argumentos da linha de comando
    ia = IA()
    args = sys.argv

    # Verificar os argumentos e chamar o método main com a imagem fornecida
    if args[1] == "--url" or args[1] == "--file":
        ia.main(image=args[2])

main()
