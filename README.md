# Classificação de imagens com TensorFlow

Este código realiza a classificação de imagens utilizando a biblioteca TensorFlow.
Requisitos

- Python 3.x
- TensorFlow 2.x

# Como utilizar

Baixe o código em um diretório de sua preferência.
Acesse o diretório onde o código foi baixado e execute o seguinte comando para classificar uma imagem pela URL:



    python3 main.py --url {url_da_imagem}

Exemplo:

    python3 main.py --url https://www.example.com/image.jpg

Para classificar uma imagem local, execute o seguinte comando:

    python3 main.py --file {caminho_da_imagem}

Exemplo:

    python3 main.py --file /home/user/images/image.jpg

# Como funciona

O código carrega as imagens presentes em um diretório e separa em um conjunto de dados de treinamento e outro de validação. Em seguida, é realizado o pré-processamento dos dados e o modelo de rede neural convolucional é definido.

O modelo é compilado com a função de perda e métrica apropriadas e então treinado com o conjunto de dados de treinamento. Após o treinamento, o modelo é capaz de realizar a classificação de novas imagens.

Ao receber uma imagem como entrada, o programa a processa e utiliza o modelo treinado para realizar a classificação, retornando a categoria da imagem e a porcentagem de acerto da classificação.
