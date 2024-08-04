# Previsão de Roupas com Rede Neural Convolucional

Este repositório contém um projeto para prever categorias de roupas utilizando uma rede neural convolucional (CNN) com a base de dados Fashion MNIST. O código está escrito em Python e faz uso de bibliotecas populares de machine learning. O objetivo é criar e avaliar um modelo preditivo para classificar imagens de roupas em diferentes categorias.

## Funcionalidades Implementadas

1. ### Pré-processamento de Dados
   - Carregamento da base de dados Fashion MNIST e preparação dos dados para uso no modelo, incluindo normalização e expansão das dimensões das imagens.

2. ### Treinamento do Modelo
   - Configuração e treinamento de uma CNN para classificar imagens de roupas. O modelo é treinado com 100 épocas e utiliza técnicas como aumento de dados, normalização e dropout para melhorar a performance.

3. ### Avaliação do Modelo
   - Avaliação do modelo com métricas como precisão (accuracy) e visualização de desempenho através da matriz de confusão. Exibição de exemplos classificados erroneamente para análise qualitativa.

4. ### Salvamento e Carregamento do Modelo
   - Salvamento do modelo treinado em formato JSON e pesos em formato HDF5. Carregamento do modelo salvo para futuras previsões e avaliações.

## Ferramentas Utilizadas
- **Python**: Linguagem de programação principal.
- **Bibliotecas Python**: TensorFlow, Keras, NumPy, Matplotlib, scikit-learn.
- **Ambiente de Desenvolvimento**: Qualquer IDE Python como PyCharm ou Visual Studio Code.

## Estrutura do Projeto

- **`main.py`**: Script principal para carregar o modelo, realizar previsões e visualizar a matriz de confusão e exemplos.
- **`functions.py`**: Contém funções auxiliares para plotar a matriz de confusão e exibir exemplos.
- **`modelTraining/`**: Pasta contendo o modelo salvo e os pesos.
  - `model.json`: Arquivo com a arquitetura do modelo.
  - `model.weights.h5`: Arquivo com os pesos do modelo treinado.

## Como Usar

1. **Clone o repositório**:
    - `git clone <URL do repositório>`

2. **Instale as dependências necessárias**:
    - `pip install tensorflow numpy matplotlib scikit-learn`

3. **Certifique-se de que os arquivos do modelo estão na pasta `modelTraining/`**:
    - `model.json`
    - `model.weights.h5`

4. **Execute o script principal**:
    - `python main.py`
    - O script carregará o modelo, realizará previsões sobre o conjunto de teste e exibirá a matriz de confusão e exemplos classificados de forma incorreta.

5. **Modificações**:
    - Sinta-se à vontade para modificar o código em utils.py e main.py para ajustar os parâmetros do modelo ou experimentar com diferentes conjuntos de dados.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para enviar pull requests com melhorias, correções de bugs ou novas funcionalidades. Para discussões e sugestões, por favor, abra uma issue.