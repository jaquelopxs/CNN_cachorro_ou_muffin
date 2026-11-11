# Classificador Chihuahua vs. Muffin

Classificação de imagens com Redes Neurais Convolucionais (CNNs) para resolver o clássico desafio **Chihuahua vs. Muffin**, comparando a mesma arquitetura implementada em **TensorFlow/Keras** e em **PyTorch**.

## 📋 Etapas do projeto

1. **Instalação e importação:** configuração das bibliotecas (TensorFlow, PyTorch, scikit-learn, Pillow, entre outras) e definição da semente aleatória 42 para garantir a reprodutibilidade.
2. **Carregamento e validação:** leitura das imagens nas extensões aceitas e verificação de arquivos corrompidos com o Pillow.
3. **Distribuição e divisão:** contagem de imagens por classe e separação estratificada em treino (70%), validação (15%) e teste (15%).
4. **Pipelines de entrada:** criação de um `tf.data.Dataset` (com `prefetch`) e de um `DataLoader` do PyTorch, com as imagens redimensionadas para 64×64 pixels.
5. **Modelagem:** arquiteturas CNN equivalentes nos dois frameworks, com 3 blocos convolucionais (32, 64 e 128 filtros) seguidos de Max-Pooling, uma camada densa de 128 neurônios e Dropout de 0,4 para reduzir o overfitting.
6. **Treinamento:** otimizador Adam, função de perda Binary Cross-Entropy e parada antecipada (Early Stopping) com paciência de 4 épocas.
7. **Avaliação:** acurácia, precisão, recall e F1-score com scikit-learn, matrizes de confusão e teste com imagens externas obtidas por URL.

## 📊 Resultados

Os dois modelos foram avaliados no mesmo conjunto de teste (210 imagens). A classe positiva usada no cálculo da precisão, do recall e do F1-score é **Muffin**.

| Modelo | Acurácia | Precisão | Recall | F1-score |
|---|---|---|---|---|
| TensorFlow/Keras | 93,33% | 90,67% | 90,67% | 90,67% |
| PyTorch | 89,05% | 89,39% | 78,67% | 83,69% |

### Conclusões

- O modelo em **TensorFlow/Keras** teve o melhor desempenho em todas as métricas, com F1-score de 90,67%.
- O modelo em **PyTorch** teve precisão próxima à do Keras (89,39%), mas recall bem menor (78,67%). Ou seja, quando ele classifica uma imagem como muffin, geralmente acerta, mas deixa de reconhecer cerca de 1 em cada 5 muffins, classificando-os como chihuahua.
- Como as arquiteturas são equivalentes, a diferença provavelmente vem de detalhes de implementação e treinamento entre os frameworks, como a inicialização dos pesos e o momento em que a parada antecipada interrompeu o treino.
- Os testes com imagens da internet servem como demonstração. Eles não substituem a avaliação no conjunto de teste.

## ▶️ Como executar

O projeto foi desenvolvido para rodar no **Google Colab**.

1. Baixe a base de imagens: [muffin](https://drive.google.com/drive/folders/1MFwqUVXqWklFK1ihstgnrHynh8AH7OCM?usp=drive_link).[cachorro](https://drive.google.com/drive/folders/162nyisf-Ke8yf7SqdHMgF1Dp8V-dKCvA?usp=drive_link).
2. No seu Google Drive, crie a pasta `Sistemas Inteligentes` com duas subpastas, `chihuahua` e `muffin`, e coloque as imagens de cada classe na pasta correspondente.
3. Abra o código no Colab, ative a GPU em **Ambiente de execução → Alterar o tipo de ambiente de execução** e execute as etapas em ordem.

Bibliotecas utilizadas: `tensorflow`, `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `pillow`, `pandas`, `numpy` e `requests`.

## 💾 Arquivos gerados

Ao final da execução, os arquivos abaixo são salvos automaticamente na pasta de resultados do Google Drive. Eles **não estão incluídos neste repositório**.

- `modelo_chihuahua_muffin.keras`: arquitetura e pesos do modelo Keras.
- `modelo_chihuahua_muffin.pth`: dicionário de estados (state dict) do modelo PyTorch.
- `metricas_modelos.csv`: tabela comparativa das métricas.
