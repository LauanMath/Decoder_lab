
# Laboratório 3 - Implementando o Decoder

Este projeto implementa, em **NumPy**, as três partes pedidas no enunciado do **Laboratório 3**:

1. **Máscara Causal (Look-Ahead Mask)**
2. **Ponte Encoder-Decoder (Cross-Attention)**
3. **Loop de Inferência Auto-Regressivo**


## Arquivos

- `decoder.py` → implementação completa do laboratório


## Como executar

```bash
python decoder.py
```


## O que foi implementado

### 1) Máscara Causal
A função `create_causal_mask(seq_len)` retorna uma matriz quadrada `[seq_len, seq_len]` onde:

- diagonal principal = `0`
- triângulo inferior = `0`
- triângulo superior = `-inf`

Depois, o código:

- cria matrizes fictícias `Q` e `K`
- calcula `QK^T / sqrt(d_k)`
- soma a máscara
- aplica `softmax`
- imprime as probabilidades para provar que as posições futuras ficaram com **0.0**


### 2) Cross-Attention
O código cria tensores fictícios com as dimensões pedidas no enunciado:

- `encoder_output`: `[1, 10, 512]`
- `decoder_state`: `[1, 4, 512]`

Depois implementa a função:

```python
cross_attention(encoder_out, decoder_state)
```

Ela faz:

- projeção de `decoder_state` para `Q`
- projeção de `encoder_out` para `K`
- projeção de `encoder_out` para `V`
- cálculo do **Scaled Dot-Product Attention**
- sem máscara causal, como pedido

---

### 3) Loop de Inferência Auto-Regressivo
O código implementa a função:

```python
generate_next_token(current_sequence, encoder_out)
```

Ela:

- recebe a sequência já gerada
- simula o estado final do decoder
- projeta linearmente para um vocabulário fictício de **10.000 tokens**
- aplica `softmax`
- retorna um vetor de probabilidades

Depois, o loop:

- chama essa função iterativamente
- usa `argmax`
- adiciona o token à lista
- encerra imediatamente quando o token gerado for `<EOS>`


## Exemplo de execução

```bash
python decoder.py
```

Saída esperada:

- impressão da máscara causal
- impressão das probabilidades com posições futuras zeradas
- shapes do cross-attention
- loop gerando tokens até `<EOS>`
  
- Versão
  V1.0
