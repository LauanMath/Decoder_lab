
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# =========================================================
# UTILITÁRIOS
# =========================================================
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# =========================================================
# TAREFA 1 - MÁSCARA CAUSAL (LOOK-AHEAD MASK)
# =========================================================
def create_causal_mask(seq_len):

    mask = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
    mask = np.where(np.isneginf(mask), mask, 0.0)
    return mask


def prove_causal_mask():
    print("\n" + "=" * 70)
    print("TAREFA 1 - PROVA DA MÁSCARA CAUSAL")
    print("=" * 70)

    seq_len = 5
    d_k = 4

    Q = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 0.0, 0.0, 1.0],
    ])

    K = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 0.0, 0.0, 1.0],
    ])

    M = create_causal_mask(seq_len)
    scores = (Q @ K.T) / np.sqrt(d_k)
    masked_scores = scores + M
    attention_probs = softmax(masked_scores, axis=-1)

    print("Máscara causal M:")
    print(M)
    print("\nScores QK^T / sqrt(d_k):")
    print(scores)
    print("\nScores mascarados:")
    print(masked_scores)
    print("\nProbabilidades de atenção após softmax:")
    print(attention_probs)

    print("\nVerificação: probabilidades nas posições futuras devem ser 0.0")
    for i in range(seq_len):
        future_probs = attention_probs[i, i+1:]
        if future_probs.size > 0:
            print(f"Linha {i} -> futuras = {future_probs}")


# =========================================================
# TAREFA 2 - CROSS-ATTENTION
# =========================================================
def cross_attention(encoder_out, decoder_state, seed=42):
    """
    encoder_out:  [batch, seq_len_frances, d_model]
    decoder_state:[batch, seq_len_ingles, d_model]
    """
    rng = np.random.default_rng(seed)
    d_model = encoder_out.shape[-1]

    W_q = rng.normal(0, 0.02, size=(d_model, d_model))
    W_k = rng.normal(0, 0.02, size=(d_model, d_model))
    W_v = rng.normal(0, 0.02, size=(d_model, d_model))

    Q = decoder_state @ W_q                  
    K = encoder_out @ W_k                    
    V = encoder_out @ W_v                    

    scores = (Q @ np.transpose(K, (0, 2, 1))) / np.sqrt(d_model) 
    attn_weights = softmax(scores, axis=-1)                    
    context = attn_weights @ V 

    return Q, K, V, scores, attn_weights, context


def test_cross_attention():
    print("\n" + "=" * 70)
    print("TAREFA 2 - CROSS-ATTENTION")
    print("=" * 70)

    rng = np.random.default_rng(123)

    encoder_output = rng.normal(size=(1, 10, 512))
    decoder_state = rng.normal(size=(1, 4, 512))

    Q, K, V, scores, attn_weights, context = cross_attention(
        encoder_output, decoder_state, seed=7
    )

    print(f"Shape encoder_output: {encoder_output.shape}")
    print(f"Shape decoder_state:  {decoder_state.shape}")
    print(f"Shape Q:              {Q.shape}")
    print(f"Shape K:              {K.shape}")
    print(f"Shape V:              {V.shape}")
    print(f"Shape scores:         {scores.shape}")
    print(f"Shape attn_weights:   {attn_weights.shape}")
    print(f"Shape context:        {context.shape}")

    print("\nExemplo - soma das probabilidades da 1ª posição do decoder:")
    print(attn_weights[0, 0].sum())

    print("\nPrimeiros 10 pesos de atenção da 1ª query:")
    print(attn_weights[0, 0, :10])


# =========================================================
# TAREFA 3 - LOOP DE INFERÊNCIA AUTO-REGRESSIVO
# =========================================================
VOCAB_SIZE = 10_000
EOS_TOKEN = "<EOS>"
START_TOKEN = "<START>"

VOCAB = [f"token_{i}" for i in range(VOCAB_SIZE - 2)] + [START_TOKEN, EOS_TOKEN]
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for idx, token in enumerate(VOCAB)}

START_ID = TOKEN_TO_ID[START_TOKEN]
EOS_ID = TOKEN_TO_ID[EOS_TOKEN]


def generate_next_token(current_sequence, encoder_out, seed=None):

    if seed is None:
        seed = sum((i + 1) * sum(ord(c) for c in token) for i, token in enumerate(current_sequence)) % (2**32)

    rng = np.random.default_rng(seed)

    decoder_final_state = rng.normal(size=(512,))

    W_vocab = rng.normal(0, 0.02, size=(512, VOCAB_SIZE))
    logits = decoder_final_state @ W_vocab  # [VOCAB_SIZE]

    step = len(current_sequence) - 1

    logits[START_ID] = -1e9

    if step >= 4:
        logits[EOS_ID] += 3.5
    if step >= 6:
        logits[EOS_ID] += 6.0

    probs = softmax(logits, axis=-1)
    return probs


def inference_loop(max_steps=20):
    print("\n" + "=" * 70)
    print("TAREFA 3 - LOOP DE INFERÊNCIA AUTO-REGRESSIVO")
    print("=" * 70)

    rng = np.random.default_rng(999)
    encoder_out = rng.normal(size=(1, 10, 512))

    current_sequence = [START_TOKEN]

    print("Sequência inicial:", current_sequence)

    while True:
        probs = generate_next_token(current_sequence, encoder_out)

        next_token_id = int(np.argmax(probs))
        next_token = ID_TO_TOKEN[next_token_id]

        current_sequence.append(next_token)

        print(f"Passo {len(current_sequence)-1:02d} -> token gerado: {next_token}")

        if next_token == EOS_TOKEN:
            break

        if len(current_sequence) - 1 >= max_steps:
            print("Limite de segurança atingido. Encerrando loop.")
            break

    print("\nFrase final:")
    print(current_sequence)

    return current_sequence


# =========================================================
# EXECUÇÃO
# =========================================================
if __name__ == "__main__":
    prove_causal_mask()
    test_cross_attention()
    inference_loop()
