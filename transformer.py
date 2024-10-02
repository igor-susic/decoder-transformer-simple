"""
This script implements training for a simple transformer model
"""
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def generate_batches(encoded_corpus: Tensor, batch_size: int, text_chunk_size: int) -> tuple[Tensor, Tensor]:
    """
    Data should be N characters, where x is chars until element x and where Y is next element

    Let's say we choose:
        [47., 57.,  1., 58., 39., 49., 43., 52.] for X
        [57.,  1., 58., 39., 49., 43., 52., 0.] would be Y
    """
    random_points_in_dataset: Tensor = torch.randint(high=len(encoded_corpus) - text_chunk_size, size=(batch_size,))

    x_train: Tensor = torch.stack([encoded_corpus[rp:rp + text_chunk_size] for rp in random_points_in_dataset])
    y_train: Tensor = torch.stack([encoded_corpus[rp + 1:rp + text_chunk_size + 1] for rp in random_points_in_dataset])

    return x_train, y_train


class FeedFrowardNet(nn.Module):
    def __init__(self, model_dimensionality: int):
        """
        This consists of two linear transformations with a ReLU activation in between
        """
        super().__init__()

        INTERNAL_DIMENSION = 96

        self.ffn = nn.Sequential(
            nn.Linear(in_features=model_dimensionality, out_features=INTERNAL_DIMENSION),
            nn.ReLU(),
            nn.Linear(in_features=INTERNAL_DIMENSION, out_features=model_dimensionality)
        )

    def forward(self, x: Tensor):
        """
        Input shape (Batch, text_chunk, model_dimensionality)
        """
        out = self.ffn(x)

        return out  # (Batch, text_chunk, model_dimensionality)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, d_v: int, model_dimensionality: int):
        super().__init__()

        self.query = nn.Linear(in_features=model_dimensionality, out_features=d_k, bias=False)
        self.key = nn.Linear(in_features=model_dimensionality, out_features=d_k, bias=False)
        self.value = nn.Linear(in_features=model_dimensionality, out_features=d_v, bias=False)

        self.register_buffer(name="tril", tensor=torch.tril(torch.ones(TEXT_CHUNK_SIZE, TEXT_CHUNK_SIZE)))

    def forward(self, x: Tensor):
        """
        Input tensor will be of shape: (batch, text_chunk, model_dimensionality)
        """


        q: Tensor = self.query(x)  # (Batch, text_chunk, d_k)
        k: Tensor = self.key(x)  # (Batch, text_chunk, d_k)
        v: Tensor = self.value(x)  # (Batch, text_chunk, d_v)

        _, sequence_size, D_K = k.shape


        # We want to transpose last two dimensions to get (Batch, text_chunk, d_k) @ (Batch, d_k, text_chunk)
        score = q @ k.transpose(-2, -1)  # (Batch, text_chunk, text_chunk)
        score = score / math.sqrt(D_K)  # (Batch, text_chunk, text_chunk)
        # (Batch, text_chunk, text_chunk)
        score = score.masked_fill(~self.tril.to(torch.bool)[:sequence_size, :sequence_size], float(-math.inf))
        score = F.softmax(input=score, dim=1)  # (Batch, text_chunk, text_chunk)
        score = score @ v  # (Batch, text_chunk, text_chunk) @ (Batch, text_chunk, d_v)

        return score  # (Batch, text_chunk, d_v)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, number_of_heads: int, d_k: int, d_v: int, model_dimensionality: int) -> None:
        super().__init__()

        self.scaled_dot_product_attention_stack = nn.ModuleList(
            [ScaledDotProductAttention(
                d_k=d_k,
                d_v=d_v,
                model_dimensionality=model_dimensionality
            ) for _ in range(number_of_heads)]
        )

        self.linear = nn.Linear(in_features=number_of_heads * d_v, out_features=model_dimensionality)

    def forward(self, x: Tensor):
        """
        Should look like this:
            - input into -> linear layer -> into scaled dot product attention -> concat -> linear

        Input tensor will be of shape: (batch, text_chunk, model_dimensionality)
        """
        results = []

        for head in self.scaled_dot_product_attention_stack:
            out: Tensor = head(x)  # (Batch, text_chunk, d_v)
            results.append(out)

        concatenated_heads = torch.concat(results, dim=-1)  # (Batch, text_chunk, d_v * number_of_heads)
        # d_v * number_of_heads == model_dimensionality

        out = self.linear(concatenated_heads)  # (Batch, text_chunk, model_dimensionality)

        return out  # (Batch, text_chunk, model_dimensionality)


class Block(nn.Module):
    def __init__(self, model_dimensionality: int, d_k: int, d_v: int, number_of_att_heads: int):
        super().__init__()

        self.masked_multi_head_attention = MaskedMultiHeadAttention(
            number_of_heads=number_of_att_heads,
            d_k=d_k,
            d_v=d_v,
            model_dimensionality=model_dimensionality
        )

        self.first_normalization_layer = nn.LayerNorm(model_dimensionality)

        self.feed_forward_net = FeedFrowardNet(
            model_dimensionality=model_dimensionality
        )

        self.second_normalization_layer = nn.LayerNorm(model_dimensionality)

    def forward(self, x: Tensor):
        """
        Forward pass should look like:
            - input -> masked multihead -> add residual and layer norm 1 -> ffn -> add and layer norm 2

        Input tensor will be of shape: (batch, text_chunk, model_dimensionality)
        """

        # (Batch, text_chunk, model_dimensionality)
        out: Tensor = self.masked_multi_head_attention(self.first_normalization_layer(x))


        out1 = x + out  # (Batch, text_chunk, model_dimensionality)

        out2 = self.feed_forward_net(self.second_normalization_layer(out1))  # (Batch, text_chunk, model_dimensionality)
        out2 = out1 + out2  # (Batch, text_chunk, model_dimensionality)

        return out2  # (Batch, text_chunk, model_dimensionality)


class Transformer(nn.Module):
    def __init__(
            self,
            vocabulary_size: int,
            model_dimensionality: int,
            positional_embeddings_len: int,
            d_k: int,
            d_v: int,
            number_of_att_heads: int,
            number_of_blocks: int
    ):
        super().__init__()

        self.embedding_table = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=model_dimensionality)
        self.position_table = nn.Embedding(num_embeddings=positional_embeddings_len, embedding_dim=model_dimensionality)

        self.stacked_blocks = nn.Sequential(
            *[Block(
                model_dimensionality=model_dimensionality,
                d_k=d_k,
                d_v=d_v,
                number_of_att_heads=number_of_att_heads
            ) for _ in range(number_of_blocks)]
        )

        self.un_embedding_layer = nn.Linear(in_features=model_dimensionality, out_features=vocabulary_size)

    def forward(self, x: Tensor):
        """
        :param x: Has the shape of (1, TEXT_CHUNK_SIZE)

        :return:
        """

        _, T = x.shape

        # Output will be shape (batch, text_chunk, model_dimensionality)
        input_embeddings: Tensor = self.embedding_table(x)

        # Output will be (text_chunk, model_dimensionality)
        positional_embeddings: Tensor = self.position_table(torch.arange(T))


        embeddings: Tensor = input_embeddings.add(positional_embeddings)  # (batch, text_chunk, model_dimensionality)


        out = self.stacked_blocks(embeddings)  # (Batch, text_chunk, model_dimensionality)
        out1 = self.un_embedding_layer(out)  # (Batch, text_chunk, vocabulary_size)

        return out1

    def generate(self, max_length: int, input_text: Tensor) -> Tensor:
        """
        1. Start from some text chunk
        2. use predicted text to predict again
        3. do until max length
        """
        for i in range(max_length):
            input_truncated = input_text[:, -TEXT_CHUNK_SIZE:]  # Take last chunk of text
            logits: Tensor = self(input_truncated)  # (1, text_chunk, vocab_size)

            logits = logits[:, -1, :]
            probabilities: Tensor = F.softmax(logits)  # (1, text_chunk, vocab_size)

            idx_next = torch.multinomial(probabilities, num_samples=1)
            input_text = torch.cat((input_text, idx_next), dim=1)

        return input_text


if __name__ == '__main__':
    with open("./shakespeare.txt") as dataset_file:
        data = dataset_file.read()

    vocabulary = set(data)  # Set of unique characters in dataset
    vocabulary = list(vocabulary)
    vocabulary.sort()  # Sort the vocab for easier inspection

    # Mapping tables
    index_to_char = {index: char for index, char in enumerate(vocabulary)}
    char_to_index = {char: index for index, char in enumerate(vocabulary)}

    # Encode and decode functions
    encode = lambda string: [char_to_index.get(c) for c in string]
    decode = lambda embedding: "".join([index_to_char.get(index) for index in embedding])

    TEXT_CHUNK_SIZE = 8

    train_set_percentage = 0.9
    train_test_split_index = int(len(data) * 0.9)

    train_set = data[:train_test_split_index]
    test_set = data[train_test_split_index:]

    train_set = torch.tensor(encode(train_set), dtype=torch.long)
    test_set = torch.tensor(encode(test_set), dtype=torch.long)

    model = Transformer(
        vocabulary_size=len(vocabulary),
        model_dimensionality=64,
        positional_embeddings_len=TEXT_CHUNK_SIZE,
        d_k=64 // 4,
        d_v=64 // 4,
        number_of_att_heads=4,
        number_of_blocks=6
    )

    # Init loss and optimizer
    loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
    )

    model.train()

    loss = None
    for step in range(100_000):

        # x_tr shape (batch_size, text_chunk)
        # y_tr shape (batch_size, text_chunk)
        x_tr, y_tr = generate_batches(encoded_corpus=train_set, batch_size=32, text_chunk_size=TEXT_CHUNK_SIZE)

        # compute error between predicted word and actual word
        pred: Tensor = model(x_tr)  # (Batch, text_chunk, vocabulary_size)

        # Reshape into one array of next tokens which we expect (list of indexes)
        batch_size, text_chunk = y_tr.shape
        target = y_tr.view(batch_size * text_chunk)

        # Reshape logits (predictions) into batch array of (batch_size * text_chunk, vocab)
        batch_size_m, text_chunk_m, vocab_size_m = pred.shape
        logits = pred.view(batch_size_m * text_chunk_m, vocab_size_m)

        loss = loss_fn(logits, target)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            loss_score = loss.item()
            print(f"loss: {loss_score:>7f}")

    print(decode(
        model.generate(
            input_text=torch.tensor(encode("ROMEO"), dtype=torch.long).unsqueeze(dim=0),
            max_length=1_000)
        [0].tolist()
    ))
