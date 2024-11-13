# from tempfile import tempdir
import torch
# from typing import Type
from torch import nn
from dataset import TextDataset
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.categorical import Categorical
# import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)
PAD_IDX = 3

def scaled_softmax_attention(query, key, value):
    """
    Args:
        query: torch.Tensor (..., L, D)
        key: torch.Tensor (..., L, D)
        value: torch.Tensor (..., L, D)
    Returns:
        res: torch.Tensor (..., L, D), output of the attention layer (\softmax(Q K^T / d) V
        attention: torch.Tensor (..., L, L), attention weights (\softmax(Q K^T / d))

    L is the length of sequence, D is the embedding dimension
    """

    L = query.shape[-2]
    D = query.shape[-1]
    # print(query.shape, key.shape)
    mask = (torch.triu(torch.ones(L, L)) == 1).transpose(0, 1).reshape((1, L, L))
    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    attention = torch.matmul(query, torch.transpose(key, -1, -2)) / (D ** 0.5)
    # print(attention.shape)
    attention = torch.where(mask == 1, attention, torch.full((1, L, L), float('-inf')))
    attention = torch.softmax(attention, dim=-1)
    res = torch.matmul(attention, value)

    return res, attention


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: dimensionality of embedding (total)
            num_heads: number of heads (must divide embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self._reset_parameters()

    # original implementation uses this initialization
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: torch.Tensor (B, L, D)
            return_attention: If specified, returns attention along with outputs
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)

        B is batch size, L is the length of sequence, D is the embedding dimension
        """

        B, L, D = x.shape
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        outputs = []
        attention = []
        for i in range(self.num_heads):
          l_index = self.head_dim * i
          r_index = self.head_dim * (i + 1)
          outputs_i, attention_i = scaled_softmax_attention(query[:,:,l_index:r_index], key[:,:,l_index:r_index], value[:,:,l_index:r_index])
          outputs.append(outputs_i)
          attention.append(attention_i.reshape(B, 1, L, L))
        outputs = torch.cat(outputs, dim=2)
        attention = torch.cat(attention, dim=1)
        outputs = self.o_proj(outputs)

        if return_attention:
            return outputs, attention
        else:
            return outputs


class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, feedforward_dim, activation=nn.ReLU, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            feedforward_dim - Dimensionality of the hidden layer in the MLP
            activation - activation function in FFN
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.activation = activation
        self.dropout = dropout
        self.multihead_attention = MultiheadAttention(embed_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, return_attention=False):
        """
        Args:
            x: torch.Tensor (B, L, D)
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)
        """

        outputs, attention = self.multihead_attention(x, return_attention=True)
        x = self.norm1(self.dropout(x + outputs))
        x = self.norm2(self.dropout(x + self.feedforward(x)))
        outputs = x

        if return_attention:
            return outputs, attention
        else:
            return outputs


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        self.pe = torch.zeros(max_len, embed_dim)

        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        freqs = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * \
                          (-torch.log(torch.tensor([10000]))[0].item() / embed_dim)).unsqueeze(0)

        arguments = positions * freqs
        self.pe[:, 0::2] = torch.sin(arguments)
        self.pe[:, 1::2] = torch.cos(arguments)
        self.pe = self.pe.unsqueeze(0)

        self.pe = nn.Parameter(self.pe, requires_grad=False)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        # self.register_buffer('pe', self.pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x

    
def generate_square_subsequent_mask(sz):
#         mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 0).transpose(0, 1).bool()
#         print(mask.dtype)
#         print(mask[:3, :3])
        return mask
    

class TransformerForStoryGeneration(nn.Module):

    def __init__(
        self,
        dataset: TextDataset,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int,
        activation = nn.GELU,
        dropout: float = 0.0,
        device: torch.device = 'cpu'
    ):
        super().__init__()

        self.dataset = dataset
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.num_layers = num_layers
        self.activation = activation
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length
        self.dropout = dropout
        self.device = device

        # define layers
        # self.input_embedding = nn.Linear(input_dim, embed_dim, bias=False)
        self.input_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_dim,
                                      padding_idx=dataset.pad_id)
        self.positional_encoding = PositionalEncoding(embed_dim, self.max_length)

        # self.encoder_blocks = []
        # for i in range(num_layers):
        #   self.encoder_blocks.append(EncoderBlock(embed_dim, num_heads, feedforward_dim, activation, dropout))
        # for i in range(num_layers):
        #   self.encoder_blocks.append(nn.TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout, activation, batch_first=True))
        # self.encoder = nn.Sequential(
        #     *self.encoder_blocks
        # )
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=feedforward_dim, dropout=dropout, activation=activation(), batch_first=True), num_layers=num_layers)

        self.classifier = nn.Linear(embed_dim, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        indices = indices[:,:lengths.max()]
        indices = indices.to(self.device)
        embeds = self.input_embedding(indices)
        embeds = self.positional_encoding(embeds)
        L = embeds.shape[1]
        src_mask = generate_square_subsequent_mask(L)
#         src_padding_mask = (indices == PAD_IDX)
#         src_padding_mask = src_padding_mask.float().masked_fill(mask == True, float('-inf')).masked_fill(mask == False, float(0.0))
#         outputs = self.encoder(embeds, mask=src_mask, src_key_padding_mask=src_padding_mask)
        outputs = self.encoder(embeds, mask=src_mask)
        logits = self.classifier(outputs)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        tokens = self.dataset.text2ids(prefix)
        tokens = [self.dataset.bos_id] + tokens
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)

        embeds = self.input_embedding(tokens)
        embeds = self.positional_encoding(embeds)
        L = embeds.shape[1]
        src_mask = generate_square_subsequent_mask(L)
        outputs = self.encoder(embeds, mask=src_mask)
        logits = self.classifier(outputs)

        new_tokens = Categorical(logits=(logits[:, -1:]/temp)).sample()
        tokens = torch.cat([tokens, new_tokens], dim=1)

        while tokens.shape[1] < self.max_length:
            if new_tokens.item() == self.dataset.eos_id:
                break

            embeds = self.input_embedding(tokens)
            embeds = self.positional_encoding(embeds)
            L = embeds.shape[1]
            src_mask = generate_square_subsequent_mask(L)
            outputs = self.encoder(embeds, mask=src_mask)
            logits = self.classifier(outputs)
            
            new_tokens = Categorical(logits=(logits[:, -1:]/temp)).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)

        return self.dataset.ids2text(tokens.squeeze()[1:-1])


# class LanguageModel(nn.Module):
#     def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
#                  rnn_type: Type = nn.RNN, rnn_layers: int = 1, device: torch.device = 'cpu'):
#         """
#         Model for text generation
#         :param dataset: text data dataset (to extract vocab_size and max_length)
#         :param embed_size: dimensionality of embeddings
#         :param hidden_size: dimensionality of hidden state
#         :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
#         :param rnn_layers: number of layers in RNN
#         """
#         super(LanguageModel, self).__init__()
#         self.dataset = dataset  # required for decoding during inference
#         self.vocab_size = dataset.vocab_size
#         self.max_length = dataset.max_length

#         """
#         YOUR CODE HERE (вЉѓпЅЎвЂўМЃвЂївЂўМЂпЅЎ)вЉѓв”Ѓвњївњївњївњївњївњї
#         Create necessary layers
#         """
#         self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size,
#                                       padding_idx=dataset.pad_id)
#         self.rnn = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
#         self.linear = nn.Linear(in_features=hidden_size, out_features=self.vocab_size)
#         self.device = device

#     def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
#         """
#         Compute forward pass through the model and
#         return logits for the next token probabilities
#         :param indices: LongTensor of encoded tokens of size (batch_size, length)
#         :param lengths: LongTensor of lengths of size (batch_size, )
#         :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
#         """

#         """
#         YOUR CODE HERE (вЉѓпЅЎвЂўМЃвЂївЂўМЂпЅЎ)вЉѓв”Ѓвњївњївњївњївњївњї
#         Convert indices to embeddings, pass them through recurrent layers
#         and apply output linear layer to obtain the logits
#         """
#         indices = indices[:,:lengths.max()]
#         indices = indices.to(self.device)
#         embeds = self.embedding(indices)
#         # print('embeds', embeds.shape, flush=True)
#         packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
#         # print('packed_embeds', packed_embeds.data.shape, flush=True)
#         outputs, hidden = self.rnn(packed_embeds)
#         # outputs, hidden = self.rnn(embeds)
#         # print('outputs', outputs.data.shape, flush=True)
#         outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
#         # print('outputs_and_lengths', outputs.shape, lengths.shape)
#         logits = self.linear(outputs)
#         return logits

#     @torch.inference_mode()
#     def inference(self, prefix: str = '', temp: float = 1.) -> str:
#         """
#         Generate new text with an optional prefix
#         :param prefix: prefix to start generation
#         :param temp: sampling temperature
#         :return: generated text
#         """
#         self.eval()
#         """
#         YOUR CODE HERE (вЉѓпЅЎвЂўМЃвЂївЂўМЂпЅЎ)вЉѓв”Ѓвњївњївњївњївњївњї
#         Encode the prefix (do not forget the BOS token!),
#         pass it through the model to accumulate RNN hidden state and
#         generate new tokens sequentially, sampling from categorical distribution,
#         until EOS token or reaching self.max_length.
#         Do not forget to divide predicted logits by temperature before sampling
#         """
#         tokens = self.dataset.text2ids(prefix)
#         tokens = [self.dataset.bos_id] + tokens
#         tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)

#         embeds = self.embedding(tokens)
#         # print(tokens)
#         output, hidden = self.rnn(embeds)
#         # print(output.size(), hidden.size())
#         logits = self.linear(output)

#         new_tokens = Categorical(logits=(logits[:, -1:]/temp)).sample()
#         tokens = torch.cat([tokens, new_tokens], dim=1)

#         while tokens.shape[1] < self.max_length:
#             if new_tokens.item() == self.dataset.eos_id:
#                 break

#             embeds = self.embedding(new_tokens)
#             output, hidden = self.rnn(embeds, hidden)
#             logits = self.linear(output)
            
#             new_tokens = Categorical(logits=(logits[:, -1:]/temp)).sample()
#             tokens = torch.cat([tokens, new_tokens], dim=1)

#         return self.dataset.ids2text(tokens.squeeze()[1:-1])