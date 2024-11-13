import os
import torch
# from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset, random_split
# from sklearn.model_selection import train_test_split
from tqdm import tqdm


class TextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42
    VAL_RATIO = 0.05

    def __init__(self, data_file: str, data_file_for_tokenizer: str, train: bool = True, sp_model_prefix: str = None,
                 vocab_size: int = 5000, normalization_rule_name: str = 'nmt_nfkc_cf',
                 model_type: str = 'bpe', max_length: int = 216):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: txt file containing texts
        :param train: whether to use train or validation split
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """
        print("Start building TextDataset")
        if not os.path.isfile(sp_model_prefix + '.model'):
            # train tokenizer if not trained yet
          SentencePieceTrainer.train(
              input=data_file_for_tokenizer, vocab_size=vocab_size,
              model_type=model_type, model_prefix=sp_model_prefix,
              normalization_rule_name=normalization_rule_name,
              pad_id=3,
          )
        # load tokenizer from file
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model')
        
        print("Prepared model for TextDataset")

        with open(data_file) as file:
            texts = file.readlines()
            
        print("Read all texts for forming TextDataset")

        lengths = [len(texts) - int(len(texts)*self.VAL_RATIO), int(len(texts)*self.VAL_RATIO)]
        # train_texts, val_texts = train_test_split(texts, test_size=self.VAL_RATIO, random_state=self.TRAIN_VAL_RANDOM_SEED)
        train_texts, val_texts = random_split(texts, lengths)
        self.texts = list(train_texts if train else val_texts)
        
        print("Start encoding")
        self.indices = self.sp_model.encode(self.texts)
        print("Encoding finished")

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
            self.sp_model.bos_id(), self.sp_model.eos_id()
        self.max_length = max_length
        self.vocab_size = self.sp_model.vocab_size()

    def text2ids(self, texts):
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.sp_model.encode(texts)

    def ids2text(self, ids):
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.indices)

    def __getitem__(self, item: int):
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """
        
        """
        YOUR CODE HERE (вЉѓпЅЎвЂўМЃвЂївЂўМЂпЅЎ)вЉѓв”Ѓвњївњївњївњївњївњї
        Take corresponding index array from self.indices,
        add special tokens (self.bos_id and self.eos_id) and 
        pad to self.max_length using self.pad_id.
        Return padded indices of size (max_length, ) and its actual length
        """
        indices = self.indices[item].copy()
        indices = [self.bos_id] + indices + [self.eos_id]
        indices = indices[:self.max_length]
        padded = torch.full((self.max_length, ), self.pad_id)
        padded[:len(indices)] = torch.tensor(indices)
        return padded, len(indices)