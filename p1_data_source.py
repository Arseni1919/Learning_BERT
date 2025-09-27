import random
import typing
from collections import Counter

import numpy as np
from torch.utils.data import Dataset
import  pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

general_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class IMDBBertDataset(Dataset):
    CLS =general_tokenizer.cls_token
    PAD = general_tokenizer.pad_token
    SEP = general_tokenizer.sep_token
    MASK = general_tokenizer.mask_token
    UNK = general_tokenizer.unk_token

    MASK_PERCENTAGE = 0.15  # How much words to mask

    MASKED_INDICES_COLUMN = 'masked_indices'
    TARGET_COLUMN = 'indices'
    NSP_TARGET_COLUMN = 'is_next'
    TOKEN_MASK_COLUMN = 'token_mask'

    OPTIMAL_LENGTH_PERCENTILE = 70

    def __init__(
            self, path, ds_from=None, ds_to=None, should_include_text=False,
    ):
        self.ds: pd.Series = pd.read_csv(path)['review']

        if ds_from is not None or ds_to is not None:
            self.ds = self.ds[ds_from:ds_to]

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.counter = Counter()
        self.vocab = None

        self.optimal_sentence_length = None
        self.should_include_text = should_include_text

        if should_include_text:
            self.columns = ['masked_sentence', self.MASKED_INDICES_COLUMN, 'sentence',
                            self.TARGET_COLUMN, self.TOKEN_MASK_COLUMN, self.NSP_TARGET_COLUMN]
        else:
            self.columns = [self.MASKED_INDICES_COLUMN,
                            self.TARGET_COLUMN, self.TOKEN_MASK_COLUMN, self.NSP_TARGET_COLUMN]

        self.df = self.prepare_dataset()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ...

    def _find_optimal_sentence_length(self, lengths: typing.List[int]):
        arr = np.array(lengths)
        return int(np.percentile(arr, self.OPTIMAL_LENGTH_PERCENTILE))

    def _fill_vocab(self):
        self.vocab = self.tokenizer.get_vocab()

    def _create_item(self,first: typing.List[int], second: typing.List[int], target: int = 1):
        update_first, first_mask = self._preprocess_sentence(first.copy())
        update_second, second_mask = self._preprocess_sentence(second.copy())
        nsp_sentence = update_first + [self.SEP] + update_second
        nsp_indices = self.vocab.lookup_indices(nsp_sentence)
        inverse_token_mask = first_mask + [True] + second_mask

    def _mask_sentence(self, sentence: typing.List[str]):
        len_s = len(sentence)
        inverse_token_mask = [True for _ in range(max(len_s, self.optimal_sentence_length))]

        mask_amount = round(len_s * self.MASK_PERCENTAGE)
        for _ in range(mask_amount):
            i = random.randint(0, len_s - 1)

            if random.random() < 0.8:
                sentence[i] = self.MASK
            else:
                j = random.randint(5, len(self.vocab) - 1)
                sentence[i] = self.vocab.lookup_token(j)
            inverse_token_mask[i] = False

        return sentence, inverse_token_mask

    def prepare_dataset(self) -> pd.DataFrame:
        sentences = []
        nsp = []  # next sentence prediction
        sentence_lens = []

        # Split dataset on sentences
        for review in self.ds:
            review_sentences = review.split('. ')
            sentences += review_sentences
            self._update_length(review_sentences, sentence_lens)
        self.optimal_sentence_length = self._find_optimal_sentence_length(sentence_lens)
        print('Created vocabulary')
        for sentence in tqdm(sentences):
            s = self.tokenizer(sentence)
            self.counter.update(s)

        self._fill_vocab()

        print("Preprocessing dataset")
        for review in tqdm(self.ds):
            review_sentences = review.split('. ')
            if len(review_sentences) > 1:
                for i in range(len(review_sentences) - 1):
                    # True NSP item
                    first, second = self.tokenizer(review_sentences[i]),self.tokenizer(review_sentences[i + 1])
                    nsp.append(self._create_item(first, second, 1))

                    # False NSP item
                    first, second = self._select_false_nsp_sentences(sentences)
                    first, second = self.tokenizer(first), self.tokenizer(second)
                    nsp.append(self._create_item(first, second, 0))

        df = pd.DataFrame(nsp, columns=self.columns)

