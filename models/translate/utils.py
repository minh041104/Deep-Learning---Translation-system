import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
import math

class Vocabulary:
    def __init__(self, name, sentences, nlp):
        self.name = name
        self.word2index = {"<sos>": 0, "<eos>": 1, "<unk>": 2}
        self.word2count = {}
        self.index2word = {0: "<sos>", 1: "<eos>", 2: "<unk>"}
        self.n_words = 3  
        self.nlp = nlp
        self.add_sentences(sentences)

    def add_sentences(self, sentences):
        for sentence in sentences:
            self.add_sentence(sentence)

    def add_sentence(self, sentence):
        for word in self.nlp.tokenizer(sentence):
            self.add_word(word.text)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def get_index(self, word):
        return self.word2index.get(word, self.word2index["<unk>"])  # Trả về UNK nếu từ không có
        
def indexesFromSentence(vocab, sentence, nlp):
    return [vocab.word2index.get(word.text, vocab.word2index['<unk>']) for word in nlp.tokenizer(sentence)]

def tensorFromSentence(vocab, sentence, nlp, device, eos_token=1):
    indexes = indexesFromSentence(vocab, sentence, nlp)
    indexes.append(eos_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)
def get_dataloader(data, src_vocab, tgt_vocab, src_nlp, tgt_nlp, device, batch_size = 64,\
                   max_length=20, sos_token = 0, eos_token = 1):
    input_ids = np.zeros((len(data), max_length), dtype=np.int32)
    target_ids = np.zeros((len(data), max_length), dtype=np.int32)
    for idx, row in enumerate(data):
        inp = row['en']
        tgt = row['vi']
        inp_ids = indexesFromSentence(src_vocab, inp, src_nlp)[:max_length-1]
        tgt_ids = indexesFromSentence(tgt_vocab, tgt, tgt_nlp)[:max_length-1]
        inp_ids.append(eos_token)
        tgt_ids.append(eos_token)

        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    tensor_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                            torch.LongTensor(target_ids).to(device))
    dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)
    return dataloader

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))