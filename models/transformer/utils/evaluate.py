import torch
import numpy as np
from torchtext.data.metrics import bleu_score
from rouge_score import rouge_scorer


def translate_sentence(sentence, model, SRC, TRG, device, k, max_len):
    model.eval()
    tokens = SRC.preprocess(sentence)  # Token hóa câu
    indexed = []

    for tok in tokens:
        if tok in SRC.vocab.stoi:  # Kiểm tra từ trong từ điển
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(SRC.vocab.stoi["<unk>"])  # Dùng <unk> nếu từ không có trong từ điển

    # Chuyển danh sách sang tensor
    sentence_tensor = torch.LongTensor([indexed]).to(device)

    # Áp dụng thuật toán tìm kiếm (ví dụ: beam search)
    prediction = beam_search(sentence_tensor, model, SRC, TRG, device, k, max_len)

    return prediction

def bleu(valid_src_data, valid_trg_data, model, SRC, TRG, device, k, max_strlen):
    pred_sents = []
    for sentence in valid_src_data:
        pred_trg = translate_sentence(sentence, model, SRC, TRG, device, k, max_strlen)
        pred_sents.append(pred_trg)

    pred_sents = [TRG.preprocess(sent) for sent in pred_sents]
    trg_sents = [[sent.split()] for sent in valid_trg_data]

    return bleu_score(pred_sents, trg_sents)

def rouge(valid_src_data, valid_trg_data, model, SRC, TRG, device, k, max_strlen):
    pred_sents = []
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    score = []
    score_rouge1 = 0
    for sentence, trg_sents in zip(valid_src_data, valid_trg_data):
        pred_trg = translate_sentence(sentence, model, SRC, TRG, device, k, max_strlen)
        # pred_sents.append(pred_trg)
        scores = rouge.score(trg_sents, pred_trg)
        score_rouge1 += scores['rouge1'].fmeasure  # Sử dụng fmeasure cho ROUGE-1
    
    return score_rouge1/len(valid_src_data)
