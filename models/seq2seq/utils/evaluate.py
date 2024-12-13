import random
from utils.preprocessor import tensorFromSentence
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import torch

def evaluate(encoder, decoder, src_sentence, src_vocab, tgt_vocab, src_nlp, device):
    
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensorFromSentence(src_vocab, src_sentence, src_nlp, device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == 1 or idx.item() == 0:
                break
            decoded_words.append(tgt_vocab.index2word[idx.item()])
    return decoded_words

def translate_sentence(encoder, decoder, sentences, src_vocab, tgt_vocab, src_nlp, device):
    results = []
    for s in sentences:
        output_words = evaluate(encoder, decoder, s, src_vocab, tgt_vocab, src_nlp, device)
        output_sentence = ' '.join(output_words)
        results.append(output_sentence)
    return results

def evaluateRandom(encoder, decoder, data, src_vocab, tgt_vocab, src_nlp, tgt_nlp, device, n=10):
    total_bleu = 0
    for i in range(n):
        pair = random.choice(data)
        print('English:\t', pair['en'])
        print('Vi_true:\t', pair['vi'])

        output_words = evaluate(encoder, decoder, pair['en'], src_vocab, tgt_vocab, src_nlp, device)
        output_sentence = ' '.join(output_words)
        print('Vi_pred:\t', output_sentence)

        reference = [[token.text for token in tgt_nlp.tokenizer(pair['vi'])]]
        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu(reference, output_words, smoothing_function=chencherry.method1)
        # bleu_score = sentence_bleu(reference, output_words)
        total_bleu += bleu_score

        print('-' * 12)

    avg_bleu = total_bleu / n
    print(f"Average BLEU score: {avg_bleu:.4f}")

def calcBLEU(encoder, decoder, test_data, src_vocab, tgt_vocab, src_nlp, tgt_nlp, device):
    total_bleu = 0
    
    for pair in test_data:
        output_words = evaluate(encoder, decoder, pair['en'], src_vocab, tgt_vocab, src_nlp, device)
        reference = [[token.text for token in tgt_nlp.tokenizer(pair['vi'])]]
        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu(reference, output_words, smoothing_function=chencherry.method1)
        total_bleu += bleu_score

    avg_bleu = total_bleu / len(test_data)
    return avg_bleu

def calculate_meteor(reference, hypothesis):
    from nltk.translate.meteor_score import single_meteor_score
    # Chuyển chuỗi thành danh sách các token
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    return single_meteor_score(reference_tokens, hypothesis_tokens)

def calcMetrics(encoder, decoder, test_data, src_vocab, tgt_vocab, src_nlp, tgt_nlp, device):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    total_bleu = 0
    total_rouge = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    total_meteor = 0

    for pair in test_data:
        output_words = translate_sentence(encoder, decoder, [pair['en']], src_vocab, tgt_vocab, src_nlp, device)[0]
        output_sentence = ' '.join(output_words)

        reference = pair['vi']
        reference_tokens = reference.split()  # Tách token
        hypothesis_tokens = output_sentence.split()  # Tách token

        chencherry = SmoothingFunction()
        bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=chencherry.method1)
        total_bleu += bleu_score

        rouge_scores = calculate_rouge(reference, output_sentence)
        for key in total_rouge:
            total_rouge[key] += rouge_scores[key]

        meteor_score = calculate_meteor(reference, output_sentence)  # Không cần tách token thêm nữa
        total_meteor += meteor_score

    avg_bleu = total_bleu / len(test_data)
    avg_rouge = {key: total_rouge[key] / len(test_data) for key in total_rouge}
    avg_meteor = total_meteor / len(test_data)

    return avg_bleu, avg_rouge, avg_meteor