from deep_translator import GoogleTranslator
from datasets import load_dataset, load_from_disk, Dataset
import time
import os
import pandas as pd
def download_data(path_to_save, path_download="bentrevett/multi30k"):
    ds = load_dataset(path_download)
    ds.save_to_disk(path_to_save)
    print("Download successful")

def load_from_disk(path):
    return load_from_disk(path)

# Decrease the number of requests to GoogleTranslate API
def translate_dataset(dataset, src_language ='en', tgt_language='vi',
                      max_length_request = 1000, time_request = 0.2):

    translator = GoogleTranslator(source=src_language, target=tgt_language)
    print("Start translating")
    data_trans = []
    len_data = len(dataset)
    i = 0
    num_request = 1
    sum_sentences = 0
    while i < len_data:
        sum_length = 0
        request_text = ''
        num_sentences = 0
        # Optimize length of request_text. Each sentence have split by enter char
        while i < len_data and sum_length < max_length_request:
            en_sentene = dataset[i] + '\n'
            cur_len = len(en_sentene)
            if sum_length + cur_len < max_length_request:
                sum_length += cur_len
                request_text += en_sentene 
                num_sentences += 1
                i += 1
            else: 
                break
        # Translate request_text and add to data_trans
        result_translated = translator.translate(request_text).split('\n')
        # Check if length of translated sentences is not equal number of sentences in request_text
        if len(result_translated) != num_sentences:
            print("Not equal:",len(result_translated), num_sentences)
            return data_trans
        for result in result_translated:
            data_trans.append(result)
        sum_sentences += num_sentences
        print(num_request, f"{sum_sentences} sentences")
        num_request += 1
        
        # Sleep between 2 request
        time.sleep(time_request)

    # Check if number of sentences translated is not equal number of src sentences
    if sum_sentences != len_data:
        print(sum_sentences, len_data)
    print("Done!")
    return data_trans

def save_data(src_data, tgt_data, path_to_save):
    with open(path_to_save, 'w', encoding='utf-8-sig') as file:
        for src, tgt in zip(src_data, tgt_data):
            line = f"{src}\t{tgt}"
            file.write(line + '\n')
    print("Save successful")

def get_dataset(name, root_path):

    path = os.path.join(root_path, name)
    src_data, tgt_data = None, None
    with open(path, 'r', encoding='utf-8-sig') as f:
        data = f.readlines()
        src_data = [line.split('\t')[0].strip() for line in data]
        tgt_data = [line.split('\t')[1].strip() for line in data]
    dataset = Dataset.from_pandas(pd.DataFrame({'en':src_data, 'vi':tgt_data}))
    return dataset
