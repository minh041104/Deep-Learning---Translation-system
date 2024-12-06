from transformers import MarianMTModel, MarianTokenizer
from models.text_extractor import extract_text_from_url
import re

def translate_text(url, target_language, num_sentences=1):
    language_map = {
        "English": "en",
        "Spanish": "es",
        "Vietnamese": "vi",
        "German": "de",
        "French": "fr"
    }
    target_code = language_map.get(target_language)
    
    if not target_code:
        raise ValueError(f"Ngôn ngữ đích '{target_language}' không được hỗ trợ.")
    
    original_text = extract_text_from_url(url)
    if not original_text:
        raise ValueError("Không thể trích xuất văn bản từ URL.")
    
    original_txt = ""
    translated_text = ""

    sentences = re.split(r'(?<=[.!?])\s+', original_text.strip())

    def translate_segment(segment_text):
        model_name_to_en = f"Helsinki-NLP/opus-mt-mul-en"
        tokenizer_to_en = MarianTokenizer.from_pretrained(model_name_to_en)
        model_to_en = MarianMTModel.from_pretrained(model_name_to_en)
        
        encoded_text = tokenizer_to_en([segment_text], return_tensors="pt", truncation=True, padding=True)
        translated_to_en = model_to_en.generate(**encoded_text)
        english_text = tokenizer_to_en.decode(translated_to_en[0], skip_special_tokens=True)

        if target_code != "en":
            model_name_to_target = f"Helsinki-NLP/opus-mt-en-{target_code}"
            tokenizer_to_target = MarianTokenizer.from_pretrained(model_name_to_target)
            model_to_target = MarianMTModel.from_pretrained(model_name_to_target)
            
            encoded_text = tokenizer_to_target([english_text], return_tensors="pt", truncation=True, padding=True)
            translated_to_target = model_to_target.generate(**encoded_text)
            final_translation = tokenizer_to_target.decode(translated_to_target[0], skip_special_tokens=True)
        else:
            final_translation = english_text
        
        return final_translation

    for i in range(0, len(sentences), num_sentences):
        segment = '. '.join(sentences[i:i + num_sentences]) 
        original_txt += segment
        translated_text += translate_segment(segment) + " " 
        yield original_txt.strip(), translated_text.strip()
