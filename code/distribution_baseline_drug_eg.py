from transformers import AutoModelForTokenClassification
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import pandas as pd
import numpy as np


id2label = {
    0: "O",
    1: "B-DRUG",
    2: "I-DRUG",
    3: "B-EFFECT",
    4: "I-EFFECT",

}
label2id = {
    "O": 0, "B-DRUG": 1, "I-DRUG": 2, "B-EFFECT": 3, "I-EFFECT": 4
}



def find_token_indices(input_ids, token_ids):
    start_index = None
    for i in range(len(input_ids) - len(token_ids) + 1):
        if input_ids[i:i+len(token_ids)] == token_ids:
            start_index = i
            break
    return list(range(start_index, start_index+len(token_ids))) if start_index is not None else []

def classify_tokens(input_ids, drug_indices, effect_indices):
    token_types = ['other'] * len(input_ids)
    cls_index = input_ids.index(tokenizer.cls_token_id)
    sep_indices = [i for i, token_id in enumerate(input_ids) if token_id == tokenizer.sep_token_id]
    for idx in drug_indices:
        token_types[idx] = 'drug'
    for idx in effect_indices:
        token_types[idx] = 'effect'
    token_types[cls_index] = 'cls'
    for idx in sep_indices:
        token_types[idx] = 'sep'
    return token_types


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForTokenClassification.from_pretrained("michiyasunaga/BioLinkBERT-base", num_labels=5,
                                                        id2label=id2label, label2id=label2id).to(device)
tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
model.to(device)
model.eval()
dataset_path = '../data/test_label.csv'
dataset = pd.read_csv(dataset_path)
layer_index = 9
attention_results = {head: {'drug': {'cls': 0, 'sep': 0, 'drug': 0, 'effect': 0, 'other': 0},
                            'effect': {'cls': 0, 'sep': 0, 'drug': 0, 'effect': 0, 'other': 0}}
                     for head in range(model.config.num_attention_heads)}


for index, sample in dataset.iterrows():
    inputs = tokenizer(sample['text'], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions

    input_ids = inputs["input_ids"].squeeze().tolist()
    drug_indices = find_token_indices(input_ids, tokenizer.encode(sample['drug'], add_special_tokens=False))
    effect_indices = find_token_indices(input_ids, tokenizer.encode(sample['effect'], add_special_tokens=False))
    token_types = classify_tokens(input_ids, drug_indices, effect_indices)


    for head_idx in range(model.config.num_attention_heads):
        attention = attentions[layer_index][0, head_idx].cpu().numpy()
        for token_idx, token_type in enumerate(token_types):
            if token_type in ['drug', 'effect']:
                focus_index = np.argmax(attention[token_idx])
                focus_type = token_types[focus_index]
                attention_results[head_idx][token_type][focus_type] += 1


for head, data in attention_results.items():
    print(f"Layer 9 Head {head}:")
    for token_type, counts in data.items():
        total = sum(counts.values())
        if total > 0:
            percentages = {k: v / total * 100 for k, v in counts.items()}
            print(f"  {token_type.capitalize()} token attention: {percentages}")
