import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import ast

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

num_layers = 12
num_heads = 12


def find_token_indices(input_ids, token_ids):
    start_index = None
    for i in range(len(input_ids) - len(token_ids) + 1):
        if input_ids[i:i + len(token_ids)] == token_ids:
            start_index = i
            break
    return list(range(start_index, start_index + len(token_ids))) if start_index is not None else []


def get_o_indices(ner_tags, max_length):
    return [idx for idx, tag in enumerate(ner_tags) if tag == "O" and idx < max_length]


def aggregate_attention_scores(attentions, token_indices):
    num_layers = len(attentions)
    num_heads = attentions[0].size(1)
    attention_scores = np.zeros((num_layers, num_heads))

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            attention_matrix = attentions[layer_idx][0][head_idx].cpu().numpy()
            valid_indices = [idx for idx in token_indices if idx < attention_matrix.shape[1]]
            if not valid_indices:
                continue
            token_attention = attention_matrix[:, valid_indices].mean(axis=1)
            attention_scores[layer_idx, head_idx] += token_attention.mean()

    return attention_scores


model_name = "michiyasunaga/BioLinkBERT-base"

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=5, id2label=id2label, label2id=label2id,
                                                        output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
dataset_path = './models/biolink/test_label.csv'
dataset = pd.read_csv(dataset_path)
drug_attention_agg = np.zeros((num_layers, num_heads))
effect_attention_agg = np.zeros((num_layers, num_heads))
o_attention_agg = np.zeros((num_layers, num_heads))

for index, sample in dataset.iterrows():

    inputs = tokenizer(sample['text'], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions

    input_ids = inputs["input_ids"].squeeze().tolist()

    drug_indices = find_token_indices(input_ids, tokenizer.encode(sample['drug'], add_special_tokens=False))
    effect_indices = find_token_indices(input_ids, tokenizer.encode(sample['effect'], add_special_tokens=False))

    o_indices = get_o_indices(sample['ner_tags'], 80)

    if drug_indices:
        drug_attention_agg += aggregate_attention_scores(attentions, drug_indices)
    if effect_indices:
        effect_attention_agg += aggregate_attention_scores(attentions, effect_indices)
    if o_indices:
        o_attention_agg += aggregate_attention_scores(attentions, o_indices)

drug_scores_array = np.array(drug_attention_agg)
effect_scores_array = np.array(effect_attention_agg)
o_scores_array = np.array(o_attention_agg)

# Compute the mean difference between the two samples
mean_diff_drug_vs_o = np.mean(drug_scores_array) - np.mean(o_scores_array)
mean_diff_effect_vs_o = np.mean(effect_scores_array) - np.mean(o_scores_array)

# Perform the two-tailed t-test
_, p_value_drug_vs_o = ttest_ind(drug_scores_array, o_scores_array, nan_policy='omit')
_, p_value_effect_vs_o = ttest_ind(effect_scores_array, o_scores_array, nan_policy='omit')

# Convert to one-tailed p-values
if mean_diff_drug_vs_o > 0:
    p_value_drug_vs_o = p_value_drug_vs_o / 2
else:
    p_value_drug_vs_o = 1 - p_value_drug_vs_o / 2

if mean_diff_effect_vs_o > 0:
    p_value_effect_vs_o = p_value_effect_vs_o / 2
else:
    p_value_effect_vs_o = 1 - p_value_effect_vs_o / 2

# Output the one-tailed p-values
print(f"One-tailed P-value for drug vs O: {p_value_drug_vs_o}")
print(f"One-tailed P-value for effect vs O: {p_value_effect_vs_o}")