# Dataset description
1. The test_label.csv file is the test set file used in my experiment

# File Description
1. The distribution_baseline_drug_eg.py file is a percentage summary of the attention distribution of drug and effect tokens in the ninth layer on the biolink baseline model. (ps: I divided tokens into five categories: cls, sep, drug, effect, and other to help me observe which type of token the attention of drug and effect is more inclined to observe in each head).
2. The p_value_baseline_drug_eg.py file is an example in the biolink baseline model. Using the one-tail test in ttest on the head dimension, the p-value of drug and effect for other tokens (I marked it as o_attention) is calculated. (ps: If you only need to calculate on each head of the ninth layer, you only need to use the attention value of the ninth layer when accumulating attention)
