# Dataset description
1.

# File Description
1. The distribution_baseline_drug_eg.py file is an example in the baseline model. Using the one-tail test in ttest, the proportion of drug to five categories of tokens is calculated.
2. The p_value_baseline_drug_eg.py file is an example in the baseline model. Using the one-tail test in ttest on the head dimension, the p-value of drug and effect for other tokens (I marked it as o_attention) is calculated. (ps: If you only need to calculate on each head of the ninth layer, you only need to use the attention value of the ninth layer when accumulating attention)
