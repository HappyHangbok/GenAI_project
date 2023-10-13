import os
import matplotlib.pyplot as plt
import language_tool_python
import nltk
from nltk.translate.bleu_score import sentence_bleu
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import torch
import ssl
import matplotlib
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

matplotlib.use('Agg')
# Disable SSL certificate verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Handles the case where the above method is not available
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# Ensure NLTK data is downloaded
nltk.download('punkt')

# [Define generate_text_markov and generate_text_gpt as per your existing code]
def generate_text_markov(input_text, num_words):
    words = input_text.split()
    model = {}
    for i in range(len(words) - 1):
        if words[i] not in model:
            model[words[i]] = []
        model[words[i]].append(words[i + 1])

    current_word = random.choice(words)
    generated_text = [current_word]
    for _ in range(num_words - 1):
        next_word_candidates = model.get(current_word, [])
        if not next_word_candidates:
            break
        next_word = random.choice(next_word_candidates)
        generated_text.append(next_word)
        current_word = next_word

    return ' '.join(generated_text)

def generate_text_gpt(seed_text, num_words):
    # Load pre-trained model and tokenizer from transformers
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Encode input text
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=num_words + len(input_ids[0]),
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,  # Add this line
            attention_mask=torch.ones(input_ids.shape)  # And this line
        )

    # Decode the output and return text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


# Read input text for Markov chain generator from a file
with open('./input.txt', 'r') as file:
    input_text = file.read()

# Tool for grammar check
tool = language_tool_python.LanguageTool('en-US')

# Generate text samples.
num_samples = 100

markov_texts = [generate_text_markov(input_text, 20) for _ in range(num_samples)]
gpt_texts = [generate_text_gpt("Generative AI has", 20) for _ in range(num_samples)]

# Save Generated Texts
for i, (m_text, g_text) in enumerate(zip(markov_texts, gpt_texts)):
    for text, dir_name in zip([m_text, g_text], ['./markov', './gpt']):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(f'{dir_name}/text{i+1}.txt', 'w') as f:
            f.write(text)

# Metric 1: BLEU Score
reference_text = "Generative AI has revolutionized the field of technology, bringing forth advancements in various applications."
reference_tokens = nltk.word_tokenize(reference_text.lower())

smoothie = SmoothingFunction().method4  # Use smoothing method 4 as an example
# Apply smoothing in BLEU calculation
bleu_markov = [
    sentence_bleu(
        [reference_tokens],
        nltk.word_tokenize(text.lower()),
        smoothing_function=smoothie  # Apply smoothing here
    ) for text in markov_texts
]

bleu_gpt = [
    sentence_bleu(
        [reference_tokens],
        nltk.word_tokenize(text.lower()),
        smoothing_function=smoothie  # And here
    ) for text in gpt_texts
]

# Metric 2: Grammar Check
grammar_markov = [len(tool.check(text)) / len(text.split()) for text in markov_texts]
grammar_gpt = [len(tool.check(text)) / len(text.split()) for text in gpt_texts]

# fig, ax = plt.subplots(2, 1, figsize=(10, 12))
#
# # Define distinct colors for boxplots
# colors = ['skyblue', 'lightgreen']
# labels = ['Markov Chain', 'GPT-2']
#
# # BLEU Score Plot
# bleu_data = [bleu_markov, bleu_gpt]
# bp1 = ax[0].boxplot(bleu_data, patch_artist=True)
# ax[0].set_title('Comparison of Text Generation Methods', fontsize=15)
# ax[0].set_ylabel('BLEU Score', fontsize=12)
# ax[0].set_xticks([1, 2])
# ax[0].set_xticklabels(labels)
# ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
#
# # Coloring and adding data points for BLEU
# for patch, color in zip(bp1['boxes'], colors):
#     patch.set_facecolor(color)
# for i, data in enumerate(bleu_data, 1):
#     ax[0].plot([i]*num_samples, data, 'o', color='black', markersize=5)
#
# # Grammar Mistakes Plot
# grammar_data = [grammar_markov, grammar_gpt]
# bp2 = ax[1].boxplot(grammar_data, patch_artist=True)
# ax[1].set_title('Grammar Mistakes in Generated Texts', fontsize=15)
# ax[1].set_ylabel('Ratio of Grammar Mistakes to Words', fontsize=12)
# ax[1].set_xticks([1, 2])
# ax[1].set_xticklabels(labels)
# ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
#
# # Coloring and adding data points for Grammar
# for patch, color in zip(bp2['boxes'], colors):
#     patch.set_facecolor(color)
# for i, data in enumerate(grammar_data, 1):
#     ax[1].plot([i]*num_samples, data, 'o', color='black', markersize=5)
#
# # Adding a legend using the patches from the boxplots
# legend_elements = [plt.Line2D([0], [0], color=colors[0], lw=4, label='Markov Chain'),
#                    plt.Line2D([0], [0], color=colors[1], lw=4, label='GPT-2')]
# ax[0].legend(handles=legend_elements, loc='upper right')
#
# plt.tight_layout()
# plt.savefig("./boxplot.png")

fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Define distinct colors for scatter plots
colors = ['skyblue', 'lightgreen']
labels = ['Markov Chain', 'GPT-2']

# BLEU Score Scatter Plot
for i, data in enumerate([bleu_markov, bleu_gpt]):
    ax[0].scatter(range(1, len(data) + 1), data, color=colors[i], label=labels[i], alpha=0.7)

ax[0].set_title('Comparison of Text Generation Methods', fontsize=15)
ax[0].set_ylabel('BLEU Score', fontsize=12)
ax[0].set_xlabel('Individual Index', fontsize=12)
ax[0].legend()
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Grammar Mistakes Scatter Plot
for i, data in enumerate([grammar_markov, grammar_gpt]):
    ax[1].scatter(range(1, len(data) + 1), data, color=colors[i], label=labels[i], alpha=0.7)

ax[1].set_title('Grammar Mistakes in Generated Texts', fontsize=15)
ax[1].set_ylabel('Ratio of Grammar Mistakes to Words', fontsize=12)
ax[1].set_xlabel('Individual Index', fontsize=12)
ax[1].legend()
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("./scatter_plot.png")

