import pandas as pd
import numpy as np
import fasttext
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("Dataset\\TruthfulQA.csv")  

# Load FastText model (Method 1)
ft_model = fasttext.load_model("cc.en.300.bin")  # Path to downloaded FastText .bin file

# Load BERT model (Method 2)
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to compute FastText sentence embedding (300-dim)
def get_fasttext_embedding(sentence):
    words = sentence.lower().split()
    vectors = [ft_model.get_word_vector(w) for w in words if w]
    return np.mean(vectors, axis=0).tolist() if vectors else [0.0] * 300

# Function to compute BERT sentence embedding (384-dim)
def get_bert_embedding(sentence):
    return bert_model.encode(sentence).tolist()

# Apply embeddings
df["gen_method1"] = df["generated_text"].apply(get_fasttext_embedding)
df["act_method1"] = df["actual_text"].apply(get_fasttext_embedding)

df["gen_method2"] = df["generated_text"].apply(get_bert_embedding)
df["act_method2"] = df["actual_text"].apply(get_bert_embedding)

# Select only needed columns
final_df = df[[
    "generated_text",
    "actual_text",
    "gen_method1",
    "act_method1",
    "gen_method2",
    "act_method2"
]]

# Save to CSV (embeddings as strings)
final_df.to_csv("sentence_embeddings_all_methods.csv", index=False)

print("✅ Embeddings from both methods saved successfully.")
