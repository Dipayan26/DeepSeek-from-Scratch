# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
# model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")





import torch
from transformers import AutoTokenizer
from typing import List, Tuple

def load_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Loads a FASTA file.
    Returns a list of tuples: (header, sequence).
    """
    sequences = []
    header = None
    seq_chunks = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):  # header line
                if header is not None:
                    sequences.append((header, "".join(seq_chunks)))
                header = line[1:]  # remove ">"
                seq_chunks = []
            else:
                seq_chunks.append(line)

        # append last entry
        if header is not None:
            sequences.append((header, "".join(seq_chunks)))

    return sequences


def filter_by_length(sequences, max_len=1500):
    """
    Filters sequences based on allowed maximum length.
    """
    return [(h, s) for h, s in sequences if len(s) <= max_len]



def tokenize_sequences(sequences, tokenizer):
    """
    Tokenizes sequences using ESM tokenizer.
    
    Returns:
        input_ids:  LongTensor [batch, seq_len]
        attention_mask: LongTensor [batch, seq_len]
    """
    seq_only = [s for _, s in sequences]  # extract sequences

    # ESM tokenizer automatically:
    # - adds BOS/EOS tokens
    # - converts amino acids into integer IDs
    encoded = tokenizer(
        seq_only,
        return_tensors="pt",
        padding=True,     # pad to max seq in batch
        truncation=False  # we manually filtered lengths
    )

    return encoded["input_ids"], encoded["attention_mask"]



# ================================
#        MAIN PIPELINE
# ================================

fasta_path = r"C:\Users\dipay\Documents\Deep_seek_from_scratch\DeepSeek-from-Scratch\Protein_data\human_swissprot_oneliner.fasta"

# 1. Load FASTA
all_sequences = load_fasta(fasta_path)
len(all_sequences)

# 2. Filter for seq ≤ 1500 aa
filtered_sequences = filter_by_length(all_sequences, max_len=1500)
len(filtered_sequences)
# 3. Load ESM-2 tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/esm2_t33_650M_UR50D"
)


# 4. Tokenize
input_ids, attention_mask = tokenize_sequences(filtered_sequences[:100], tokenizer)


print("Input IDs shape:", input_ids.shape)
print("Attention mask shape:", attention_mask.shape)

tokenizer.vocab_size
# >>> tokenizer.vocab_size
# 33



