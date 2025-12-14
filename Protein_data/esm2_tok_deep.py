import os
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# ============================================================
#              FASTA LOADING FUNCTIONS
# ============================================================

def load_fasta(path):
    """
    Loads a FASTA file and returns list of (header, sequence).
    """
    sequences = []
    header = None
    seq_chunks = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, "".join(seq_chunks)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)

        # last entry
        if header is not None:
            sequences.append((header, "".join(seq_chunks)))

    return sequences


def filter_sequences(sequences, max_len=1500):
    """Keep sequences length ≤ max_len."""
    return [(h, s) for h, s in sequences if len(s) <= max_len]


# ============================================================
#               DATASET PREPARATION (TRAIN/VAL)
# ============================================================

def split_train_val(sequences, train_ratio=0.8):
    """
    Splits into train and validation sets.
    """
    n = len(sequences)
    split = int(n * train_ratio)
    return sequences[:split], sequences[split:]


def tokenize_batch(sequences, tokenizer):
    """
    Tokenizes a list of protein sequences.
    Returns:
       list of token ID lists
       list of lengths
    """
    seq_list = [s for _, s in sequences]
    encoded = tokenizer(
        seq_list,
        return_tensors=None,
        padding=False,
        truncation=False
    )

    # encoded is a dict: list of lists
    input_ids = encoded["input_ids"]

    lengths = [len(x) for x in input_ids]

    return input_ids, lengths


# ============================================================
#               BINARY FILE CREATION
# ============================================================

def write_bin_file(filename, ids_list, lengths):
    """
    Writes a .bin token file like TinyStories.
    """

    total_len = np.sum(lengths, dtype=np.uint64)
    dtype = np.uint16  # vocab < 65536 → safe
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(total_len,))

    idx = 0
    for token_ids in tqdm(ids_list, desc=f"Writing {filename}"):
        t = np.array(token_ids, dtype=np.uint16)
        arr[idx : idx + len(t)] = t
        idx += len(t)

    arr.flush()
    size_mb = os.path.getsize(filename) / (1024 * 1024)

    print(f"✓ {filename}: {total_len:,} tokens ({size_mb:.1f} MB)")
    return filename


# ============================================================
#                  MAIN PIPELINE
# ============================================================

# def prepare_fasta_with_esm(fasta_path="proteins.fasta"):
#     print("=" * 60)
#     print("PREPARING FASTA DATASET WITH ESM2 TOKENIZER")
#     print("=" * 60)

#     if os.path.exists("train.bin") and os.path.exists("validation.bin"):
#         print("✓ Dataset has already been tokenized.")
#         return

#     # -----------------------
#     # Load and filter FASTA
#     # -----------------------
#     print("Loading FASTA...")
#     sequences = load_fasta(fasta_path)

#     print(f"Loaded {len(sequences):,} sequences.")

#     sequences = filter_sequences(sequences, max_len=1500)
#     print(f"Filtered to {len(sequences):,} sequences (≤ 1500 aa).")

#     # -----------------------
#     # Train/Validation split
#     # -----------------------
#     train_set, val_set = split_train_val(sequences[:2000], train_ratio=0.8)

#     print(f"Train: {len(train_set):,} sequences")
#     print(f"Validation: {len(val_set):,} sequences")

#     # -----------------------
#     # Load ESM tokenizer
#     # -----------------------
#     tokenizer = AutoTokenizer.from_pretrained(
#         "facebook/esm2_t33_650M_UR50D"
#     )
#     vocab_size = tokenizer.vocab_size
#     print(f"Tokenizer vocabulary size: {vocab_size}")

#     # -----------------------
#     # Tokenize datasets
#     # -----------------------
#     print("Tokenizing train set...")
#     train_ids, train_lengths = tokenize_batch(train_set, tokenizer)

#     print("Tokenizing validation set...")
#     val_ids, val_lengths = tokenize_batch(val_set, tokenizer)

#     # -----------------------
#     # Write binary files
#     # -----------------------
#     write_bin_file("train_prot.bin", train_ids, train_lengths)
#     write_bin_file("validation_prot.bin", val_ids, val_lengths)

#     print("\nDataset preparation completed!")
#     print("You can now run: python main.py train")




print("=" * 60)
print("PREPARING FASTA DATASET WITH ESM2 TOKENIZER")
print("=" * 60)

if os.path.exists("train.bin") and os.path.exists("validation.bin"):
    print("✓ Dataset has already been tokenized.")


# -----------------------
# Load and filter FASTA
# -----------------------
print("Loading FASTA...")
sequences = load_fasta(r"C:\Users\dipay\Documents\Deep_seek_from_scratch\DeepSeek-from-Scratch\Protein_data\human_swissprot_oneliner.fasta")

print(f"Loaded {len(sequences):,} sequences.")

sequences = filter_sequences(sequences, max_len=1500)
print(f"Filtered to {len(sequences):,} sequences (≤ 1500 aa).")

# -----------------------
# Train/Validation split
# -----------------------
train_set, val_set = split_train_val(sequences, train_ratio=0.8)

print(f"Train: {len(train_set):,} sequences")
print(f"Validation: {len(val_set):,} sequences")

# -----------------------
# Load ESM tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(
    "facebook/esm2_t33_650M_UR50D"
)
vocab_size = tokenizer.vocab_size
print(f"Tokenizer vocabulary size: {vocab_size}")

# -----------------------
# Tokenize datasets
# -----------------------
print("Tokenizing train set...")
train_ids, train_lengths = tokenize_batch(train_set, tokenizer)

print("Tokenizing validation set...")
val_ids, val_lengths = tokenize_batch(val_set, tokenizer)

# -----------------------
# Write binary files
# -----------------------
write_bin_file("train_prot.bin", train_ids, train_lengths)
write_bin_file("validation_prot.bin", val_ids, val_lengths)

print("\nDataset preparation completed!")
print("You can now run: python main.py train")



# prepare_fasta_with_esm(r"C:\Users\dipay\Documents\Deep_seek_from_scratch\DeepSeek-from-Scratch\Protein_data\human_swissprot_oneliner.fasta")
# Run script
# if __name__ == "__main__":
#     prepare_fasta_with_esm(r"C:\Users\dipay\Documents\Deep_seek_from_scratch\DeepSeek-from-Scratch\Protein_data\human_swissprot_oneliner.fasta")
