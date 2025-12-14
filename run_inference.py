



"""
Inference script for DeepSeek-V3 using ESM-2 tokenization
"""

import torch
from transformers import AutoTokenizer
from models import DeepSeekConfig, DeepSeekV3


def generate_text(prompt, tokenizer, max_tokens=100, temperature=0.8, top_k=50):
    """Generate protein text from a prompt using ESM2 tokens."""
    
    # Build config dynamically from vocab size
    config = DeepSeekConfig(
        vocab_size= 33  #tokenizer.vocab_size,  # IMPORTANT
        block_size=128,
        n_layer=2,
        n_head=4,
        n_embd=256,
        kv_lora_rank=64,
        q_lora_rank=64,
        n_experts=4,
        n_experts_per_token=2,
        mtp_num_heads=1,
        dropout=0.1
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = DeepSeekV3(config)

    try:
        model.load_state_dict(torch.load("best_deepseek_v3.pt", map_location=device))
        print("✓ Loaded trained model")
    except FileNotFoundError:
        print("⚠️ No trained model found, using random weights")

    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def format_params(n):
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        else:
            return str(n)

    print(f"Total parameters: {total_params:,} ({format_params(total_params)})")
    print(f"Trainable parameters: {trainable_params:,} ({format_params(trainable_params)})")

    # ---------------------------
    # TOKENIZE WITH ESM-2
    # ---------------------------

    encoded = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt"
    )

    context = encoded["input_ids"].to(device)  # shape [1, seq_len]

    # ---------------------------
    # GENERATE
    # ---------------------------
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

    # Decode back into protein characters
    raw = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    result = raw.replace(" ", "") 

    return result


if __name__ == "__main__":

    print("=" * 60)
    print("DEEPSEEK-V3 PROTEIN GENERATION (ESM-2 TOKENIZER)")
    print("=" * 60)

    # ----------------------------------
    # Load ESM-2 tokenizer (Recommended)
    # ----------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t33_650M_UR50D"
    )
    print(f"✓ Loaded ESM2 tokenizer (vocab size = {tokenizer.vocab_size})")

    # Protein prompts
    my_prompts = [
        "MVLRRLLAALLHSPQLVERL",
        "MSYTLDSLGNPSAYRRVTETRSSFSRVSGSP",
        "MPLSLGAEMWGPEAWLLLLLLLASFTGR",
    ]

    max_tokens = 80
    temperature = 0.8
    top_k = 20   # ESM vocab is small → lower top_k is enough

    for i, prompt in enumerate(my_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        print("-" * 40)

        result = generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

        print("Generated:", result)
        print()

    print("=" * 60)
    print("DONE! Edit prompts and re-run!")
    print("=" * 60)





'''Simple script to run inference with DeepSeek-V3
Change the prompts below and run this file to generate text!

'''

# import torch
# import tiktoken
# from models import DeepSeekConfig, DeepSeekV3

# def generate_text(prompt, max_tokens=100, temperature=0.8, top_k=50):
#     """Generate text from your prompt."""
    
#     # Model configuration (same as training)
#     # config = DeepSeekConfig(
#     #     vocab_size=50257,
#     #     block_size=128,
#     #     n_layer=4,
#     #     n_head=4,
#     #     n_embd=256,
#     #     kv_lora_rank=64,
#     #     q_lora_rank=96,
#     #     n_experts=4,
#     #     n_experts_per_token=2,
#     #     mtp_num_heads=1,
#     #     dropout=0.1
#     # )
#     config = DeepSeekConfig(
#         vocab_size=50257,
#         block_size=128, #1024
#         n_layer=2,#8 # fewer layers
#         n_head=4, #8 # small number of heads
#         n_embd=256,#512 # smaller embedding size
#         kv_lora_rank=64, #128 # reduced LoRA rank
#         q_lora_rank=64, #128
#         n_experts=4, # fewer experts
#         n_experts_per_token=2,
#         mtp_num_heads=1,
#         dropout=0.1
#     )
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Load model
#     model = DeepSeekV3(config)
#     try:
#         model.load_state_dict(torch.load("best_deepseek_v3.pt", map_location=device))
#         print("✓ Loaded trained model")
#     except FileNotFoundError:
#         print("⚠️  No trained model found, using random weights")
    
#     model = model.to(device)
#     model.eval()
    
#     # Tokenize input
#     tokenizer = tiktoken.get_encoding("gpt2")
#     context = torch.tensor(tokenizer.encode_ordinary(prompt)).unsqueeze(0).to(device)
    
#     # Generate
#     with torch.no_grad():
#         generated = model.generate(context, max_tokens, temperature, top_k)
    
#     # Convert back to text
#     result = tokenizer.decode(generated.squeeze().tolist())
#     return result


# if __name__ == "__main__":
#     print("=" * 60)
#     print("DEEPSEEK-V3 TEXT GENERATION")
#     print("=" * 60)
    
#     # ============================================
#     # CHANGE THESE PROMPTS TO WHATEVER YOU WANT!
#     # ============================================
    
#     # my_prompts = [
#     #     "Once upon a time",
#     #     "The little girl found a magic",
#     #     "In the future, artificial intelligence will",
#     #     "The secret to happiness is",
#     #     "Yesterday I went to the store and",
#     # ]
    
    
#     my_prompts = [
#         "MVLRRLLAALLHSPQLVERL",
#         "MSYTLDSLGNPSAYRRVTETRSSFSRVSGSP",
#         "MPLSLGAEMWGPEAWLLLLLLLASFTGR",
#     ]
    
#     # ============================================
#     # CHANGE THESE PARAMETERS TO EXPERIMENT!
#     # ============================================
    
#     max_tokens = 80        # How many words to generate
#     temperature = 0.8      # 0.1=boring, 0.8=balanced, 1.2=crazy
#     top_k = 50            # Vocabulary limit
    
#     # Generate text for each prompt
#     for i, prompt in enumerate(my_prompts, 1):
#         print(f"\n{i}. Prompt: '{prompt}'")
#         print("-" * 40)
        
#         result = generate_text(
#             prompt=prompt,
#             max_tokens=max_tokens,
#             temperature=temperature, 
#             top_k=top_k
#         )
        
#         print(f"Generated: {result}")
#         print()
    
#     print("=" * 60)
#     print("DONE! Edit the prompts above and run again!")
#     print("=" * 60)
    
