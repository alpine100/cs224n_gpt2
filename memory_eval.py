import torch
import gc
import matplotlib.pyplot as plt
from config import GPT2Config
from models.gpt2 import GPT2Model

def profile_model_memory(model, seq_lengths, batch_size=1):
    model.eval()
    model.cuda()
    memory_usage_mb = []
    
    for length in seq_lengths:
        try:
            # 1. Create dummy input_ids and attention_mask for this sequence length
            # input_ids: [batch_size, seq_len], vocab_size is usually 50257 for GPT2
            dummy_ids = torch.randint(0, 50000, (batch_size, length)).cuda()
            
            # attention_mask: [batch_size, seq_len] (1 for real tokens, 0 for pad)
            dummy_mask = torch.ones(batch_size, length).cuda()
            
            # 2. Reset PyTorch's memory tracker to get a clean measurement
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            
            # 3. Forward pass (no gradients needed for inference memory)
            with torch.no_grad():
                _ = model(dummy_ids, dummy_mask)
                
            # 4. Grab the peak memory allocated during the forward pass
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mb = peak_bytes / (1024 ** 2) # Convert to Megabytes
            memory_usage_mb.append(peak_mb)
            
            print(f"Seq Length {length: <5} | Peak Memory: {peak_mb:.2f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Seq Length {length: <5} | OOM (Out of Memory)")
                memory_usage_mb.append(None) # Mark as OOM
                torch.cuda.empty_cache() 
                gc.collect()
            else:
                raise e
                
    return memory_usage_mb

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is required to profile GPU memory!")
        exit()

    print("Initializing Custom GPT2 Model...")
    # Initialize your custom model using your config
    config = GPT2Config(
        hidden_size=768, 
        num_hidden_layers=12, 
        num_attention_heads=12, 
        intermediate_size=768*3
    )
    model = GPT2Model(config)
    
    # Test sequence lengths from 128 tokens up to 4096 tokens
    seq_lengths = [128, 256, 512, 1024, 1536, 2048, 3072, 4096]
    
    print("Profiling Memory Usage...")
    memory_results = profile_model_memory(model, seq_lengths)
    
    # --- Plotting the Results ---
    # Filter out OOM (None) values for plotting
    valid_lengths = [l for l, m in zip(seq_lengths, memory_results) if m is not None]
    valid_memory = [m for m in memory_results if m is not None]

    plt.figure(figsize=(8, 5))
    plt.plot(valid_lengths, valid_memory, marker='o', linestyle='-', color='b', linewidth=2)
    
    plt.title('Custom GPT-2 Attention Memory Footprint', fontsize=14)
    plt.xlabel('Sequence Length (Tokens)', fontsize=12)
    plt.ylabel('Peak GPU Memory (MB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(seq_lengths, rotation=45)
    
    plt.tight_layout()
    plt.savefig('custom_attention_memory.png', dpi=300)
    print("Plot saved to 'custom_attention_memory.png'")