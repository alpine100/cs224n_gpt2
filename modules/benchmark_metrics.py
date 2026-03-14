import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

class Benchmark:

    benchmarks = []

    '''Throughput'''
    @staticmethod
    def get_flops_forward(B,H,S,Dh,causal=True):
        #2 matmuls * 2BHSSSh
        flops = 4*B*H*S*S*Dh
        return (flops/2) if causal else flops

    @staticmethod
    def get_flops_backward(B,H,S,Dh,causal=True):
        #back is 2.5x forward flops
        return int(Benchmark.get_flops_forward(B,H,S,Dh,causal)*2.5)

    '''Memory bandwidth'''
    @staticmethod
    def get_attention_membw(B, H, S, Dh, dtype, causal=True):
        ebytes = torch.finfo(dtype).bits // 8
        read = 3 * B * H * S * Dh * ebytes #read Q K V
        write = B * H * S * Dh * ebytes #write O
        write += B * H * S * 4 #LSE (fp32)
        return read + write

    @staticmethod
    def get_attention_membw_backward(B, H, S, Dh, dtype):
        ebytes = torch.finfo(dtype).bits // 8 #reads: Q, K, V, O, dO, LSE, Delta
        read = 5 * B * H * S * Dh * ebytes    # Q,K,V,O,dO
        read += 2 * B * H * S * 4             # LSE, Delta (fp32)
        write = 3 * B * H * S * Dh * ebytes   # writes: dQ (atomic), dK, dV
        return read + write

    @staticmethod
    def report_metrics(provider, mode, B, H, S, Dh, dtype, ms):
        """
        Compute and print FLOP/s, memory bandwidth, and tokens/sec
        given a measured latency in milliseconds.
        """
        seconds = ms / 1e3

        # Throughput (FLOPS)
        if mode == "Forward Pass":
            flops = Benchmark.get_flops_forward(B, H, S, Dh, causal=True)
        else:
            flops = Benchmark.get_flops_backward(B, H, S, Dh, causal=True)

        tflops = (flops / seconds) / 1e12  # TFLOP/s

        # Memory bandwidth
        if mode == "Forward Pass":
            mem_bytes = Benchmark.get_attention_membw(B, H, S, Dh, dtype)
        else:
            mem_bytes = Benchmark.get_attention_membw_backward(B, H, S, Dh, dtype)

        membw_tbs = (mem_bytes / seconds) / 1e12  # TB/s
        tokens_per_s = (B * S) / seconds         # tokens per second

        # --- Tokens/sec ---
        # one "token" = one row in the sequence dimension, across all batch/head
        tokens = B * S
        tokens_per_sec = tokens / seconds

        print(
            f"[{mode}] S={S:>6} | "
            f"Latency: {ms:.3f} ms | "
            f"TFLOP/s: {tflops:.2f} | "
            f"BW: {membw_tbs:.3f} TB/s | "
            f"Tokens/s: {tokens_per_sec:,.0f}"
        )
        Benchmark.benchmarks.append({
            "mode":         mode,
            "provider":     provider,
            "S":            S,
            "ms":           ms,
            "tflops":       tflops,
            "membw_tbs":    membw_tbs,
            "tokens_per_s": tokens_per_s,
        })
        return {
            "mode": mode, "S": S, "ms": ms,
            "tflops": tflops, "membw_tbs": membw_tbs,
            "tokens_per_sec": tokens_per_sec,
        }

    @staticmethod
    def get_gpu_peaks():
        """
        Returns (peak_tflops_fp16, peak_membw_tbs) for the current GPU.
        Values are approximate — check your GPU spec sheet for exact numbers.
        """
        name = torch.cuda.get_device_name(0).lower()
        peaks = {
            # (TFLOP/s fp16 tensor core, memory BW TB/s)
            "a100": (312, 2.0),
            "h100": (989, 3.35),
            "v100": (125, 0.9),
            "l4":   (121, 0.3),
            "t4":   (65,  0.32)
        }
        for key, vals in peaks.items():
            if key in name:
                return vals
        print(f"WARNING: Unknown GPU '{name}', returning None")
        return None, None
    
    @staticmethod
    def plot_benchmark_results(save_path="../output/plots", show=True, tag=None):
        os.makedirs(save_path, exist_ok=True)
        df = pd.DataFrame(Benchmark.benchmarks)

        modes     = df["mode"].unique()       # ["Forward Pass", "Back Prop"]
        providers = df["provider"].unique()   # ["triton", "sdpa"]
        metrics   = [
            ("ms",           "Latency (ms)",         "Latency vs Sequence Length"),
            ("tflops",       "TFLOP/s",              "Compute Throughput vs Sequence Length"),
            ("membw_tbs",    "Memory Bandwidth (TB/s)", "Memory Bandwidth vs Sequence Length"),
            ("tokens_per_s", "Tokens / Second",       "Tokens per Second vs Sequence Length"),
        ]

        colors    = {"triton": "#4C72B0", "sdpa": "#DD8452"}
        linestyle = {"triton": "-",       "sdpa": "--"}
        marker    = {"triton": "o",       "sdpa": "s"}

        # one figure per mode, subplots = one per metric
        for mode in modes:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"Flash Attention Benchmark — {mode}", fontsize=15, fontweight="bold")
            axes = axes.flatten()

            df_mode = df[df["mode"] == mode]

            for ax, (col, ylabel, title) in zip(axes, metrics):
                for prov in providers:
                    df_prov = df_mode[df_mode["provider"] == prov].sort_values("S")
                    ax.plot(
                        df_prov["S"],
                        df_prov[col],
                        label=prov,
                        color=colors[prov],
                        linestyle=linestyle[prov],
                        marker=marker[prov],
                        linewidth=2,
                        markersize=5,
                    )

                ax.set_title(title, fontsize=11)
                ax.set_xlabel("Sequence Length", fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_xscale("log", base=2)
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                ax.tick_params(axis="x", rotation=30)
                ax.legend(fontsize=9)
                ax.grid(True, which="both", linestyle=":", alpha=0.6)

                # add peak BW / FLOP reference lines where relevant
                peak_tflops, peak_bw = Benchmark.get_gpu_peaks()
                if col == "tflops" and peak_tflops:
                    ax.axhline(peak_tflops, color="red", linestyle=":", linewidth=1.2,
                            label=f"HW peak ({peak_tflops} TFLOP/s)")
                    ax.legend(fontsize=9)
                if col == "membw_tbs" and peak_bw:
                    ax.axhline(peak_bw, color="red", linestyle=":", linewidth=1.2,
                            label=f"HW peak ({peak_bw} TB/s)")
                    ax.legend(fontsize=9)

            plt.tight_layout()
            tag_str = f"{tag}_" if tag else ""
            fname = os.path.join(save_path, f"{tag_str}Detailed benchmark_{mode.lower().replace(' ', '_')}.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved: {fname}")
            if show:
                plt.show()
            plt.close(fig)
