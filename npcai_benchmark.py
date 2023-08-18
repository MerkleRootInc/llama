import time, psutil, fire, subprocess

from typing import Optional
from llama import Llama

def main(
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = 512,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    input_sequences = [
        "What's 2+2?"
    ]

    print(f"Benchmarking inference for {len(input_sequences)} input sequences...\n")

    # todo:
    # is there a way to force the number of outputted tokens?
    # get instructions together for running

    for sequence in input_sequences:
        tokens = [ generator.tokenizer.encode(sequence, False, False) ]
        start_time = time.time()
        gpu_percent = "N/A"
        vram_usage = "N/A"

        result = generator.generate(
            tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        end_time = time.time()
        cpu_percent = psutil.cpu_percent(interval=None)
        if has_nvidia_gpu():
            gpu_percent = get_gpu_utilization()
            vram_usage = get_vram_usage()

        gen_tokens = result[0][0]
        gen_text = generator.tokenizer.decode(gen_tokens)
        gen_speed = len(gen_tokens) / (end_time - start_time)

        print()
        print(f"Input sequence length: {len(tokens[0])}")
        print(f"Total inference time (sec.): {round(end_time - start_time, 2)}")
        print(f"Tokens generated: {len(gen_tokens)}")
        print(f"Inference-adjusted rate (tokens/sec.): {round(gen_speed, 2)}")
        print(f"CPU utilization (%): {cpu_percent}")
        print(f"GPU utilization (%): {gpu_percent}")
        print(f"vRAM usage (MB): {vram_usage}")

        print()
        print("Generated Text:")
        print(gen_text)

        print()
        print("=" * 40)

def has_nvidia_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if "NVIDIA-SMI" in result.stdout:
            return True
        return False
    except Exception as e:
        return False

def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print("Error:", e)
        return "N/A"

def get_vram_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        return int(result.stdout.strip())
    except Exception as e:
        print("Error:", e)
        return "N/A"

if __name__ == "__main__":
    fire.Fire(main)