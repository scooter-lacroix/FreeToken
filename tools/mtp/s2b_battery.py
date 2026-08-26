"""S2b verification battery: dual-GPU (split=56) serving evidence.

Measures: TTFT + tok/s for short prompts (streaming), a ~3k-token prefill
TTFT (the 23k harness prompt exceeds the 4096 test ctx cap by mandate),
and coherence across three knowledge/reasoning prompts. Prints one summary
block per test for the bring-up log.
"""
import json
import time
import urllib.request

URL = "http://127.0.0.1:1919/v1/chat/completions"
MODEL = "Qwen3.8-27B-Fable-Distill-Q6_K.gguf"


def stream(prompt: str, max_tokens: int, temperature: float = 0.7):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(delta["content"])
                ntok += 1
    total = time.perf_counter() - t0
    return ttft, ntok, total, "".join(text)


def main():
    tests = [
        ("coherence-science", "Explain why the sky is blue at noon but red at sunset. Three sentences.", 120),
        ("coherence-reasoning", "A farmer has 17 sheep. All but 9 run away. How many are left? Answer with the number and one sentence of reasoning.", 80),
        ("coherence-code", "Write a Python function that returns the n-th Fibonacci number using memoization. Code only.", 150),
    ]
    print("=" * 72)
    for name, prompt, mt in tests:
        ttft, ntok, total, text = stream(prompt, mt)
        tps = ntok / (total - ttft) if total > ttft else 0.0
        print(f"[{name}] TTFT {ttft:.2f}s | {ntok} tok in {total:.2f}s | {tps:.1f} tok/s")
        print(f"  output: {text[:280].replace(chr(10), ' | ')}")
    # ~3k-token prefill TTFT
    long_prompt = "You are given a long technical document to hold in context.\n" + (
        "The QuickSilver rendering pipeline processes frame graphs through six stages: "
        "ingest, validation, scheduling, rasterization, compositing, and presentation. "
        "Each stage owns a queue and a worker pool sized by heuristics from the prior frame. "
    ) * 160
    ttft, ntok, total, text = stream(
        long_prompt + "\n\nIn ONE sentence, name the six stages listed above.", 60
    )
    approx_in = len(long_prompt) // 4
    print(f"[prefill-~{approx_in}tok] TTFT {ttft:.2f}s | {ntok} tok out | answers: {text[:200]}")
    print("=" * 72)


if __name__ == "__main__":
    main()
