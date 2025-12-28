from transformers import AutoTokenizer

USE_VLLM = False

if USE_VLLM:
    from vllm import LLM, SamplingParams
else:
    from nanovllm import LLM, SamplingParams

def main():
    path = "public/models/Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True) # default tensor_parallel_size=1
    # llm = LLM(path, enforce_eager=True, tensor_parallel_size=8)

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=256)
    prompts = [
        "introduce yourself",
        # "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        if USE_VLLM:
            print(f"Completion: {output.outputs[0].text!r}")
        else:
            print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
