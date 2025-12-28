import os
from jetengine import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("./public/models/LLaDA/LLaDA-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm = LLM(
        path, 
        enforce_eager=False, 
        tensor_parallel_size=1, 
        mask_token_id=126336, 
        block_length=1024, 
        gpu_memory_utilization=0.9
        )
    sampling_params = SamplingParams(
        temperature=1.0, 
        topk=0, 
        topp=1.0, 
        max_tokens=2048,
        remasking_strategy="low_confidence_dynamic", 
        block_length=1024, 
        denoising_steps=1024, 
        dynamic_threshold=0.90
        )

    questions = [
        "Consider the geometric sequence frac{125}{9}, frac{25}{3}, 5, 3, ... What is the eighth term of the sequence? Express your answer as a common fraction.",
        "A regular pentagon is rotated counterclockwise about its center. What is the minimum number of degrees it must be rotated until it coincides with its original position?",
        "If a snack-size tin of peaches has 40 calories and is 2% of a person's daily caloric requirement, how many calories fulfill a person's daily caloric requirement?",
    ]

    prompts_list = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt + "\n Solve the problem step by step.\n"}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in questions
    ]
    
    prompts_list = prompts_list * 1000 
    
    print(f"Total requests: {len(prompts_list)}")
    
    outputs = llm.generate_streaming(
        prompts_list, sampling_params, max_active=512)

    for output in outputs:
        print("\n")
        print(f"Completion: {output['text']!r}")
        print(f"Total Length: {len(output['token_ids'])!r}")
        print(f"Throughput: {output['Throughput']!r} tokens/sec")


if __name__ == "__main__":
    main()
