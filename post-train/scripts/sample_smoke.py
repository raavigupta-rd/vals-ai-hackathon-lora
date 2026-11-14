"""Smoke test: sample from a Tinker base model (from the Tinker quickstart)."""

import asyncio

import tinker

BASE_MODEL = "Qwen/Qwen3.5-9B"


async def main():
    service_client = tinker.ServiceClient(display_name="smoke-sample")
    sampling_client = await service_client.create_sampling_client(base_model=BASE_MODEL)
    tokenizer = sampling_client.get_tokenizer()
    params = tinker.types.SamplingParams(max_tokens=50, temperature=0.7)

    # Plain-text continuation
    model_input = tinker.types.ModelInput.from_ints(tokenizer.encode("The capital of France is"))
    result = await sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=params)
    print("Plain:", tokenizer.decode(result.sequences[0].tokens))

    # Chat-formatted prompt
    conversation_tokens = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    model_input = tinker.types.ModelInput.from_ints(conversation_tokens)
    result = await sampling_client.sample(prompt=model_input, num_samples=1, sampling_params=params)
    print("Chat: ", tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True))


if __name__ == "__main__":
    asyncio.run(main())
