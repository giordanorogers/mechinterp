from src.models import ModelandTokenizer
from src.probing.prompt import ProbingPrompt
from src.utils.typing import TokenizerOutput

def get_lm_generated_answer(
    mt: ModelandTokenizer,
    prompt: ProbingPrompt,
    block_separator: str = "\n#",
    use_kv_cache: bool = True,
    max_new_tokens: int = 30,
):
    with mt.generate(
        TokenizerOutput(
            data=dict(
                input_ids=prompt.tokenized["input_ids"],
                attention_mask=prompt.tokenized["attention_mask"]
            )
        ),
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=use_kv_cache,
    ) as gen_trace:
        output = mt.generator.output.save()

    generation = mt.tokenizer.decode(
        output.sequences[0][prompt.tokenized["input_ids"].shape[-1] :],
        skip_special_tokens=False,
    ).strip()

    if block_separator in generation:
        generation = generation.split(block_separator)[0].strip()
    answer = generation

    return answer