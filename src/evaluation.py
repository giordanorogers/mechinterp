import logging
from src.models import ModelandTokenizer
from src.functional import predict_next_token
from src.probing.prompt import BiAssociationPrefix, prepare_probing_input
from src.probing.utils import get_lm_generated_answer

logger = logging.getLogger(__name__)

def get_connection_on_entity_pair(
    mt: ModelandTokenizer,
    entities: tuple[str, str],
    prefix_generator: BiAssociationPrefix,
    n_valid=6,
    n_none=2,
    enable_reasoning=False,
    return_next_token_probs=False,
    answer_prefix: str = "",
):
    prefix = prefix_generator.get_prefix(n_valid=n_valid, n_none=n_none)
    connection_prompt = prepare_probing_input(
        mt=mt,
        entities=(entities[0], entities[1]),
        prefix=prefix,
        answer_marker=prefix_generator.answer_marker,
        question_marker=prefix_generator.question_marker,
        block_separator=prefix_generator.block_separator,
        is_a_reasoning_model=enable_reasoning,
        answer_prefix=answer_prefix
    )

    answer = get_lm_generated_answer(
        mt=mt,
        prompt=connection_prompt,
        #is_a_reasoning_model=enable_reasoning
    )
    answer = answer.split("\n")[0]

    if return_next_token_probs:
        if enable_reasoning is False:
            return answer, predict_next_token(
                mt=mt, inputs=connection_prompt.prompt, k=15
            )
        else:
            logger.warning(
                "Next token probs are not meaningful for reasoning LMs. Will decode to <think> token always"
            )
            return answer, None

    return answer