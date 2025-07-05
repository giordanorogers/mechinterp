# type: ignore
import logging
import torch
from tqdm import tqdm
import numpy as np
from typing import Literal, Union, Optional
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from src.utils.typing import PredictedToken, PathLike, TokenizerOutput
from src.models import ModelandTokenizer
from src.functional import interpret_logits, get_module_nnsight, get_all_module_states
from src.tokens import align_patching_positions

logger = logging.getLogger(__name__)

@torch.inference_mode()
def patched_run(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    states: dict[tuple[str, int], torch.Tensor],
) -> torch.Tensor:
    # Enter a tracing context
    with mt.trace(inputs, scan=False) as trace:
        for location in states:
            layer_name, token_idx = location
            module = get_module_nnsight(mt, layer_name)
            current_states = (
                module.output if ("mlp" in layer_name) else module.output[0]
            )
            current_states[0, token_idx, :] = states[location]
        logits = mt.output.logits[0][-1].save()
    return logits

def get_window(layer_name_format, idx, window_size, n_layer):
    return [
        layer_name_format.format(i)
        for i in range(
            max(0, idx - window_size // 2), min(n_layer - 1, idx + window_size // 2) + 1
        )
    ]

def get_score(
    logits: torch.Tensor,
    token_id: int | list[int],
    metric: Literal["logit", "prob", "log_norm", "log_rank_inv"] = "logit",
    return_individual_scores: bool = False,
    k: int | None = 20,
) -> Union[float, torch.Tensor]:
    token_id = [token_id] if isinstance(token_id, int) else token_id
    logits = logits.squeeze()
    logits = logits.softmax(dim=-1) if metric == "prob" else logits
    if metric == "log_norm":
        assert k is not None, "k must be provided for log_norm"
        denom = logits.topk(k=k, dim=-1).values.mean(dim=-1)
        # logger.debug(f"{logits.shape} | {logits[token_id]=} | {denom=}")
        # logits = logits / denom #! ratio of logits is a weird metric
        logits = logits - denom  #! difference probably makes more sense (?)
    elif metric == "log_rank_inv":
        assert k is not None, "k must be provided for log_rank_inv"
        rank = logits.argsort(dim=-1, descending=True) + 1
        inv_reward = [rank_reward(rank[t], k=k) for t in token_id]
        inv_reward = sum(inv_reward) / len(inv_reward)
        return inv_reward
    score = logits[token_id].mean().item()
    if not return_individual_scores:
        return score
    individual_scores = {t: logits[t].item() for t in token_id}
    return score, individual_scores

@torch.inference_mode()
def calculate_indirect_effects(
    mt: ModelandTokenizer,
    locations: list[tuple[int, int]],
    clean_input: TokenizerOutput,
    patch_states: dict[
        tuple[str, int], torch.Tensor
    ], # expects the states to be in clean_states
    patch_ans_t: int,
    layer_name_format: str,
    window_size: int = 1,
    metric: Literal["logit", "prob", "log_norm"] = "prob",
) -> dict[tuple[str, int], float]:
    indirect_effects = {loc: -1 for loc in locations}
    for loc in tqdm(locations):
        layer_names = get_window(layer_name_format, loc[0], window_size, mt.n_layer)
        token_idx = loc[1]
        states = {(l, token_idx): patch_states[(l, token_idx)] for l in layer_names}
        affected_logits = patched_run(
            mt=mt,
            inputs=clean_input,
            states=states
        )
        indirect_effects[loc] = get_score(
            logits=affected_logits,
            token_id=patch_ans_t,
            metric=metric,
            return_individual_scores=False
        )
    return indirect_effects

@dataclass
class CausalTracingResult(DataClassJsonMixin):
    patch_input_toks: list[str]
    corrupt_input_toks: list[str]
    trace_start_idx: int
    answer: list[PredictedToken]
    low_score: float
    base_score: float
    indirect_effects: torch.Tensor
    subj_range: Optional[tuple[int, int]]
    normalized: bool
    kind: Literal["residual", "mlp", "attention"] = "residual"
    window: int = 1
    metric: Literal["logit", "prob"] = "prob"

    def from_npz(file: np.lib.npyio.NpzFile | PathLike):
        if isinstance(file, PathLike):
            file = np.load(file, allow_pickle=True)

        return CausalTracingResult(
            patch_input_toks=file["patch_input_toks"].tolist(),
            corrupt_input_toks=file["corrupt_input_toks"].tolist(),
            trace_start_idx=file["trace_start_idx"].item(),
            answer=file["answer"].tolist(),
            subj_range=file["subj_range"].tolist(),
            low_score=file["low_score"].item(),
            base_score=file["base_score"].item(),
            indirect_effects=torch.tensor(file["indirect_effects"]),
            normalized=file["normalized"].item(),
            kind=file["kind"].item(),
            window=file["window"].item(),
            metric=file["metric"].item(),
        )
    
@torch.inference_mode()
def trace_important_states(
    mt: ModelandTokenizer,
    prompt_template: str,
    clean_subj: str,
    patched_subj: str,
    clean_input: Optional[TokenizerOutput] = None,
    patched_input: Optional[TokenizerOutput] = None,
    kind: Literal["residual", "mlp", "attention"] = "residual",
    window_size: int = 1,
    normalize=True,
    trace_start_marker: Optional[str] = None,
    metric: Literal["logit", "prob", "log_norm"] = "prob",
    ans_tokens: Optional[list[int] | int] = None,
) -> CausalTracingResult:
    aligned = align_patching_positions(
        mt=mt,
        prompt_template=prompt_template,
        clean_subj=clean_subj,
        patched_subj=patched_subj,
        clean_input=clean_input,
        patched_input=patched_input,
        trace_start_marker=trace_start_marker,
    )

    clean_input = aligned["clean_input"]
    #print(f"{clean_input=}")
    #print(f"{clean_input["input_ids"][0]=}")
    print(f"Decoded clean input: {mt.tokenizer.decode(clean_input['input_ids'][0])}")
    patched_input = aligned["patched_input"]
    print(f"Decoded patched input: {mt.tokenizer.decode(patched_input['input_ids'][0])}")
    subj_range = aligned["subj_range"]
    trace_start_idx = aligned["trace_start_idx"]

    print(f"===> {trace_start_idx=}")

    if trace_start_marker is None:
        trace_start_idx = 0
        if (
            clean_input.input_ids[0][0]
            == patched_input.input_ids[0][0]
            == mt.tokenizer.pad_token_id
        ):
            trace_start_idx = 1

    # base run with the patched subject
    patched_states = get_all_module_states(mt=mt, input=patched_input, kind=kind)

    if ans_tokens is None:
        # interested answer
        logits = patched_run(mt=mt, inputs=patched_input, states={})
        answer = interpret_logits(tokenizer=mt.tokenizer, logits=logits)[0]
        base_score = get_score(logits=logits, token_id=answer.token_id, metric=metric)
        logger.debug(f"{answer=}")

        # clean run
        clean_logits = patched_run(mt=mt, inputs=clean_input, states={})
        clean_answer, track_ans = interpret_logits(
            tokenizer=mt.tokenizer,
            logits=clean_logits,
            interested_tokens=[answer.token_id],
        )
        clean_answer = clean_answer[0]
        track_ans = track_ans[answer.token_id][1]

        logger.debug(f"{clean_answer=}")
        logger.debug(f"{track_ans=}")
        assert (
            answer.token != clean_answer.token
        ), "Answers in the clean and corrupt runs are the same"

        low_score = get_score(
            logits=clean_logits, token_id=answer.token_id, metric=metric
        )

        ans_tokens = [answer.token_id] # NOTE: This 
        answer = [answer]
    else:
        ans_tokens = [ans_tokens] if isinstance(ans_tokens, int) else ans_tokens
        logger.debug(
            f'tracing answer for {[f"{t}({mt.tokenizer.decode(t)})" for t in ans_tokens]}'
        )
        base_score, base_indv_scores = get_score(
            logits=patched_run(
                mt=mt,
                inputs=patched_input,
                states={},
            ),
            token_id=ans_tokens,
            metric=metric,
            return_individual_scores=True
        )
        answer = []
        logger.debug(f"{base_score=} | {base_indv_scores=}")

        for tok in base_indv_scores:
            pred = PredictedToken(
                token=mt.tokenizer.decode(tok),
                token_id=tok,
            )
            if metric in ["logit", "prob"]:
                setattr(pred, metric, base_indv_scores[tok])
            else:
                pred.metadata = {metric: base_indv_scores[tok]}
            
            answer.append(pred)

        low_score, low_indv_scores = get_score(
            logits=patched_run(
                mt=mt,
                inputs=clean_input,
                states={},
            ),
            token_id=ans_tokens,
            metric=metric,
            return_individual_scores=True,
        )
        logger.debug(f"{low_score=} | {low_indv_scores=}")

    assert (
        low_score < base_score
    ), f"{low_score=} | {base_score=} >> low_score must be less than base_score"
    logger.debug(f"{base_score=} | {low_score=}")

    layer_name_format = None
    if kind == "residual":
        layer_name_format = mt.layer_name_format
    elif kind == "mlp":
        layer_name_format = mt.mlp_module_name_format
    elif kind == "attention":
        layer_name_format = mt.attn_module_name_format
    else:
        raise ValueError("kind must be one of 'residual', 'mlp', 'attention'")
    
    logger.debug(f"---------- tracing important states | {kind=} ----------")

    # Calculate indirect effects in the patched run
    # Use minimum length to avoid KeyError when clean and patched inputs have different lengths
    max_token_idx = min(clean_input.input_ids.size(1), patched_input.input_ids.size(1))
    locations = [
        (layer_idx, token_idx)
        for layer_idx in range(mt.n_layer)
        for token_idx in range(trace_start_idx, max_token_idx)
    ]
    indirect_effects = calculate_indirect_effects(
        mt=mt,
        locations=locations,
        clean_input=clean_input,
        patch_states=patched_states,
        patch_ans_t=ans_tokens,
        layer_name_format=layer_name_format,
        window_size=window_size,
        metric=metric
    )

    indirect_effect_matrix = []
    for token_idx in range(trace_start_idx, max_token_idx):
        indirect_effect_matrix.append(
            [
                indirect_effects[(layer_idx, token_idx)]
                for layer_idx in range(mt.n_layer)
            ]
        )
    
    indirect_effect_matrix = torch.tensor(indirect_effect_matrix)
    if normalize:
        logger.info(f"{base_score=} | {low_score=}")
        indirect_effect_matrix = (indirect_effect_matrix - low_score) / (
            base_score - low_score
        )
    
    return CausalTracingResult(
        patch_input_toks=[
            mt.tokenizer.decode(tok) for tok in patched_input.input_ids[0]
        ],
        corrupt_input_toks=[
            mt.tokenizer.decode(tok) for tok in clean_input.input_ids[0]
        ],
        trace_start_idx=trace_start_idx,
        answer=answer,
        subj_range=subj_range,
        low_score=low_score,
        base_score=base_score,
        indirect_effects=indirect_effect_matrix,
        normalized=normalize,
        kind=kind,
        window=window_size,
        metric=metric,
    )