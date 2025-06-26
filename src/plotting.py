import os
import matplotlib.pyplot as plt
from typing import Optional, Literal
from src.trace import CausalTracingResult

def get_color_map(kind: Literal["residual", "mlp", "attention"] = "residual"):
    if kind == "residual":
        return "Purples"
    if kind == "mlp":
        return "Greens"
    if kind == "attention":
        return "Reds"
    return "Greys"

def replace_special_tokens(token_list, pad_token="[PAD]"):
    for i, token in enumerate(token_list):
        if token.startswith("<|") and token.endswith("|>"):
            token_list[i] = pad_token
    return token_list

def plot_trace_heatmap(
    result: CausalTracingResult,
    savepdf: Optional[str] = None,
    model_name: Optional[str] = None,
    scale_range: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    color_map: Optional[str] = None
):
    scores = result.indirect_effects
    corrupt_tokens = replace_special_tokens(result.corrupt_input_toks)
    patch_tokens = replace_special_tokens(result.patch_input_toks)

    if scale_range is None and result.normalized is True:
        scale_range = (0, 1)

    tokens = []
    shifted_subj_range = (
        result.subj_range[0] - result.trace_start_idx,
        result.subj_range[1] - result.trace_start_idx,
    )
    for idx, (corrupt_tok, patch_tok) in enumerate(
        zip(
            corrupt_tokens[result.trace_start_idx :],
            patch_tokens[result.trace_start_idx :]
        )
    ):
        if idx in range(*shifted_subj_range):
            tokens.append(
                f'"{patch_tok}" / "{corrupt_tok}"'
                if corrupt_tok != patch_tok
                else f'"{corrupt_tok}"*'
            )
        else:
            tokens.append(f'"{corrupt_tok}"')

    plt.rcdefaults()
    with plt.rc_context(
        rc={
            "font.family": "Times New Roman",
            "font.size": 6,
        }
    ):
        fig, ax = plt.subplots(figsize=(3.5, len(tokens) * 0.08 + 1.8), dpi=200)
        scale_kwargs = dict(
            # vmin=result.low_score if scale_range is None else scale_range[0],
        )
        if scale_range is not None:
            scale_kwargs["vmin"] = scale_range[0]
            scale_kwargs["vmax"] = scale_range[1]

        heatmap = ax.pcolor(
            scores,
            cmap=get_color_map(result.kind) if color_map is None else color_map,
            **scale_kwargs,
        )

        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(scores))])
        ax.set_xticks([0.5 + i for i in range(0, scores.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, scores.shape[1] - 6, 5)))
        # print(len(tokens))
        ax.set_yticklabels(tokens)

        if title is None:
            title = f"Indirect Effects of {result.kind.upper()} Layers"
        ax.set_title(title)

        if result.window == 1:
            ax.set_xlabel(f"single restored layer within {model_name}")
        else:
            ax.set_xlabel(
                f"center of interval of {result.window} restored {result.kind.upper()} layers"
            )
        metric_marker = {
            "prob": "p",
            "logit": "lgt",
            "log_norm": "ln",
        }
        color_scale = plt.colorbar(heatmap)
        ans_label = "Ans" if len(result.answer) > 1 else f'"{result.answer[0].token}"'

        color_scale.ax.set_title(
            f"{metric_marker[result.metric]}({ans_label})",
            y=-0.12,
            fontsize=8,
        )

        if savepdf is not None:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight", dpi=300)
        plt.show()