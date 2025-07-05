def trace_important_states(
    mt: ModelandTokenizer,
    prompt_template: str,
    clean_subj: str,
    patched_subj: str,
    clean_input: Optional[TokenizerOutput] = None,
    patched_input,
    kind,
    window_size,
    normalize,
    trace_start_marker,
    metric,
    ans_tokens
) -> CausalTracingResult:
    aligned = align_patching_positions(
        mt=mt,
        prompt_template=prompt_
    )