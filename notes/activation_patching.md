# Activation Patching

## Notes from "How to use and interpret activation patching

Alternatively we could use path patching to confirm the preceise interactions. Say we want to test whether the Peace embedding is necessary as an input to L0H0, as an input to L1N42, or both. For this we could patch only the corresponding paths, and find that denoising (1) "Nobel -> L0H0" and (2) "Peace -> L1N42" paths is sufficient. Alternatively, we might find that noising every path except for (1) "Nobel -> L0H0", (2) :L0 -> L1N42", and (3) "Peace -> L1N42" does not break performance. Note again that denoising only required restoring two paths (restoring a cross-section of the circuit) while noising required leaving 3 paths clean (the full circuit).

### 3.2 Concepts & Gotchas

**Sensitivity & prompt choice:** A positive patching result implies you have found activations dealing with the difference between the clean and corrupt prompt. Make sure to consider all degrees of freedom in a task, and consider multiple sets of corrupted prompts if necessary.

**Scope of activation patching:** More generally, activation patching is always based on prompt distributions, and does not make statements for model behavior outside these specific distributions.

**No minimality:** Here, and in many parts of the literature, a circuit is treated as a collection of model components that are responsible for a particular model behavior. We typically make no claims that we have found the smalles such collection of components, we only test that this collection is sufficient.

**Backup behavior & OR-gates:** In some cases researchers have discovered "Backup heads", components that are not normally doing the task but jump into action of other componentts are distrupted. For example, in IOI when one ablates a name mover head (a key component of the circuit) a backup name mover head will activate and then do the task instead.

It can be helpful to think of these as OR-gates where either component is sufficient for the model to work. This does not fit well into our attemtps of defininig a circuit, nor plays well with the circuit finding methods above. Despite the name mover heads being important, if we ablate them then, due to backup heads compensating, the name movers look less important. Fortunately, backup behavior seems to be lossy, i.e. if the original component boosted the logits by +X, the backup compensates for this by boosting less than X (the Hypdra effect paper found 0.7\*X). Thus these backup component weaken the visibility of the original component, but it is usually still visible since even 0.3\*X is a relatively large effect.

**Negative components:** Some work in this area noticed attention heads that consistently negatively affected performance, and noising them would increase performance. This is problematic, because it makes it hard to judge the quality of a circuit analysis: it may look like we've fully recovered (or more than fully recovered!) performance, by finding half the positive components but excluding all negative ones. This is an unsolved problem. Conmy et al. propose using KL divergence as a metric to address this, which penalizes any deviation (positive or negative), at the cost of also tracking lots of variation we may not care about.

### 4. Metrics and common pitfalls

So far we talked about "preserving" and "restoring" performance, but in practice, model performance is not binary but a scale. Typically we find some components matter a lot, while others provide a small increease in performance. For the best interpretability we might look for a circuit restoring e.g. 90% of the model's performance, rather than reaching exactly 1--%. A useful framing is the "pareto fronteir" of circuit size vs. performance recovered -- recovering 80% of performance with 1% of the components is more impressive than 90% of the performance with 10% of the components, but there will always be a minimum circuit size to recover a given level of performance.

It's easy to treat metrics as an after-thought, but we believe the right or wrong choice of metric can significantly change the interpretation of patching results. Especially for exploratory patching, the wrong metric can be misleading. The choice of metric matters less for confirmatory patching, where you expect a binary-ish answer ("have I found the circuit or not") and all metrics should agree.

#### 4.1 The logit difference

Logit difference measures to what extent the model knows the correct answer, and it allows us to be specific: We can control for things we don't want to measure (e.g. components that boost both, Mary and John, in the IOI example) by choosing the right logits to compare (e.g. Mary vs John, or multiple-choice answers). The metric also is a mostly linear function of the residual stream (unlike probability-based metrics) which makes it easy to directly attribute logit difference to individual components ("direct logit attribution", "logit lens"). It's also a "softer" metric, allowing us to see partial effects on the model even if they don't change the rank of the output tokens (unlike e.g. accuracy), which is crucial for exploratory patching.

*Intuition for why logits and logit differences (LDs) are a natural unit for transformers:* The residual stream and output of a transformer is a sum of components. Every comonent added to the residual stream corresponds to an addition to the logit difference (as the logit difference corresponds to a residual stream direction, up to layuer norm). A model component can easily change the logit difference by some absolute amount (e.g. +1 logit difference). It cannot easily change the logit difference by a relative amount (logit difference *= 1.5), or change the probabilities by a specific amount (prob += 0.20).

### 5. Summary

In most situations, use activation patching instead of ablations. Different corrupted prompts give you different information, be careful about what you choose and try to test a range of prompts.

There are two different directions you can patch in: denoising and noising. These are not symmetric. Be aware of what a patching result implies!

- Denoising (a clean -> corrupt patch) shows whether the patched activations were sufficient to restore the model behavior. This implies the components make up a cross-section of the circuit.
- Noising (a corrupt -> clean patch) shows whether the patched activations were necessary to maintain the model behavior. This implies the components are part of the circuit.

## References

- [Nanda, Neel; Attribution Patching: Activation Patching At Industrial Scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching#what-cant-activation-patching-teach-us=)