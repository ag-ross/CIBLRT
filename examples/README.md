# CIB-LRT practitioner examples from: Beyond Consistent Scenarios: Deriving Indirect Influence, Transition Resistance, and Adjustment Dynamics

Standalone worked examples for CIB practitioners who want to apply Linear Response Theory (LRT), as introduced in `Beyond Consistent Scenarios: Deriving Indirect Influence, Transition Resistance, and Adjustment Dynamics`, to their own Cross-Impact Balance study. No external data files are needed; the cross-impact matrix is hardcoded in each script.

## Current examples

**`example_policy_shock.py`**: Unit-impulse shock analysis at the high-ambition attractor of a 10-descriptor energy-transition CIM (PyCIB's built-in `DATASET_C10`). Asks: if Policy Stringency receives a unit push, how does that signal propagate through the cross-impact network, in which direction, and in what sequence? Produces a ranked console summary, the Type I cross-impact multiplier diagonal with sign-reversal flags, and a plot with 5th–95th percentile Monte Carlo uncertainty ribbons.

## Dependencies

numpy, scipy, matplotlib, and PyCIB are required. If scipy is missing, the script falls back to the numpy-only `_lrt_shim` shipped under `workings/_lrt_shim/`.

```
pip install numpy scipy matplotlib
pip install git+https://github.com/ag-ross/PyCIB.git
```

## Usage

```
cd examples
python example_policy_shock.py
```

Two output files are written to the same folder: `output_policy_shock.pdf` (the IRF plot with Monte Carlo ribbons) and `output_policy_shock_summary.txt` (a plain-text record of the run parameters, ranked response table, Type I multiplier diagonal, and any MC warnings).

To adapt the example to your own CIM, replace `DESCRIPTORS` and `IMPACTS` near the top of the script, change `SHOCK_DESCRIPTOR` to the descriptor you want to push, and update `ATTRACTOR_PROFILE` to identify the scenario of interest.

---

## Using these results carefully

The shock-response outputs are structural indicators derived from the cross-impact matrix at a specific attractor. Used appropriately, they add genuine analytical value beyond standard CIB; misread, they can be misleading. The following guidance is drawn directly from the CIB-LRT methodology.

### What the results tell you, and what they do not

**The time axis is dimensionless.** The horizontal axis is model time, not calendar time. A descriptor that peaks early responds before others; that ordering is a property of the cross-impact network's eigenstructure and is robust. Converting that ordering into years requires external empirical calibration that the CIB elicitation cannot provide. Do not attach calendar dates to the curves without additional grounding. One partial remedy is available within the CIB framework: the expert panel can be asked to supply a single temporal calibration anchor, namely a collective judgement on the approximate real-world duration before the fastest-responding descriptor would show a meaningful response to the shock. That anchor fixes the location of the first IRF peak in calendar time, rescaling the axis without altering the derived ordering of responses. The resulting calendar axis carries the same epistemological status as the cross-impact scores themselves: expert-elicited, qualitative, and conditional on panel composition and descriptor framing. It does not identify propagation speeds in any econometric sense. If this approach is used, report the elicited anchor explicitly, present the dates as illustrative ranges rather than projections, and note that econometric validation would be required before the implied timescales could bear quantitative policy inference.

**The ordering of responses is robust; absolute magnitudes are not.** Which descriptors respond positively versus negatively, and which peak earlier versus later, are invariant to the IO-3 rescaling choice and stable across the range of CIM score uncertainty. The absolute heights of the curves depend on the rescaling target (ρ_target) and should not be compared across studies that used different values. Always report ρ_target alongside any quoted result.

**The shock profile is attractor-specific.** The same push on a descriptor produces a structurally different response at each consistent scenario, because W is evaluated at the attractor states. A shock profile computed at one attractor does not generalise to another. If you want to compare how a push propagates across scenarios, run the analysis at each attractor separately and fix ρ_target to the same value for all of them.

**Sign reversals in the shock response are explained by the Type I cross-impact multiplier.** If a descriptor's response curve changes sign during the horizon (rising initially and then reversing direction before returning to zero), this is not noise or a modelling artefact. It is a structural consequence of feedback loops in the cross-impact network, captured by the Type I cross-impact multiplier Λ = (I − W)⁻ᵀ. A negative diagonal entry Λ_jj indicates that sustained pressure on descriptor j, propagated through all indirect feedback chains, ultimately reverses its own cumulative activation. This is attractor-specific: the same descriptor may show a positive diagonal at one consistent scenario and a negative one at another. The script prints the multiplier diagonal alongside the shock-response summary; inspect it for any descriptor whose response curve crosses zero.

**The linearisation is local.** All LRT outputs are derived from W evaluated at one attractor. They are formally valid for small perturbations. Treat them as network-derived structural fingerprints of that attractor, not as literal predictions of system behaviour under large forcing.

**Results are conditional on the elicited CIM.** A different expert panel, facilitation process, or descriptor framing could produce a substantially different matrix and correspondingly different shock profiles. The Monte Carlo ribbons address sensitivity to the magnitude of individual agreed scores, but not to structural uncertainty about which pairs are positive, negative, or zero. More fundamentally, the cross-impact scores are expert-elicited qualitative judgements about pairwise influence direction and strength. They were designed to support consistency analysis, not to estimate propagation coefficients. The IRF and shock profile produce a time axis, a ranking of descriptors by response strength, and sign information for each descriptor. These outputs resemble those of an identified structural model, and that resemblance invites a causal reading that the scores were never designed to support. This is the error of mistaking elicitation for identification. Read the outputs as structural fingerprints of the elicited matrix at that attractor, not as estimated causal effects.

### Things to always do

- **Run the IO-3 sensitivity check.** Vary `IO3_TARGET_RHO` across the range 0.5–0.99 and confirm that the ranking of descriptors by response strength and the sign of each response remain stable. This invariance is a property of the eigenstructure of W and should hold for any well-elicited CIM; verify it holds for yours before reporting. Always report the value of `IO3_TARGET_RHO` you used alongside any quoted result.

- **Report and vary the calibration horizon t₁ if using susceptibility or cross-scenario IRF analysis.** The calibration horizon t₁ does not appear in this example, which computes the unit-impulse IRF directly. However, if you extend the workflow to the susceptibility matrix or cross-scenario IRF (as done in the main paper), all absolute output values will depend on t₁ (the time window used in the susceptibility computation). In that case, run the analysis across a range of t₁ values (e.g. 0.5 to 10.0) and confirm that the ranking of descriptors by response strength is stable; always report t₁ alongside any quoted result. The accompanying paper and supplementary materials document this sensitivity.

- **Inspect the Monte Carlo ribbons.** Narrow ribbons confirm a finding is a robust structural property of the network. Wide ribbons (particularly where the band straddles zero) mean the direction of that descriptor's response is genuinely uncertain at the elicitation noise level used, and should not be reported as a firm conclusion. Use N = 10,000 draws for smooth bands (the default in this script). Always report the noise scale σ used; the default in this script is σ = 0.5, grounded in published estimates of CIM score uncertainty.

- **Vary MC_SIGMA and confirm conclusions hold across the range.** σ = 0.5 is the default, not a fixed truth about elicitation uncertainty. Re-run with `MC_SIGMA` set to, e.g., 0.25, 0.5, and 1.0, and confirm that the sign and ranking of descriptor responses are stable, even as ribbon width naturally grows with σ. This mirrors the σ-sensitivity check reported in Supplementary Figure S3 of the paper, and is distinct from simply reporting the σ value used at a single setting.

- **Compare shock profiles across attractors only with fixed ρ_target.** If you run the shock at two different consistent scenarios and want to compare the results, both runs must use the same rescaling target. Without that, differences in curve height conflate modelling choices with genuine structural differences between attractors.

- **Verify your consistent scenarios before running LRT.** The LRT objects are defined at a confirmed consistent scenario. An incorrect attractor profile (one that is not a genuine fixed point of the cross-impact network) will silently produce outputs that carry no structural meaning. Use PyCIB's consistency check to confirm that your `ATTRACTOR_PROFILE` satisfies the argmax condition before interpreting any results.

- **Report which attractor was used.** The shock profile is attractor-specific. Always identify the consistent scenario by its full descriptor-state profile, not just a shorthand label, so that results are reproducible and comparable across studies.

- **Check stability after rescaling.** The script prints a stability confirmation. A Monte Carlo rejection rate substantially above zero is a diagnostic that the rescaled network is near the stability boundary and the results need scrutiny.

### Things to avoid

- Do not read the model-time axis as real-world duration.
- Do not report absolute curve heights as scale-free measures of influence; the ordering and sign are what matter.
- Do not generalise a shock profile from one attractor to another without re-running the analysis at the target attractor.
- Do not treat a wide MC ribbon as noise to be averaged away: it signals genuine sensitivity to elicitation uncertainty.
w
### Extending to cross-scenario IRFs and perturbation budgets

This script covers two of the four closed-form analytical objects described in the paper: the unit-impulse shock analysis and the Type I cross-impact multiplier. The remaining two (the cross-scenario impulse response function (IRF) and the perturbation budget) require adaptations to the same workflow. These details are given fully in the replication code, but are more involved.

## Licence

Details are provided in the LICENSE file in the main project folder.

### Cite as:

See `README.md` in the main project folder.

---

## References

See `README.md` in the main project folder.
