# What an Upstream Sound-Event Tag Costs, and Which Part of It: Claim, Packaging, and Declared Confidence

> [!IMPORTANT]
> **ICASSP 2027 submission — under review.** This work has **not** been announced as accepted.

**[Project Page](https://owinhun.github.io/Confidence/)** · **[Manuscript PDF](./paper/archive/Confidence_v89.pdf)**

**Seunghun Oh\* · Jaehong Myung\* · Unsang Park†**  
\* Equal contribution · † Corresponding author  
Department of Artificial Intelligence / Department of Computer Science and Engineering, Sogang University

This repository contains research materials for **What an Upstream Sound-Event Tag Costs, and Which Part of It: Claim, Packaging, and Declared Confidence**, a diagnostic study of how an upstream sound-event detector's textual message influences downstream audio and audio–visual language models. We separate the **claim itself**, the **way the message is packaged**, and **declared confidence**. This is an evaluation and analysis study, not a new detector or an inference-time method guaranteed to improve performance.

## Overview

In a cascaded audio or audio–visual system, a detector may first label an event and pass its decision to a language model as text. That text can help, but it can also conflict with the recording. A single comparison between “with tag” and “without tag” does not reveal which part of the message produces the effect.

We examine three components:

| Component | Example / intervention | Diagnostic question |
| --- | --- | --- |
| **Claim** | `Claim: "The clip does not contain the sound of X."` | What happens when the upstream stage makes a true or false assertion? |
| **Declared confidence** | Change the field label from `Priority: high` to `Confidence: high`, while keeping the claim and graded value fixed. | Does an *epistemic* confidence field matter beyond a generic graded field? |
| **Packaging** | Present the tag as a bare sentence or in a structured Source/Claim/Confidence block. | How does the bundled message format change the influence of the claim? |

Importantly, **claim insertion** and the **Confidence-versus-Priority field-name change** are single-factor comparisons. The packaging comparison changes a bundle of message properties and is interpreted as a measured format effect, **not** as the isolated causal effect of any one formatting feature.

## Method

1. **Prepare controlled examples.** Evaluate sound events drawn from 12 VGGSound classes, alongside constructed colour, noise, and silence attributes. The protocol keeps the relevant recording and question fixed when comparing message interventions.
2. **Screen perceptual eligibility.** Before evaluating metadata effects, use no-metadata accuracy and both answer-order recalls to identify cells where a model can reliably judge the property. Report excluded and sensitivity-only cells separately.
3. **Intervene on one component at a time.** Compare signal-only, empty-block, source-only, non-epistemic `Priority`, and declared `Confidence` conditions. Keep the claim wording identical across claim-bearing conditions; sweep verbal confidence levels from very low to very high and examine a numeric scale.
4. **Measure paired responses.** Read deterministic yes/no option-letter logits and examine accuracy, probability of override (POR), truth-aligned margins, and claim-aligned and symmetric contrasts (β and α). Use source-clustered bootstrap intervals, paired tests, and answer-order audits as defined in the manuscript.
5. **Test a real cascade.** Calibrate CLAP-based sound-event tags, forward actual detector decisions with alternative serializations, compare with omitting the tag, and examine error-repair and break-even analyses. The findings are conditional on the tested readers, detector, data and serialization.

Six evaluated reader models are **Qwen2.5-Omni-7B, Qwen3-Omni-30B-A3B, VideoLLaMA2.1-7B-AV, Qwen2-Audio-7B, Audio Flamingo 3, and Nemotron 3 Nano Omni**. Not all model–attribute cells qualify for every headline analysis; order-robust and sensitivity-only results are separated in the paper.

## Key Findings (v89 manuscript)

- **The claim is the largest source of change.** In the controlled condition walk, inserting a false claim is a larger margin step than subsequently naming its confidence in most reported ladder cells.
- **The confidence label matters.** Relative to the non-epistemic `Priority` control, the `Confidence` field yields approximately **2.1–3.9×** the high-to-low claim-aligned margin swing across the 10 headline cells (order-averaged). This is a margin result; it should not be interpreted as a corresponding accuracy multiplier.
- **Forwarding a real detector's tag can hurt.** In the CLAP cascade, bare textual forwarding reduces accuracy by **6.6–11.1 percentage points** in each of the **four order-robust** reader cells. Audio Flamingo 3 is reported separately as sensitivity-only, not included in that four-cell claim.
- **Packaging moderates the effect.** A structured block attenuates the true-versus-false claim span relative to a bare sentence in the order-robust cascade cells. Because packaging is a bundle, the analysis does not attribute this solely to a single formatting element.
- **The forwarding decision depends on relative accuracy.** Error-repair and break-even analyses show that improving the upstream stage can reverse the outcome for a reader; omission is not asserted to be universally optimal for every possible detector.

These summaries reflect the **archived v89 manuscript**. Check them against the **18 September 2026 source package** before treating them as final current-submission claims.

## Manuscript Versions

| Version | Location | Status |
| --- | --- | --- |
| **18 September 2026 source package** | [`paper/source/`](./paper/source/) | Latest package identified in the repository description; build from `Template.tex`. |
| **v89 manuscript (7 September 2026)** | [`paper/archive/icassp_body_confidence_v89.tex`](./paper/archive/icassp_body_confidence_v89.tex) · [`Confidence_v89.pdf`](./paper/archive/Confidence_v89.pdf) | Archived reproducible snapshot, **not** labelled as the latest submission. |

The attached v89 ZIP was built using `pdflatex Template`, `bibtex Template`, `pdflatex Template`, `pdflatex Template`. The newer source should be compiled and verified independently before publishing a current-version PDF.

## Installation and Reproduction

**Status: research scripts are available, but a fully portable, verified reproduction setup has not been released.** Some scripts depend on the original `/workspace/svhalluc` layout. Model checkpoints, datasets, cached generations and Python environments are not bundled with this repository.

```bash
git clone https://github.com/owinhun/Confidence.git
cd Confidence
```

Relevant research material is organized under [`code/scripts/`](./code/scripts/) and [`code/research_reports/`](./code/research_reports/). The commands above only check out the repository; **they do not reproduce the experiments**. A pinned environment, data preparation instructions and validated end-to-end commands will be added if and when those resources are released.

## Release Status

- [x] Manuscript source and archived v89 snapshot organized
- [x] Research and analysis scripts preserved
- [ ] Current-submission PDF compiled, verified and approved for public release
- [ ] Project page published through GitHub Pages
- [ ] Portable dependency / hardware specifications published
- [ ] Dataset and model-access instructions documented
- [ ] End-to-end reproduction commands validated

## News

- **September 2026** — Manuscript submitted to **ICASSP 2027**; review is ongoing. This is a submission announcement, **not** an acceptance announcement.

## Acknowledgements

The following acknowledgement is transcribed from the **v89 manuscript**; verify its wording against the latest submitted version before final publication:

> This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2024-00438027, Development of an intelligent security and risk response system based on multi-modal data integration analysis), and by the IITP under the Artificial Intelligence Innovation Human Resources Development grant (IITP-RS-2026-25547954) funded by the Korea government (MSIT).

## Contact

For research inquiries, please open an issue in this repository. Contact information can be added after confirming the preferred public address.

## Citation

The paper is **under review** and does not yet have an official proceedings citation. The following is a **provisional manuscript citation**, not an accepted-conference citation:

```bibtex
@unpublished{oh2026upstreamtag,
  title  = {What an Upstream Sound-Event Tag Costs, and Which Part of It: Claim, Packaging, and Declared Confidence},
  author = {Oh, Seunghun and Myung, Jaehong and Park, Unsang},
  year   = {2026},
  note   = {Manuscript submitted to ICASSP 2027; under review}
}
```
