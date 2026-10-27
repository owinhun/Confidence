# What an Upstream Sound-Event Tag Costs: Claim, Packaging, and Declared Confidence

> [!IMPORTANT]
> **Submitted to ICASSP 2027 — under review.** This manuscript has not been announced as accepted or published.

**[Project Page](https://owinhun.github.io/Confidence/proj/proj.html)** · **[Manuscript PDF](./paper.pdf)**

**Seunghun Oh · Hyunjung Chung · Ganghyeon Jeong · Unsang Park**  
Department of Artificial Intelligence / Department of Computer Science and Engineering, Sogang University  
Corresponding author: Unsang Park

This repository contains research materials for **What an Upstream Sound-Event Tag Costs: Claim, Packaging, and Declared Confidence**, a diagnostic study of how a detector's textual message affects downstream audio and audio–visual language models. We examine **the claim itself**, **declared confidence**, and **message packaging** under controlled interventions, then test their effects in a real sound-event detection cascade. The work is an evaluation study, not a new detector or a universally effective prompt.

## Overview

A sound-event detector may identify an event and pass that decision to a multimodal model as natural language. When the detector is wrong, the transmitted text can conflict with the recording. A single comparison between using and omitting the tag does not tell us which part of the message caused the change.

We distinguish three components:

| Component | Controlled comparison | Diagnostic question |
| --- | --- | --- |
| **Claim** | Add a true or false assertion while preserving the recording and question. | What is the cost of forwarding the detector's decision? |
| **Declared confidence** | Compare `Priority: high` and `Confidence: high` with the same claim and graded value. | Does the *epistemic meaning* of a confidence field change model behavior? |
| **Packaging** | Compare a bare claim with a structured `Source / Claim / Confidence` block. | How strongly does the bundled message format amplify or attenuate a claim? |

Claim insertion and changing `Priority` to `Confidence` are single-factor comparisons. **Packaging changes several presentation features together**; its measured effect is not attributed to one isolated formatting choice.

## Method

1. **Construct matched questions.** Evaluate sound events from 12 VGGSound classes alongside diagnostic colour, noise and silence attributes. Keep the recording, target property, and question fixed within a matched comparison.
2. **Check perceptual eligibility.** Use no-metadata accuracy and two-direction recall to identify model–attribute settings in which the reader can reliably assess the underlying signal. Audit answer-option order separately.
3. **Control the message.** Compare signal-only, empty-block, source-only, claim-bearing `Priority`, and `Confidence` conditions. Preserve the claim wording across claim-bearing conditions and vary declared confidence from very low to very high, including a numeric representation.
4. **Measure paired behavior.** Use deterministic option-letter logits, accuracy, probability of override (POR), truth-aligned margins, and the claim-aligned contrast β. Report source-clustered bootstrap uncertainty and paired tests, with appropriate distinctions between headline and sensitivity analyses.
5. **Evaluate a real cascade.** Calibrate CLAP-based sound-event decisions, compare forwarding and omitting the resulting tag, and examine serialization, detector improvement, error repair and reader-specific break-even thresholds.

The study evaluates six reader models: **Qwen2.5-Omni-7B, Qwen3-Omni-30B-A3B, VideoLLaMA2.1-7B-AV, Qwen2-Audio-7B, Audio Flamingo 3, and Nemotron 3 Nano Omni**. The paper distinguishes order-robust settings from sensitivity-only cases; not every comparison includes every model–attribute pair.

## Key Findings

- **The forwarded claim has the largest effect.** Adding a false claim produces a larger adverse margin change than the subsequent high-confidence field naming step in most tested comparisons.
- **The field name matters beyond its numeric or graded value.** Naming the field `Confidence` rather than `Priority` produces a **2.1–3.9×** larger high-to-low claim-aligned margin swing in the headline comparisons. This is **not** an accuracy multiplier.
- **Unconditional forwarding can be worse than omission.** In the real CLAP cascade, forwarding the detector's decisions as bare text reduces reader accuracy by **6.6–11.1 percentage points** in all **four order-robust** cascade settings. Audio Flamingo 3 is analyzed separately as order-sensitive.
- **Packaging changes how much a claim is followed.** Structured Source/Claim blocks attenuate the true–false claim response gap relative to bare text in the order-robust cascade analyses; this is a bundled format effect.
- **Detector quality changes the decision.** A stronger upstream detector can cross a model-specific break-even threshold. The appropriate forwarding policy depends on the detector, reader, input data and serialization.

These findings describe the tested diagnostic settings; they do not establish that tags should always be omitted or that confidence fields have one universal effect across multimodal systems.

## Manuscript and Reproduction

The manuscript's LaTeX source, references and figures are available under [`paper/source/`](./paper/source/). Supporting experimental and analytical code is organized under [`code/scripts/`](./code/scripts/) and [`code/research_reports/`](./code/research_reports/).

**Research scripts are available, but a portable, validated end-to-end reproduction environment is not yet provided.** Some scripts contain paths specific to the original research workspace; model checkpoints, datasets, generated outputs and pinned environments are not bundled.

```bash
git clone https://github.com/owinhun/Confidence.git
cd Confidence
```

These commands download the repository; they are **not** a complete reproduction procedure.

## Release Status

- [x] Manuscript LaTeX source and research materials available
- [x] Research and analysis scripts preserved
- [x] Project Page HTML available in this repository
- [ ] Public PDF matching the manuscript source uploaded as `paper.pdf`
- [ ] Portable environment and dependency specifications published
- [ ] Dataset and model-access instructions documented
- [ ] End-to-end reproduction commands validated

## News

- **ICASSP 2027** — Manuscript submitted; peer review is ongoing.

## Acknowledgements

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2024-00438027, Development of an intelligent security and risk response system based on multi-modal data integration analysis), and by the IITP under the Artificial Intelligence Innovation Human Resources Development grant (IITP-RS-2026-25547954) funded by the Korea government (MSIT). The authors declare no relevant conflicts of interest.

## Contact

For questions about this research, please open an issue in this repository.

## Citation

This paper is **under review**. The following is a provisional manuscript citation, not a published conference proceedings entry.

```bibtex
@unpublished{oh2026soundeventtag,
  title  = {What an Upstream Sound-Event Tag Costs: Claim, Packaging, and Declared Confidence},
  author = {Oh, Seunghun and Chung, Hyunjung and Jeong, Ganghyeon and Park, Unsang},
  year   = {2026},
  note   = {Manuscript submitted to ICASSP 2027; under review}
}
```
