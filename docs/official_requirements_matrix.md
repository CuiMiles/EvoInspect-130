# Official requirements matrix

Verified locally on 2026-08-23 from the files under `official/`. This matrix paraphrases the
requirements and does not redistribute the source documents.

| ID | Source location | Requirement | Repository evidence | Status | Block level |
|---|---|---|---|---|---|
| AOI-01 | Huawei topic, item 2 | Accept image and video and detect appearance, dimension, missing-part, color, logic and sequence anomalies | `project_spec.yaml`, planned image/video interfaces | Planned | Critical |
| AOI-02 | Huawei topic, item 2.1 | Reference input 2500×2500; GTX 2060 or lower; model runtime <200 ms; CPU <2 s is a challenge | latency protocol in `docs/04_EVALUATION_PROTOCOL.md` | Protocol only | Critical |
| AOI-03 | Huawei topic, item 2.2 | Support few-shot or zero-shot rapid deployment and generalization | P1/P2 protocols | Protocol only | High |
| AOI-04 | Huawei topic, item 2.3 | Operator feedback should support traceable dynamic optimization | GuardedAdapt design | Design only | High |
| AOI-05 | Huawei topic, requirement 1 | Submit model explainability and selection rationale with evidence | Claim ledger and planned model cards | Partial | High |
| AOI-06 | Huawei topic, requirement 2 | Train on public datasets and self-validate generalization | dataset registry | Not started | Critical |
| AOI-07 | Huawei topic, requirement 3 | Competition score 60%; expert score 40% | `project_spec.yaml` | Recorded | Medium |
| AOI-08 | Huawei topic, requirement 4 | Adapt using only 100 labeled normal and 30 defective items, then evaluate on 1000+ test images | split protocol and hidden-test simulator plan | In progress | Critical |
| AOI-09 | Huawei topic, requirement 4 | Competition portion combines completeness 50%, answer accuracy 20%, detection time 30% | `project_spec.yaml` | Recorded; nesting interpretation unconfirmed | Medium |
| AOI-10 | Huawei topic, requirement 5 | Submit model code and usage document with public-test results; small models encouraged | package plan | Planned | Critical |
| SUB-01 | Submission spec, item 2.1 | Entry summary: PDF, no template, ≤300 Chinese characters, prescribed team/project filename | `docs/submission_manifest.yaml` | Recorded | Critical |
| SUB-02 | Submission spec, item 2.2 | Project document: complete official template, PDF, prescribed filename | `docs/template_mapping.md` | Recorded | Critical |
| SUB-03 | Submission spec, item 2.3 | Video: MP4, ≤5 minutes, ≤200 MB, prescribed filename | `docs/submission_manifest.yaml` | Recorded | Critical |
| SUB-04 | Submission spec, item 2.4 | Auxiliary material: ZIP, ≤200 MB, prescribed filename | `docs/submission_manifest.yaml` | Recorded | Critical |
| SUB-05 | Submission spec, item 4 | Submission deadline shown in the local file: 2026-09-01 23:59 | `STATUS.md` | Recorded | Critical |
| SUB-06 | Submission spec, item 1 | Explain innovation rigorously, compare prior work, report inference evidence, and disclose data/knowledge/algorithm/hardware sources | claim and third-party ledgers | Partial | High |
| TPL-01 | Project template, sections 1–4 | Preserve sections: project overview, project plan, implementation plan, references, and change history | `docs/template_mapping.md` | Recorded | Critical |

## Unresolved official ambiguities

The source does not settle annotation granularity, exact accuracy metric, whether 200 ms is
model-only or end-to-end, adaptation-time limits, exact evaluation hardware/software, video
scoring, output API, dependency/package limits, test-time adaptation, or external pretraining.
These remain blockers in `docs/08_ORGANIZER_QUESTIONS.md`; no implementation may present an
assumption as an official fact.

## Source fingerprints

- Huawei topic DOCX: `a99b7de5534d7f45e079a292031f9886acf3f0aeddfe4a930345ae99dc665b48`
- Submission specification PDF: `4ab678a24e8e36cb86c25b04150604d10ecbe8442dec11b543a80d550305b901`
- Project template PDF: `9a7a4fb64f3143dac60bead2ee9b27bff8f7b29d403320e650dbfd6e87a62d5a`

