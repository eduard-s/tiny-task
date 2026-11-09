# Comparison and product choice

## The decision

TinyTask helps a developer decide whether a repeated classification task can use a local model.
It starts with labeled examples and explicit quality targets.

Choose it when the task has a stable label set and representative calibration and test data.
The first useful workflow is a support queue with an explicit review route.

## Existing projects

| Project | Main input and purpose | Relationship to TinyTask |
| --- | --- | --- |
| Prompt2Model | A task description starts a pipeline that can retrieve or generate data and train a deployable model. | A broader starting point when the developer starts from a prompt. |
| SetFit | Labeled examples train a classifier with Sentence Transformers. | The optional TinyTask training backend. |
| TinyTask | Labeled splits and quality criteria produce a candidate, abstention policy, and artifact-bound evaluation. | A focused workflow for a deployment decision. |

Prompt2Model already connects instructions with smaller models. TinyTask cannot claim that idea as new.
See its [repository](https://github.com/neulab/prompt2model) and [paper](https://arxiv.org/abs/2308.12261).

SetFit already supplies multilingual training, distillation, model export, and evaluation interfaces.
TinyTask does not claim those capabilities as inventions.
See its [documentation](https://huggingface.co/docs/setfit/index), [distillation guide](https://huggingface.co/docs/setfit/how_to/knowledge_distillation), and [ONNX guide](https://huggingface.co/docs/setfit/tutorials/onnx).

The comparison used these source revisions on 2026-11-09:

| Project | Revision |
| --- | --- |
| Prompt2Model | `d4cd23d50c8f5edc5755ab487bcee18121d42a4f` |
| SetFit | `d70dad84de757d30f03465d120e0d3fad2c725b0` |

This review covers the documented workflows. A developer can implement similar policies with SetFit directly.

## Reasons to choose TinyTask

1. Import existing labeled records through one JSONL contract.
2. Detect known split leakage before training or evaluation.
3. Select acceptance thresholds on calibration data only.
4. Measure errors and coverage on separate test data.
5. Refuse release approval when the sample or result is inadequate.
6. Load the saved candidate with its matching acceptance report.

These steps can reduce integration work. They do not establish a better learning algorithm.
The current implementation needs no hosted dashboard or inference API.

## Choose the existing project when it fits

Use Prompt2Model when prompt-driven data discovery or generation is the main need.
Use SetFit directly when the developer needs training research, custom losses, or a different evaluation workflow.
Use TinyTask when the acceptance decision and its evidence are the main need.

## Evidence needed for adoption

The repository contains a runnable demo and public-data results.
It does not contain evidence of repeat use by external teams.

Ask three teams with repeated classification tasks to use the package on their own data.
Record setup time, false acceptance, review volume, and repeat use after one week.
Keep confidential inputs in each team's environment.

The adoption target is two repeat users from those three trials.
This is a proposed target, not an achieved result.
