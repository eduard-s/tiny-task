# Datasets and partition rules

## Current public cohort

The development run uses [CLINC150](https://github.com/clinc/oos-eval).
Its source accompanies Larson and colleagues' 2019 study of intent classification and unsupported requests.
The source license is CC BY 3.0.

The source revision is `828f8093932c8fe6ca7936c3d2e52903b1c523de`.
The source file is `data/data_full.json`.
Its SHA-256 is `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`.

The first run uses ten related support intents.
Five other banking intents become unsupported requests for this task.
The original unsupported split supplies a separate, broader unknown group.

| Split | Known requests | Related unknown requests | Original unknown requests | Total |
| --- | ---: | ---: | ---: | ---: |
| Training | 200 | 0 | 0 | 200 |
| Calibration | 200 | 100 | 100 | 400 |
| Test | 300 | 150 | 1,000 | 1,450 |

Training selects 20 examples per known label with seed `42`.
Calibration and test use the corresponding original dataset splits.
The source preparation code records the selected IDs and removed normalized duplicates.
The current cohort removes zero duplicates.

The unknown share differs between calibration and test.
This is an unknown-heavy stress case, not a production traffic estimate.
Accepted accuracy depends on that mixture. Read the separate unknown groups beside the aggregate result.

The checkpoint's original training data overlap is not established here.
This small cohort is development evidence, not an untouched generalization benchmark.

## Planned second cohort: BANKING77

[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) supplies related banking intents under CC BY 4.0.
The reviewed source revision is `57ec275d8078af65b7731c2a98be812d844a6d6b`.

Use the original test partition for the final comparison.
Divide the original training partition into training and calibration groups.
Use fixed budgets of 16, 32, and 64 training examples per known label.

For a separate unsupported-intent stress case, reserve 17 of its 77 intents before training.
Treat those intents as unknown in calibration and test data.
Publish the selected labels before the first result.

This construction creates held-out intents. It does not reproduce naturally sampled unsupported customer traffic.

## Planned multilingual cohort: MASSIVE

[MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) supplies parallel intent data under CC BY 4.0.
Start with English, Russian, and Arabic.
Use a fixed multilingual encoder and the original dataset partitions.

Keep every translation of one example ID in the same split.
Report results separately by language and for mixed-language requests.
Add reviewed mixed-language requests as a separate group.

The current package uses a fixed English checkpoint.
Multilingual checkpoint selection and evaluation remain planned work.

The current dataset page reports an unsupported legacy loading script.
Use pinned official data files or an archive after its format and hash are examined.
Do not present an unexecuted loading command as a working recipe.

## User data

Use three separate JSONL files with stable IDs.
Set `group_id` for one conversation, paraphrase family, source item, or translated example.
Keep the entire group in one split.

Teacher decisions can supply training labels through `label_source: "teacher"`.
Review calibration and test labels independently.
Teacher agreement alone does not establish correctness.

The supplied support example uses authored text. It illustrates the API and routing workflow.
It is not an independent accuracy dataset.

## Leakage limits

TinyTask rejects repeated IDs, shared group IDs, and shared normalized text across splits.
Normalization uses Unicode NFKC, case folding, and whitespace folding.
Each split also needs unique normalized text.

The audit does not discover every paraphrase or translation relationship.
Supply group IDs for those relationships.
Calibration and test data currently accept at most one record per known group.
Grouped multilingual evaluation needs separate language reports or a later clustered interval method.
Semantic duplicate analysis is later work.

## Redistribution

The repository contains authored example data and derived benchmark reports.
It does not bundle the full public datasets or downloaded model weights.
Keep source licenses, citations, revisions, and hashes with each new cohort.
