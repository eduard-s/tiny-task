# A useful support inbox demo

## User task

A developer needs candidate routes for a CSV of support messages.
The local model selects cancellation, refund, delivery, or password labels.
Unsupported messages enter a review queue.

The first run needs no API key or pretrained model download.
It trains a TF-IDF classifier from the supplied authored records.

## Run the example

From the repository root, run these commands:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/tiny-task demo --output artifacts/demo
```

Open `artifacts/demo/index.html` in a browser.
Read `report.json` for the test errors, coverage, and release criteria.

The expected result is a useful candidate demonstration with `needs_data` release status.
The small example cannot establish the default accuracy target.

## Use your inbox

Create a CSV with these columns:

```csv
id,text
ticket-1,Please cancel my subscription
ticket-2,Where is my package delivery
ticket-3,Write a haiku about a sleepy hamster
```

Run the prediction command:

```sh
.venv/bin/tiny-task predict --model artifacts/demo/model \
  --input examples/inbox.csv --output artifacts/inbox-results.jsonl
```

Join the result IDs with the original inbox.
Use `decision` to separate candidate routes from review requests.
The command does not contact a ticket system or move a message.

The example model knows only the supplied labels and vocabulary.
Train and evaluate a new candidate before a real routing workflow.

## Show the AI work

Install the neural dependencies and use `--backend setfit --allow-download` for training.
The recorded benchmark shows actual encoder fine-tuning and its evaluation.
Read the source revision, changed encoder hashes, step count, and loss history in the benchmark JSON.

The default learner also trains a real classifier.
It provides a fast baseline for the neural comparison.

## Proposed user trial

Give this workflow to three teams with repeated classification tasks.
Use their own reviewed labels and unsupported examples.
Keep the data local.

Record time to first prediction, setup failures, accepted errors, review volume, and repeat use after one week.
The target is two repeat users and useful feedback on the remaining team's obstacles.
No adoption result is claimed yet.
