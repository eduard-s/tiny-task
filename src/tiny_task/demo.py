"""A small, authored support inbox example."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .backends import TrainConfig, write_json
from .data import UNKNOWN, Record
from .policy import PolicyConfig
from .task import TinyTask

DEMO_TEXTS = {
    "cancel": [
        "Cancel my subscription please",
        "I want to cancel the monthly plan",
        "Please cancel my account renewal",
        "Stop renewing my subscription",
        "Cancel the annual subscription",
        "Please end my paid subscription",
        "I need to cancel the service",
        "Cancel my recurring membership",
        "I no longer need this subscription",
        "Please cancel the next renewal",
        "Stop my monthly membership renewal",
        "Close my subscription plan",
        "Can you cancel the subscription today",
        "I want my annual plan cancelled",
        "Please stop my subscription renewal",
        "Cancel the membership before renewal",
        "End the monthly subscription please",
        "How do I cancel this membership",
        "Please cancel my paid plan now",
        "Stop the annual subscription renewal",
        "I need my membership cancelled",
        "Cancel the service subscription today",
        "Cancel my subscription before Friday",
        "Please cancel the recurring plan",
    ],
    "refund": [
        "I need a refund for this order",
        "Please refund the duplicate charge",
        "Refund my last payment",
        "I want my payment refunded",
        "Please return my money",
        "Can I get a refund for the purchase",
        "The item was broken and I want a refund",
        "Please refund the incorrect payment",
        "Return the money from my order",
        "I need a refund for a damaged item",
        "Please process a purchase refund",
        "I paid twice and need a refund",
        "Please refund the recent order",
        "I want the duplicate payment refunded",
        "Can you return my last payment",
        "I need the purchase money refunded",
        "Please refund this damaged purchase",
        "Refund the charge from yesterday",
        "I want a refund for order twenty",
        "Please return money for my purchase",
        "Refund my accidental duplicate order",
        "Please refund the failed purchase",
        "My purchase needs a refund",
        "Can you refund the last charge",
    ],
    "shipping": [
        "Where is my delivery",
        "Track my shipping status",
        "My parcel has not arrived",
        "When will my order arrive",
        "Please send the tracking number",
        "The delivery is late",
        "I need an update on my shipment",
        "Where is the package now",
        "My shipping tracking has stopped",
        "Please track the parcel",
        "What is the delivery date",
        "The shipment is delayed",
        "Where is my late delivery",
        "Please update my parcel tracking",
        "When will this shipment arrive",
        "I need the package tracking number",
        "The parcel delivery is overdue",
        "Can you track this shipping order",
        "My package still has not arrived",
        "Please send a shipment update",
        "What happened to my parcel delivery",
        "Track the package for me please",
        "When is the delayed delivery due",
        "My order tracking needs an update",
    ],
    "password": [
        "I forgot my password",
        "Please reset my login password",
        "My password reset email never arrived",
        "I cannot sign in to my account",
        "How do I change my password",
        "The login page rejects my password",
        "Please help me reset the password",
        "I lost my account password",
        "The password reset link expired",
        "I need a new login password",
        "My account login is locked",
        "Please resend the password reset email",
        "I need to reset my forgotten password",
        "My login password does not work",
        "Please send a new password reset link",
        "The account sign in is locked",
        "Help me change this login password",
        "The reset password message is missing",
        "Please reset the password for my account",
        "I cannot log in with my password",
        "My password reset token has expired",
        "Send me another password reset email",
        "Help with a forgotten login password",
        "I need the account password changed",
    ],
}
UNKNOWN_TEXTS = [
    "What is the weather tomorrow",
    "Tell me a joke about penguins",
    "How do I bake a carrot cake",
    "Translate hello into Japanese",
    "Recommend a science fiction novel",
    "What is the capital of Iceland",
    "Explain the orbit of Neptune",
    "Write a poem about autumn",
    "How do bees make honey",
    "Find a recipe for lentil soup",
    "Can you explain photosynthesis",
    "Who painted the night sky",
    "Suggest a name for my new puppy",
    "How deep is the Pacific ocean",
    "Teach me a chord on the guitar",
    "What is the square root of nine",
    "Tell me about volcanic eruptions",
    "How does a butterfly grow",
    "Which bird flies the farthest",
    "Explain why the sky is blue",
]


def demo_records():
    splits: dict[str, list[Record]] = {"train": [], "calibration": [], "test": []}
    for label, texts in DEMO_TEXTS.items():
        for index, text in enumerate(texts):
            split = "train" if index < 12 else ("calibration" if index < 18 else "test")
            splits[split].append(
                Record(f"demo-{label}-{index}", text, label, label_source="authored")
            )
    for index, text in enumerate(UNKNOWN_TEXTS):
        split = "calibration" if index < 8 else "test"
        splits[split].append(
            Record(f"demo-unknown-{index}", text, UNKNOWN, label_source="authored")
        )
    return splits


def run_demo(output: str | Path, *, backend: str = "tfidf", allow_download: bool = False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits = demo_records()
    task = TinyTask.fit(
        splits["train"],
        splits["calibration"],
        config=TrainConfig(backend=backend, allow_download=allow_download),
        policy_config=PolicyConfig(min_calibration_accepted=8),
    )
    task.save(output / "model")
    task = TinyTask.load(output / "model")
    report = task.evaluate(splits["test"])
    task.write_report(output / "report.json", report)
    messages = [
        "Please cancel my subscription",
        "I need a refund for a broken purchase",
        "Where is my package delivery",
        "Please reset my forgotten password",
        "Write a haiku about a sleepy hamster",
    ]
    predictions = task.predict(messages)
    rows = [
        dict(id=f"ticket-{i}", text=text, **prediction)
        for i, (text, prediction) in enumerate(zip(messages, predictions, strict=True), 1)
    ]
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    write_json(
        output / "data-provenance.json",
        {
            "source": "Authored example data in tiny_task.demo. This is not a public benchmark.",
            "rows": {split: len(records) for split, records in splits.items()},
        },
    )
    body = "".join(
        f"<tr><td>{html.escape(row['text'])}</td><td>{html.escape(row['label'] or 'review')}</td>"
        f"<td>{row['class_probability']:.3f}</td><td>{html.escape(row['reason'])}</td></tr>"
        for row in rows
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>TinyTask support inbox</title><style>"
        "body{font:17px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#20312b}"
        "table{border-collapse:collapse;width:100%}td,th{padding:1rem;text-align:left;"
        "border-bottom:1px solid #cdd6d0}h1{font-size:2rem}.table{overflow:auto}"
        "</style><h1>TinyTask support inbox</h1>"
        "<p>A local classifier routes candidate messages. Uncertain messages need review.</p>"
        "<p>This authored example shows the workflow. "
        "It does not establish production accuracy.</p>"
        f"<p>Release gate: <strong>{html.escape(report['gate']['status'])}</strong>.</p>"
        '<div class="table"><table><tr><th>Message</th><th>Candidate route</th>'
        f"<th>Class score</th><th>Reason</th></tr>{body}</table></div>"
        "<p>Class scores are not calibrated probabilities of correctness.</p>"
        "<p>Open report.json for test errors, coverage, intervals, and release criteria.</p></html>"
    )
    return {"output": str(output), "gate": report["gate"]["status"], "predictions": rows}
