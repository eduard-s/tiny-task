"""Train classifiers and export explicit numeric state."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize as unit_rows

from .data import Record

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True)
class TrainConfig:
    backend: str = "tfidf"
    seed: int = 42
    max_features: int = 20000
    neural_steps: int = 20
    allow_download: bool = False

    def __post_init__(self):
        if self.backend not in {"tfidf", "setfit"}:
            raise ValueError("The backend must be tfidf or setfit.")
        if self.max_features < 1 or self.neural_steps < 1:
            raise ValueError("Training sizes must be positive.")


def _array(value, *, ndim: int):
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError("Model arrays need finite values and the expected dimensions.")
    return array


class NumericHead:
    def __init__(self, labels: list[str], coef, intercept, prototypes):
        self.labels = labels
        self.coef = _array(coef, ndim=2)
        self.intercept = _array(intercept, ndim=1)
        self.prototypes = _array(prototypes, ndim=2)
        classes, dimensions = self.prototypes.shape
        rows = 1 if classes == 2 else classes
        if (
            classes < 2
            or len(set(labels)) != classes
            or len(labels) != classes
            or self.coef.shape != (rows, dimensions)
            or self.intercept.shape != (rows,)
        ):
            raise ValueError("Classifier arrays do not match the labels and feature dimensions.")

    @classmethod
    def fit(cls, features, records: list[Record], labels: list[str], seed: int):
        classifier = LogisticRegression(C=4.0, max_iter=1000, random_state=seed)
        classifier.fit(features, [r.label for r in records])
        if list(classifier.classes_) != labels:
            raise ValueError("Classifier label order differs from the task labels.")
        prototypes = []
        expected = np.asarray([r.label for r in records])
        for label in labels:
            mean = np.asarray(features[expected == label].mean(axis=0)).reshape(1, -1)
            prototypes.append(unit_rows(mean)[0])
        return cls(labels, classifier.coef_, classifier.intercept_, prototypes)

    def score(self, features):
        logits = np.asarray(features @ self.coef.T) + self.intercept
        if len(self.labels) == 2:
            positive = expit(logits[:, 0])
            probabilities = np.column_stack((1 - positive, positive))
        else:
            probabilities = softmax(logits, axis=1)
        indices = probabilities.argmax(axis=1)
        support = np.asarray(features @ self.prototypes.T)
        similarity = np.clip(support[np.arange(len(indices)), indices], -1, 1)
        return (
            np.asarray(self.labels)[indices],
            probabilities[np.arange(len(indices)), indices],
            similarity,
        )

    def state(self):
        return {
            "labels": self.labels,
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "prototypes": self.prototypes.tolist(),
        }


class TfidfBackend:
    name = "tfidf"

    @classmethod
    def fit(cls, records: list[Record], labels: list[str], config: TrainConfig):
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents="unicode",
            sublinear_tf=True,
            max_features=config.max_features,
        )
        features = vectorizer.fit_transform([r.text for r in records])
        return cls(vectorizer, NumericHead.fit(features, records, labels, config.seed))

    def __init__(self, vectorizer, head: NumericHead):
        self.vectorizer = vectorizer
        self.head = head

    def score(self, texts: list[str]):
        return self.head.score(self.vectorizer.transform(texts))

    def export(self, directory: Path):
        return {
            "head": self.head.state(),
            "vocabulary": {word: int(index) for word, index in self.vectorizer.vocabulary_.items()},
            "idf": self.vectorizer.idf_.tolist(),
        }

    @classmethod
    def load(cls, state: dict, directory: Path):
        vocabulary = state["vocabulary"]
        if not isinstance(vocabulary, dict) or sorted(vocabulary.values()) != list(
            range(len(vocabulary))
        ):
            raise ValueError("The model vocabulary needs contiguous unique indices.")
        vectorizer = TfidfVectorizer(
            vocabulary=vocabulary, ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True
        )
        idf = _array(state["idf"], ndim=1)
        head = NumericHead(**state["head"])
        if len(idf) != len(vocabulary) or head.coef.shape[1] != len(vocabulary):
            raise ValueError("The vocabulary and classifier dimensions differ.")
        vectorizer.idf_ = idf
        return cls(vectorizer, head)


class SetFitBackend:
    name = "setfit"

    def __init__(self, encoder, head: NumericHead, training_record=None):
        self.encoder = encoder
        self.head = head
        self.training_record = training_record

    @classmethod
    def fit(cls, records: list[Record], labels: list[str], config: TrainConfig):
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer
        from setfit import SetFitModel, Trainer, TrainingArguments

        encoder = SentenceTransformer(
            BASE_MODEL,
            revision=BASE_REVISION,
            local_files_only=not config.allow_download,
            trust_remote_code=False,
            device="cpu",
            model_kwargs={"use_safetensors": True},
        )
        model = SetFitModel(model_body=encoder, model_head=LogisticRegression(C=4.0, max_iter=1000))
        model.model_body.max_seq_length = 128

        def weights_digest():
            result = hashlib.sha256()
            for name, tensor in sorted(model.model_body.state_dict().items()):
                result.update(name.encode())
                result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return result.hexdigest()

        before = weights_digest()
        dataset = Dataset.from_dict(
            {"text": [r.text for r in records], "label": [labels.index(r.label) for r in records]}
        )
        with tempfile.TemporaryDirectory(prefix="tinytask-training-") as temporary:
            args = TrainingArguments(
                output_dir=temporary,
                batch_size=16,
                num_epochs=1,
                max_steps=config.neural_steps,
                num_iterations=5,
                seed=config.seed,
                report_to="none",
                save_strategy="no",
                show_progress_bar=False,
                logging_steps=5,
            )
            trainer = Trainer(model=model, args=args, train_dataset=dataset)
            trainer.train()
            training_record = {
                "initial_encoder_sha256": before,
                "trained_encoder_sha256": weights_digest(),
                "steps": trainer.st_trainer.state.global_step,
                "history": trainer.st_trainer.state.log_history,
                "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION,
                "training_device": str(model.model_body.device),
            }
        model.model_body.to("cpu")
        features = model.model_body.encode(
            [r.text for r in records], normalize_embeddings=True, show_progress_bar=False
        )
        prototypes = []
        expected = np.asarray([r.label for r in records])
        for label in labels:
            prototypes.append(unit_rows(features[expected == label].mean(axis=0).reshape(1, -1))[0])
        if list(model.model_head.classes_) != list(range(len(labels))):
            raise ValueError("The SetFit label order differs from the task labels.")
        head = NumericHead(labels, model.model_head.coef_, model.model_head.intercept_, prototypes)
        return cls(model.model_body, head, training_record)

    def score(self, texts: list[str]):
        features = self.encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return self.head.score(features)

    def truncated(self, texts: list[str]):
        tokens = self.encoder.tokenizer(texts, truncation=False, add_special_tokens=True)
        return np.asarray([len(ids) > self.encoder.max_seq_length for ids in tokens["input_ids"]])

    def export(self, directory: Path):
        self.encoder.save_pretrained(str(directory / "encoder"), safe_serialization=True)
        return {
            "head": self.head.state(),
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "training_record": self.training_record,
        }

    @classmethod
    def load(cls, state: dict, directory: Path):
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            str(directory / "encoder"), local_files_only=True, trust_remote_code=False, device="cpu"
        )
        return cls(encoder, NumericHead(**state["head"]), state.get("training_record"))


BACKENDS: dict[str, Any] = {"tfidf": TfidfBackend, "setfit": SetFitBackend}


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
