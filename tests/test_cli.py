import json

from tiny_task.cli import main
from tiny_task.demo import run_demo


def test_cli_train_evaluate_predict(tmp_path, capsys):
    model = tmp_path / "model"
    assert (
        main(
            [
                "train",
                "--train",
                "examples/support/train.jsonl",
                "--calibration",
                "examples/support/calibration.jsonl",
                "--output",
                str(model),
                "--min-calibration-accepted",
                "8",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["predict", "--model", str(model), "--text", "Please cancel my subscription"]) == 0
    assert json.loads(capsys.readouterr().out)["label"] == "cancel"
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "evaluate",
                "--model",
                str(model),
                "--test",
                "examples/support/test.jsonl",
                "--output",
                str(report),
            ]
        )
        == 3
    )
    capsys.readouterr()
    assert (
        main(
            [
                "predict",
                "--model",
                str(model),
                "--input",
                "examples/inbox.csv",
                "--output",
                str(tmp_path / "predictions.jsonl"),
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert rows[-1]["decision"] == "review"


def test_cli_errors_are_json_and_preserve_output(task, tmp_path, capsys):
    output = tmp_path / "keep"
    output.write_text("keep")
    assert (
        main(
            ["predict", "--model", str(task.directory), "--text", "cancel", "--output", str(output)]
        )
        == 2
    )
    assert "error" in json.loads(capsys.readouterr().err)
    assert output.read_text() == "keep"


def test_demo_is_honest_and_offline(tmp_path):
    result = run_demo(tmp_path / "demo")
    assert result["gate"] == "needs_data"
    assert result["predictions"][-1]["decision"] == "review"
    assert "not establish production accuracy" in (tmp_path / "demo/index.html").read_text()


def test_jsonl_batch(task, tmp_path, capsys):
    data = tmp_path / "inbox.jsonl"
    data.write_text('{"id":"one","text":"Please refund my payment"}\n')
    assert main(["predict", "--model", str(task.directory), "--input", str(data)]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "one"
