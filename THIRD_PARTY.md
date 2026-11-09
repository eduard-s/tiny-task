# Third-party notices

TinyTask code and authored support examples use the MIT license in this repository.

- [SetFit](https://github.com/huggingface/setfit) supplies the optional training method under Apache 2.0.
- [Sentence Transformers](https://github.com/huggingface/sentence-transformers) supplies the encoder interface under Apache 2.0.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) supplies the fixed pretrained checkpoint under Apache 2.0.
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) supplies the linear classifier and text features under BSD 3-Clause.
- NumPy, SciPy, and PyTorch keep their upstream licenses.

Downloaded weights are not included in the package or Git repository.
Keep the source checkpoint license with redistributed derived models.

The benchmark uses [CLINC150](https://github.com/clinc/oos-eval), from Larson and colleagues, 2019, under CC BY 3.0.
The preparation code selects a subset, groups unsupported intents, and examines normalized duplicates.
The full source data is downloaded separately.
The source license text is in `benchmarks/CLINC_LICENSE.txt`.

BANKING77 and MASSIVE are planned evaluation sources under their stated CC BY 4.0 terms.
No result from either dataset is claimed in the current benchmark.

Prompt2Model is a compared project. Its code is not copied into TinyTask.
