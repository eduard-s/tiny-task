# Recorded development results

These results describe one seed and one CLINC150 development cohort.
They do not establish production accuracy or broad superiority.

| Backend | Policy | Known accuracy before abstention | Known coverage | Accepted accuracy | Unknown accepted | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| tfidf | always_predict | 88.7% | 100.0% | 18.3% | 1150/1150 | reject |
| tfidf | confidence_only | 88.7% | 62.0% | 85.2% | 30/1150 | reject |
| tfidf | tinytask | 88.7% | 62.0% | 85.2% | 30/1150 | reject |
| setfit | always_predict | 92.0% | 100.0% | 19.0% | 1150/1150 | reject |
| setfit | confidence_only | 92.0% | 89.0% | 79.1% | 59/1150 | reject |
| setfit | tinytask | 92.0% | 94.0% | 88.7% | 19/1150 | reject |

The test set contains 300 known and 1,150 unknown requests.
Every accepted unknown request counts as an error.
Known accuracy describes the classifier. Accepted accuracy also depends on the unknown request mixture.

| Backend | Policy | Related unknown accepted | Broader unknown accepted |
| --- | --- | ---: | ---: |
| tfidf | always_predict | 150/150 | 1000/1000 |
| tfidf | confidence_only | 6/150 | 24/1000 |
| tfidf | tinytask | 6/150 | 24/1000 |
| setfit | always_predict | 150/150 | 1000/1000 |
| setfit | confidence_only | 15/150 | 44/1000 |
| setfit | tinytask | 11/150 | 8/1000 |

The TF-IDF policies select the same accepted test requests.
The SetFit support threshold reduces unknown acceptance in this run.
Its accepted accuracy lower bound remains less than 90%, so the gate rejects the candidate.

The live teacher, Prompt2Model workflow, and ONNX service are not measured.
Read [the protocol](README.md) and the JSON records before an accuracy or cost claim.
