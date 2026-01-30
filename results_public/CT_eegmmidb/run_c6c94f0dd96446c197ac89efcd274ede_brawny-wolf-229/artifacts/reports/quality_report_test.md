# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18479
- accuracy: 0.18880891823150603
- balanced_acc: 0.33333333333333326
- macro_f1: 0.10588128186444114
- mean_max_prob: 0.4314505159854889
- brier: 0.7263392474221412

## Record level
- samples: 308
- accuracy: 0.14285714285714285
- balanced_acc: 0.33333333333332577
- macro_f1: 0.08333333333325993
- mean_max_prob: 0.4314506947994232
- brier: 0.7396185280879193

## Subject level
- samples: 22
- accuracy: 0.09090909090909091
- balanced_acc: 0.33333333333316667
- macro_f1: 0.05555555555549999
- mean_max_prob: 0.43145057559013367
- brier: 0.7549694821953566

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.08333333333325993 | 0.33333333333332577 |
| csp_lda | 0.3680381412340175 | 0.4801740812379111 |
| simple_cnn | 0.7314414877887275 | 0.728464354794142 |
| shallowconv | 0.554983367761582 | 0.5621776273372019 |
| eegnet | 0.6783140379978291 | 0.6811158699456573 |
