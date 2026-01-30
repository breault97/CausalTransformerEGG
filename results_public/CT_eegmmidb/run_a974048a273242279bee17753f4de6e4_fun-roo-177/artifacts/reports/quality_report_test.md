# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18479
- accuracy: 0.7831592618648195
- balanced_acc: 0.8010730766836728
- macro_f1: 0.7603080865798303
- mean_max_prob: 0.8589122295379639
- brier: 0.3082218817561215

## Record level
- samples: 308
- accuracy: 0.8116883116883117
- balanced_acc: 0.8576979198479636
- macro_f1: 0.7870915718335477
- mean_max_prob: 0.752423107624054
- brier: 0.27007030594979575

## Subject level
- samples: 22
- accuracy: 0.8636363636363636
- balanced_acc: 0.9285714285711877
- macro_f1: 0.8171428571422568
- mean_max_prob: 0.7021145224571228
- brier: 0.2653539685210674

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.7870915718335477 | 0.8576979198479636 |
| csp_lda | 0.3680381412340175 | 0.4801740812379111 |
| simple_cnn | 0.7314414877887275 | 0.728464354794142 |
| shallowconv | 0.554983367761582 | 0.5621776273372019 |
| eegnet | 0.6783140379978291 | 0.6811158699456573 |
