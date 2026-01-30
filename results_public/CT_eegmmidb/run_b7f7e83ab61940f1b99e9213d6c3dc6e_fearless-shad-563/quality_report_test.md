# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18479
- accuracy: 0.8348936630770063
- balanced_acc: 0.8224024610620969
- macro_f1: 0.7995133996310321
- mean_max_prob: 0.8582974076271057
- brier: 0.23396022017872453

## Record level
- samples: 308
- accuracy: 0.8896103896103896
- balanced_acc: 0.8929892429332394
- macro_f1: 0.8525771582699196
- mean_max_prob: 0.7587305903434753
- brier: 0.20356051156187796

## Subject level
- samples: 22
- accuracy: 0.9545454545454546
- balanced_acc: 0.9761904761902319
- macro_f1: 0.9209876543203085
- mean_max_prob: 0.7222155332565308
- brier: 0.19864882983876112

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.8525771582699196 | 0.8929892429332394 |
| csp_lda | 0.3680381412340175 | 0.4801740812379111 |
| simple_cnn | 0.7314414877887275 | 0.728464354794142 |
| shallowconv | 0.554983367761582 | 0.5621776273372019 |
| eegnet | 0.6783140379978291 | 0.6811158699456573 |
