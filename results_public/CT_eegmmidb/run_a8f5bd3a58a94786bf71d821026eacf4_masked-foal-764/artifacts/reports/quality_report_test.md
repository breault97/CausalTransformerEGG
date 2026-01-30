# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18478
- accuracy: 0.7978136161922286
- balanced_acc: 0.8018010279532938
- macro_f1: 0.8061112262199593
- mean_max_prob: 0.8405562043190002
- brier: 0.2822271633660302

## Record level
- samples: 308
- accuracy: 0.8181818181818182
- balanced_acc: 0.8234515014175948
- macro_f1: 0.830396012829771
- mean_max_prob: 0.704039454460144
- brier: 0.2616496730051212

## Subject level
- samples: 22
- accuracy: 0.9545454545454546
- balanced_acc: 0.9444444444443149
- macro_f1: 0.9500891265590831
- mean_max_prob: 0.637203574180603
- brier: 0.2696418141033146

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.830396012829771 | 0.8234515014175948 |
| csp_lda | 0.4261715296198054 | 0.5529100529100529 |
| simple_cnn | 0.6903910611300849 | 0.7247023809523809 |
| shallowconv | 0.6312489822504478 | 0.660218253968254 |
| eegnet | 0.730961850527068 | 0.7602513227513228 |
