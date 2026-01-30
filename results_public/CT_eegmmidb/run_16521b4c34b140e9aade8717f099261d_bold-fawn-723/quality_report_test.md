# EEGMMIDB Quality Report (test)

- record_level_agg: mean_logit
- subject_level_agg: mean_prob

## Window level
- samples: 18479
- accuracy: 0.7634071107743926
- balanced_acc: 0.7930642349324732
- macro_f1: 0.7767550291716919
- mean_max_prob: 0.8538089394569397
- brier: 0.3490682381847092

## Record level
- samples: 308
- accuracy: 0.788961038961039
- balanced_acc: 0.807414858400962
- macro_f1: 0.8007366075009582
- mean_max_prob: 0.728500247001648
- brier: 0.31969171690987885

## Subject level
- samples: 22
- accuracy: 0.7727272727272727
- balanced_acc: 0.8333333333331167
- macro_f1: 0.8222222222215296
- mean_max_prob: 0.6623733043670654
- brier: 0.3961322999676635

## Baseline comparison (record-level)
| model | record_macro_f1 | record_balanced_acc |
| --- | --- | --- |
| CT (this run) | 0.8007366075009582 | 0.807414858400962 |
| csp_lda | 0.31770130454340983 | 0.3958553791887125 |
| simple_cnn | 0.8225358509881344 | 0.820695256660169 |
| shallowconv | 0.5351897449842408 | 0.5457904019307528 |
| eegnet | 0.7345202660493148 | 0.7358163928339367 |
