| attack | accuracy | macro_f1 | obfuscated_recall | malicious_recall | false_positive_rate | evasion_rate | stats_linf | seq_linf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.9410 | 0.9404 | 0.8856 | 0.9792 | 0.0602 | 0.0000 | 0.0000 | 0.0000 |
| fgsm seq eps=0.05 constrained | 0.7718 | 0.7701 | 0.8636 | 0.9678 | 0.1117 | 0.0322 | 0.0000 | 0.0500 |
| fgsm seq eps=0.1 constrained | 0.5579 | 0.4889 | 0.8371 | 0.9553 | 0.2079 | 0.0447 | 0.0000 | 0.1000 |
| pgd seq eps=0.05 steps=10 constrained | 0.7376 | 0.7362 | 0.8576 | 0.9652 | 0.1580 | 0.0348 | 0.0000 | 0.0500 |
| pgd seq eps=0.1 steps=10 constrained | 0.5119 | 0.4476 | 0.8083 | 0.9428 | 0.3057 | 0.0572 | 0.0000 | 0.1000 |
| pgd seq eps=0.2 steps=20 constrained | 0.3927 | 0.3393 | 0.6735 | 0.8837 | 0.4989 | 0.1163 | 0.0000 | 0.2000 |
| pgd seq eps=0.1 steps=10 unconstrained | 0.3809 | 0.3425 | 0.5136 | 0.8644 | 0.3880 | 0.1356 | 0.0000 | 0.1000 |
| pgd seq eps=0.1 steps=10 constrained targeted | 0.7918 | 0.7843 | 0.8470 | 0.9394 | 0.0265 | 0.0606 | 0.0000 | 0.1000 |
