| attack | accuracy | macro_f1 | obfuscated_recall | malicious_recall | false_positive_rate | evasion_rate | stats_linf | seq_linf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.9300 | 0.9295 | 0.8909 | 0.9848 | 0.1000 | 0.0000 | 0.0000 | 0.0000 |
| fgsm seq eps=0.05 constrained | 0.6020 | 0.5674 | 0.8606 | 0.9697 | 0.2294 | 0.0303 | 0.0000 | 0.0500 |
| fgsm seq eps=0.1 constrained | 0.5080 | 0.4503 | 0.8303 | 0.9545 | 0.3529 | 0.0455 | 0.0000 | 0.1000 |
| pgd seq eps=0.05 steps=10 constrained | 0.5700 | 0.5278 | 0.8606 | 0.9697 | 0.2765 | 0.0303 | 0.0000 | 0.0500 |
| pgd seq eps=0.1 steps=10 constrained | 0.4860 | 0.4198 | 0.8182 | 0.9515 | 0.3706 | 0.0485 | 0.0000 | 0.1000 |
| pgd seq eps=0.2 steps=20 constrained | 0.4180 | 0.3723 | 0.7030 | 0.9030 | 0.4529 | 0.0970 | 0.0000 | 0.2000 |
| pgd seq eps=0.1 steps=10 unconstrained | 0.3500 | 0.3211 | 0.4970 | 0.8848 | 0.4529 | 0.1152 | 0.0000 | 0.1000 |
| pgd seq eps=0.1 steps=10 constrained targeted | 0.6960 | 0.6586 | 0.8909 | 0.9455 | 0.0588 | 0.0545 | 0.0000 | 0.1000 |
