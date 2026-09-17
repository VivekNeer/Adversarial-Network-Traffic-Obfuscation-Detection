| attack | accuracy | macro_f1 | obfuscated_recall | malicious_recall | false_positive_rate | evasion_rate | stats_linf | seq_linf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.9618 | 0.9616 | 0.9311 | 0.9826 | 0.0448 | 0.0000 | 0.0000 | 0.0000 |
| fgsm stats eps=0.05 constrained | 0.9038 | 0.9038 | 0.8674 | 0.9587 | 0.0830 | 0.0413 | 0.0500 | 0.0000 |
| fgsm stats eps=0.1 constrained | 0.6436 | 0.6190 | 0.7758 | 0.9197 | 0.1256 | 0.0803 | 0.1000 | 0.0000 |
| pgd stats eps=0.05 steps=10 constrained | 0.8940 | 0.8942 | 0.8636 | 0.9583 | 0.0852 | 0.0417 | 0.0500 | 0.0000 |
| pgd stats eps=0.1 steps=10 constrained | 0.5884 | 0.5456 | 0.7485 | 0.9152 | 0.1403 | 0.0848 | 0.1000 | 0.0000 |
| pgd stats eps=0.2 steps=20 constrained | 0.3572 | 0.3177 | 0.4220 | 0.8057 | 0.3593 | 0.1943 | 0.2000 | 0.0000 |
| pgd stats eps=0.1 steps=10 unconstrained | 0.5714 | 0.5209 | 0.7364 | 0.9102 | 0.1411 | 0.0898 | 0.1000 | 0.0000 |
| pgd stats eps=0.1 steps=10 constrained targeted | 0.8488 | 0.8458 | 0.8106 | 0.9019 | 0.0059 | 0.0981 | 0.1000 | 0.0000 |
