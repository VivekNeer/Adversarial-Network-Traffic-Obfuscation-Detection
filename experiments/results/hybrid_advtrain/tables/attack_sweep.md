| attack | accuracy | macro_f1 | obfuscated_recall | malicious_recall | false_positive_rate | evasion_rate | stats_linf | seq_linf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.9650 | 0.9649 | 0.9424 | 0.9879 | 0.0448 | 0.0000 | 0.0000 | 0.0000 |
| fgsm both eps=0.05 constrained | 0.9525 | 0.9523 | 0.9258 | 0.9814 | 0.0654 | 0.0186 | 0.0500 | 0.0500 |
| fgsm both eps=0.1 constrained | 0.9405 | 0.9403 | 0.9076 | 0.9758 | 0.0801 | 0.0242 | 0.1000 | 0.1000 |
| pgd both eps=0.05 steps=10 constrained | 0.9525 | 0.9523 | 0.9258 | 0.9814 | 0.0654 | 0.0186 | 0.0500 | 0.0500 |
| pgd both eps=0.1 steps=10 constrained | 0.9375 | 0.9372 | 0.9030 | 0.9750 | 0.0845 | 0.0250 | 0.1000 | 0.1000 |
| pgd both eps=0.2 steps=20 constrained | 0.8378 | 0.8387 | 0.8038 | 0.9364 | 0.1433 | 0.0636 | 0.2000 | 0.2000 |
| pgd both eps=0.1 steps=10 unconstrained | 0.9108 | 0.9104 | 0.8598 | 0.9530 | 0.1051 | 0.0470 | 0.1000 | 0.1000 |
| pgd both eps=0.1 steps=10 constrained targeted | 0.9620 | 0.9617 | 0.9182 | 0.9750 | 0.0301 | 0.0250 | 0.1000 | 0.1000 |
