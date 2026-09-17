| attack | accuracy | macro_f1 | obfuscated_recall | malicious_recall | false_positive_rate | evasion_rate | stats_linf | seq_linf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.9688 | 0.9686 | 0.9424 | 0.9860 | 0.0323 | 0.0000 | 0.0000 | 0.0000 |
| fgsm both eps=0.05 constrained | 0.9203 | 0.9204 | 0.8992 | 0.9686 | 0.0661 | 0.0314 | 0.0500 | 0.0500 |
| fgsm both eps=0.1 constrained | 0.7846 | 0.7847 | 0.8402 | 0.9458 | 0.1183 | 0.0542 | 0.1000 | 0.1000 |
| pgd both eps=0.05 steps=10 constrained | 0.9110 | 0.9114 | 0.8977 | 0.9689 | 0.0676 | 0.0311 | 0.0500 | 0.0500 |
| pgd both eps=0.1 steps=10 constrained | 0.7276 | 0.7252 | 0.8068 | 0.9345 | 0.1411 | 0.0655 | 0.1000 | 0.1000 |
| pgd both eps=0.2 steps=20 constrained | 0.3967 | 0.3795 | 0.5803 | 0.8587 | 0.5202 | 0.1413 | 0.2000 | 0.2000 |
| pgd both eps=0.1 steps=10 unconstrained | 0.6483 | 0.6424 | 0.6902 | 0.8814 | 0.1675 | 0.1186 | 0.1000 | 0.1000 |
| pgd both eps=0.1 steps=10 constrained targeted | 0.9033 | 0.9024 | 0.8447 | 0.9295 | 0.0110 | 0.0705 | 0.1000 | 0.1000 |
