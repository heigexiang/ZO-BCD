## softprompt zo-Adam grid search result (1 epoch)

- baseline:
    - train_loss:0.23392285406589508
    - train_accuracy:0.4260506331920624
    - test_loss:0.29128023982048035
    - test_acc:0.3018292486667633
- learning_rate=1e-05, variation_scale=0.1:
    - train_loss:0.2401987463235855
    - train_accuracy:0.4212014973163605
    - test_loss:0.28256022930145264
    - test_acc:0.3132621943950653
    - training_curve:![alt text](images/image.png)
- learning_rate=1e-05, variation_scale=0.01:
    - train_loss:0.2411266714334488
    - train_accuracy:0.41945043206214905
    - test_loss:0.28711965680122375
    - test_acc:0.29649388790130615
    - training_curve:![alt text](images/image-1.png)
- learning_rate=1e-05, variation_scale=0.001:
    - train_loss:0.23280642926692963
    - train_accuracy:0.42416489124298096
    - test_loss:0.2818285822868347
    - test_acc:0.30564025044441223
    - training_curve:![alt text](images/image-2.png)
- learning_rate=1e-06, variation_scale=0.1:
    - train_loss:0.23667888343334198
    - train_accuracy:0.43305495381355286
    - test_loss:0.2896467447280884
    - test_acc:0.291158527135849
    - training_curve:![alt text](images/image-4.png)
- learning_rate=1e-06, variation_scale=0.01:
    - train_loss:0.2403506636619568
    - train_accuracy:0.4240301847457886
    - test_loss:0.28388768434524536
    - test_acc:0.3147865831851959
    - training_curve:![alt text](images/image-5.png)
- learning_rate=1e-06, variation_scale=0.001:
    - train_loss:0.24416998028755188
    - train_accuracy:0.41931572556495667
    - test_loss:0.2891397774219513
    - test_acc:0.32698169350624084
    - training_curve:![alt text](images/image-6.png)
- learning_rate=1e-07, variation_scale=0.1:
    - train_loss:0.24101512134075165
    - train_accuracy:0.4170258641242981
    - test_loss:0.2920945882797241
    - test_acc:0.3185975253582001
    - training_curve:![alt text](images/image-7.png)
- learning_rate=1e-07, variation_scale=0.01:
    - train_loss:0.23846426606178284
    - train_accuracy:0.4237607717514038
    - test_loss:0.29296863079071045
    - test_acc:0.31707319617271423
    - training_curve:![alt text](images/image-8.png)
- learning_rate=1e-07, variation_scale=0.001:
    - train_loss:0.24343854188919067
    - train_accuracy:0.41918104887008667
    - test_loss:0.2916679084300995
    - test_acc:0.3102134168148041
    - training_curve:![alt text](images/image-9.png)
<!-- table for comparing -->
| learning_rate | variation_scale | train_loss | train_accuracy | test_loss | test_acc |
|---------------|-----------------|------------|----------------|-----------|----------|
| Baseline | Baseline | 0.23392285406589508 | 0.4260506331920624 | 0.29128023982048035 | 0.3018292486667633 |
| 1e-05         | 0.1             | 0.2401987463235855 | 0.4212014973163605 | 0.28256022930145264 | 0.3132621943950653 |
| 1e-05         | 0.01            | 0.2411266714334488 | 0.41945043206214905 | 0.28711965680122375 | 0.29649388790130615 |
| 1e-05         | 0.001           | 0.23280642926692963 | 0.42416489124298096 | 0.2818285822868347 | 0.30564025044441223 |
| 1e-06         | 0.1             | 0.23667888343334198 | 0.43305495381355286 | 0.2896467447280884 | 0.291158527135849 |
| 1e-06         | 0.01            | 0.2403506636619568 | 0.4240301847457886 | 0.28388768434524536 | 0.3147865831851959 |
| 1e-06         | 0.001           | 0.24416998028755188 | 0.41931572556495667 | 0.2891397774219513 | 0.32698169350624084 |
| 1e-07         | 0.1             | 0.24101512134075165 | 0.4170258641242981 | 0.2920945882797241 | 0.3185975253582001 |
| 1e-07         | 0.01            | 0.23846426606178284 | 0.4237607717514038 | 0.29296863079071045 | 0.31707319617271423 |
| 1e-07         | 0.001           | 0.24343854188919067 | 0.41918104887008667 | 0.2916679084300995 | 0.3102134168148041 |
<!-- end of table -->
