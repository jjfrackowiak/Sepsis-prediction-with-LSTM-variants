# Sepsis-prediction-with-LSTM-variants
[Continuation of https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN] 

Repository dedicated for currently developed implementations of two variants of LSTM* on the task of early sepsis prediction (subject of Physionet challange 2019).

\* Bidirectional LSTM (BDLSTM) and BiDirectional LSTM with additional convolutional layers (CNN-BDLSTM). 

In case of CNN-LSTM, following hyperparameters values were selected: (lookback=10, filters_1 = filters2 = filters3 = 64, dropout=0.3). Choosing filters size equal to 128 did not increase any of the performance metrics.
Both models were trained using a sliding window approach, predicting one step forward based on 13 most recent steps with a hidden layer of size 64.

### Training results:

| Metric in Validation | BDLSTM | CNN-BDLSTM |
| ------------- | ------------- | ------------- |
| Balanced Accuracy (cut-off at 0.3) | 0.69699 | 0.70610 |
| AUC Score | 0.79287  | 0.80023 |
| Weighted Loss* | 0.432 | 0.434 |

\* Because of severe class imbalance (4563 positives for 301643 observations in test set) nn.BCEWithLogitsLoss() with pos_weight = 10 was utilised. Weightening may cause the loss value to be inflated and not interpretable in absolute terms.

#### Loss throughout 10 epochs for BDLSTM:
![image](https://github.com/jjfrackowiak/Sepsis-prediction-with-LSTM-variants/assets/84077365/35674c75-6416-48ec-94c2-5612bf2e16c2)

#### Loss throughout 10 epochs for CNN-BDLSTM:
![image](https://github.com/jjfrackowiak/Sepsis-prediction-with-LSTM-variants/assets/84077365/3918e186-ddbc-4c82-9b0b-5cc1e4e8ce96)

#### ROC curve for CNN-LSTM (validation set):
![image](https://github.com/jjfrackowiak/Sepsis-prediction-with-LSTM-variants/assets/84077365/58c45096-31c6-4618-ad03-2894ccc5c160)
