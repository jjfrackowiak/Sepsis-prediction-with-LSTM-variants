# Sepsis-prediction-with-LSTM-variants
Continuation of https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN. Repository dedicated for two currently developed implementations of two variants of LSTM-based neural net* on the task of early sepsis prediction (subject of Physionet challange 2019).
\* Bidirectional LSTM (BDLSTM) and BiDirectional LSTM with additional convolutional layers (CNN-BDLSTM) and hidden_layer of size 64. 

In case of CNN-LSTM, following hyperparameters values were selected: (lookback=10, filters_1 = filters2 = filters3 = 64, dropout=0.3). Choosing filters to be 128 did not increase any of the performance metrics.
Both models were trained using a sliding window approach, predicting one step forward based on the most recent 13 steps.

### Training results:

| Metric in Validation | BDLSTM | CNN-BDLSTM |
| ------------- | ------------- | ------------- |
| Balanced Accuracy (cut-off at 0.3) | 0.69699 | 0.70610 |
| AUC Score | 0.79287  | 0.80023 |
| Weighted Loss* | 0.18424  | 0.10105 |
| Positive class frequency | 4563/301643 |

\* Because of severe class imbalance nn.BCEWithLogitsLoss() with pos_weight = 10 was utilised. Weightening may cause the loss value to be inflated and not interpretable in absolute terms.

#### Loss throughout 10 epochs for training (gray) and validation (blue):
<img width="900" alt="Loss" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/dec195ac-1ec0-4081-8176-a3b96eb3e5ce">


#### Loss throughout 10 epochs for training (gray) and validation (blue):
<img width="900" alt="Loss" src="https://github.com/jjfrackowiak/Sepsis-prediction-with-TCN/assets/84077365/dec195ac-1ec0-4081-8176-a3b96eb3e5ce">



#### ROC curve for CNN-LSTM (validation set):
![image](https://github.com/jjfrackowiak/Sepsis-prediction-with-LSTM-variants/assets/84077365/58c45096-31c6-4618-ad03-2894ccc5c160)