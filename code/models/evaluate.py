
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split


from models import transformer_classifier
from prepare_data import load_data, load_label
from tqdm import tqdm
from collections import Counter

from pathlib import Path



def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


if __name__ == "__main__":

    root_folder = Path(__file__).parents[2]

    transformer_h5 = root_folder / "code/models/transformer_h/59_transformer.h5"

    batch_size = 128
    epochs = 5
    labels_path = root_folder /'data/label_59.csv'
    n_classes = 59
    data_path = root_folder /'data/vggish_final.npz'
 

    print('start preparing data ...')
    # load data and labels
    list_audio_ids, list_features = load_data(data_path)
    dic_label = load_label(labels_path)
    dic_label = dic_label.set_index('audio_Id')
    labels = [dic_label.loc[int(audio_id)].values for audio_id in list_audio_ids]

    samples = list(zip(list_features, labels))
    print(f"There are {len(samples)} samples in the dataset.")

    strat = [a[-1] for a in labels]
    cnt = Counter(strat)
    strat = [a if cnt[a] > 2 else "" for a in strat]

    train, val = train_test_split(
        samples, test_size=0.2, random_state=1337, stratify=strat
    )

    transformer_model = transformer_classifier(n_classes=n_classes)
    transformer_model.load_weights(transformer_h5)
   
    all_labels = []
    transformer_all_preds = []
    transformer_v2_all_preds = []

    rnn_all_preds = []

    for batch_samples in tqdm(
        chunker(val, size=batch_size), total=len(val) // batch_size
    ):
        X, labels = zip(*batch_samples)

        all_labels += labels

        repeats = 16

        transformer_Y = 0

        for _ in range(repeats):
            
            X = np.array(X)

            transformer_Y += transformer_model.predict(X) / repeats
          
        transformer_all_preds.extend(transformer_Y.tolist())


    T_Y = np.array(transformer_all_preds)
    Y = np.array(all_labels)

    trsf_ave_auc_pr = 0
    trsf_v2_ave_auc_pr = 0
    rnn_ave_auc_pr = 0

    total_sum = 0

    
    print(">>>>> calculating threshold: ")

    thresholds = {}
    for i in range(n_classes):
        precision, recall, threshold = precision_recall_curve(Y[:, i], T_Y[:, i])
        f_score = np.nan_to_num((2 * precision * recall) / (precision + recall))
        thresholds[i] = threshold[np.argmax(f_score)]  # removed float()
   
    print(list(thresholds.values()))
    
    print(">>>>> result without threshold: ")

    tsrf_f1 = f1_score(Y,  (T_Y > 0.5), labels=None, pos_label=1, average='macro', sample_weight=None, zero_division='warn')
    print("transformer f1 macro: ", tsrf_f1)
    
    tsrf_f1_micro = f1_score(Y,  (T_Y > 0.5), labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
    print("transformer f1 micro: ", tsrf_f1_micro)

    average_precision_score_micro = average_precision_score(Y,  (T_Y > 0.5), average = "micro")
    print("transformer average_precision_score_micro: ", average_precision_score_micro)

    roc_auc_score_micro = roc_auc_score(Y,  (T_Y > 0.5), average = "micro")
    print("transformer roc_auc_score_micro: ", roc_auc_score_micro)

    average_precision_score_macro = average_precision_score(Y,  (T_Y > 0.5), average = "macro")
    print("transformer average_precision_score_macro: ", average_precision_score_macro)

    roc_auc_score_macro = roc_auc_score(Y,  (T_Y > 0.5), average = "macro")
    print("transformer roc_auc_score_macro: ", roc_auc_score_macro)

    
    print(">>>>> result with threshold: ")
    y_pred = (T_Y  > np.array(list(thresholds.values()))) 
    y_test = (Y > 0.5) 
    
    tsrf_f1 = f1_score(y_test,  y_pred, labels=None, pos_label=1, average='macro', sample_weight=None, zero_division='warn')
    print("transformer f1 macro: ", tsrf_f1)
    
    tsrf_f1_micro = f1_score(y_test,  y_pred, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
    print("transformer f1 micro: ", tsrf_f1_micro)

    average_precision_score_micro = average_precision_score(y_test,  y_pred, average = "micro")
    print("transformer average_precision_score_micro: ", average_precision_score_micro)

    roc_auc_score_micro = roc_auc_score(y_test,  y_pred, average = "micro")
    print("transformer roc_auc_score_micro: ", roc_auc_score_micro)

    average_precision_score_macro = average_precision_score(y_test,  y_pred, average = "macro")
    print("transformer average_precision_score_macro: ", average_precision_score_macro)

    roc_auc_score_macro = roc_auc_score(y_test,  y_pred, average = "macro")
    print("transformer roc_auc_score_macro: ", roc_auc_score_macro)

   
    for i in range(n_classes):
        if np.sum(Y[:, i]) > 0:
            trsf_auc = average_precision_score(Y[:, i], T_Y[:, i])
     
            trsf_ave_auc_pr += np.sum(Y[:, i]) * trsf_auc
            total_sum += np.sum(Y[:, i])

    trsf_ave_auc_pr /= total_sum

    print("transformer micro-average     : ", trsf_ave_auc_pr)

   