
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from models import transformer_classifier
from prepare_data import DataGenerator, load_data, load_label
from collections import Counter

from pathlib import Path


if __name__ == "__main__":

    root_folder = Path(__file__).parents[2]


    h5_name = "59_transformer.h5"
    labels_path = root_folder /'data/label_59.csv'
    n_classes = 59
    data_path = root_folder /'data/vggish_final.npz'
    batch_size = 32
    epochs = 50
    
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

    print('start training ...')
    model = transformer_classifier(n_classes=n_classes)

    checkpoint = ModelCheckpoint(
        h5_name,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=True,
    )
    reduce_o_p = ReduceLROnPlateau(
        monitor="val_loss", patience=20, min_lr=1e-7, mode="min"
    )

    model.fit_generator(
        DataGenerator(train, batch_size=batch_size),
        validation_data=DataGenerator(val, batch_size=batch_size),
        epochs=epochs,
        callbacks=[checkpoint, reduce_o_p],
        use_multiprocessing=True,
        workers=12,
        verbose=2,
        max_queue_size=64,
    )
