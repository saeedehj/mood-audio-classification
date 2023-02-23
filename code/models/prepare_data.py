import pandas as pd
import numpy as np
import glob

from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, feature_label_list, batch_size=32):
        self.feature_label_list = feature_label_list
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.feature_label_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.feature_label_list) / self.batch_size / 10))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        batch_samples = [self.feature_label_list[k] for k in indexes]
       
        x, y = self.__data_generation(batch_samples)
      
        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_samples):
        X, labels = zip(*batch_samples)
       
        X = np.array(X)
        Y = np.array(labels)

        return X, Y[..., np.newaxis]


def load_tags(file_path):
    input = open(file_path,"r")
    raw_text = input.read()
    lines_list = raw_text.split("\n")
    del lines_list[0], lines_list[-1]
    tags = {}
    for line in lines_list:
        list_mood= []
        parts= line.split("\t")
        id= parts[0].split('_')[-1].lstrip('0')
        for i in range(0, len(parts)):
            if parts[i][0:10] == 'mood/theme':
                mood= parts[i].split('---')[-1]
                list_mood.append(mood)
        tags[id] = list_mood

    return tags

def load_labels(file_path):
    input = open(file_path,"r")
    raw_text = input.read()
    tag = raw_text.split("\n")
    labels = []
    for line in tag:
        t = line.split('---')[-1]
        labels.append(t)

    return labels


def join_tags_labels(tags, labels):
    nn = np.zeros((len(tags), len(labels)))
    df = pd.DataFrame(nn, index=tags.keys() , columns=labels)

    for i in tags.keys():
        vals = tags[i]
        for j in vals:
            df.loc[i][j] = 1
    
    df = df.reset_index()
    df.rename(columns={'index':'audio_Id'}, inplace=True)
    
    return df


def join_npz_files(file_path, save_file_path):
    file_list = []
    for i in glob.glob(file_path):
       file_list.append(i)
    
    data_all = [np.load(fname, allow_pickle=True) for fname in file_list]
    audio_ids = []
    features = []
    for data in data_all:
        for id in data['audio_id']:
            audio_ids.append(id)
        for feature in data['features']:
            features.append(feature)
   
    np.savez(save_file_path, audio_id = audio_ids, features = features)


def load_label(file_path):   
    df = pd.read_csv(file_path)
    return df

def load_data(file_path):   
    with np.load(file_path, allow_pickle=True) as data:
        list_audio_ids = data['audio_id']
        list_features = data['features']

    return list_audio_ids, list_features


# if __name__ == "__main__":
#     print('start preprocessing...s')
#     from pathlib import Path

#     root_folder = Path(__file__).parents[2]

#     tag_path = root_folder / "data/tag.txt"
#     tags = load_tags(tag_path)

    # labels = load_labels(root_folder / "data/labels.txt")
    # join_npz_files(root_folder / "data/vggish/*.npz")

    
