
import numpy as np

from models import transformer_classifier
from audio_processing import generate_wav, get_feature_vggish

from itertools import compress

from pathlib import Path



def predict_mood(n_label, audio_file_path):

    # generate wav from mp3
    wav_file_path = generate_wav(audio_file_path)  
     
    # extract features using vggish
    audio_id, features = get_feature_vggish(wav_path=wav_file_path)

    # load transformer
    transformer_h = get_transformer_h(n_label=n_label)
    transformer_model = transformer_classifier(n_classes=n_label)
    transformer_model.load_weights(transformer_h)

    repeats = 8
    transformer_Y = 0
    for _ in range(repeats):
        X = np.array([features])
        transformer_Y += transformer_model.predict(X) / repeats

    # load threshold
    threshold = get_class_threshold(n_label=n_label)
    
    y_pred = (transformer_Y  > np.array(threshold))
    class_mapping = get_class_mapping(n_label=n_label)

    return list(compress(class_mapping, y_pred[0]))



def get_transformer_h(n_label):

    root_folder = Path(__file__).parents[2]
    if n_label == 59:
        return root_folder / 'code/models/transformer_h/59_transformer.h5'
   
    elif n_label == 10:
        return root_folder / 'code/models/transformer_h/10_transformer.h5'
 
    elif n_label == 8:
        return root_folder / 'code/models/transformer_h/8_transformer.h5'
 
    elif n_label == 5:
        return root_folder / 'code/models/transformer_h/5_transformer.h5'
 

def get_class_mapping(n_label):
    if n_label == 59:
        return ['action', 'adventure', 'advertising', 'ambiental',
       'background', 'ballad', 'calm', 'children', 'christmas', 'commercial',
       'cool', 'corporate', 'dark', 'deep', 'documentary', 'drama', 'dramatic',
       'dream', 'emotional', 'energetic', 'epic', 'fast', 'film', 'fun',
       'funny', 'game', 'groovy', 'happy', 'heavy', 'holiday', 'hopeful',
       'horror', 'inspiring', 'love', 'meditative', 'melancholic', 'mellow',
       'melodic', 'motivational', 'movie', 'nature', 'party', 'positive',
       'powerful', 'relaxing', 'retro', 'romantic', 'sad', 'sexy', 'slow',
       'soft', 'soundscape', 'space', 'sport', 'summer', 'trailer', 'travel',
       'upbeat', 'uplifting']
    elif n_label == 10:
        return ['action', 'adventure', 'advertising', 'calm', 'children',
       'dark', 'documentary', 'holiday', 'party', 'trailer']
    
    elif n_label == 8:
        return ['action', 'adventure', 'advertising', 'ambiental', 'ballad',
       'calm', 'dark', 'documentary']
    
    elif n_label == 5:
        return ['adventure', 'advertising', 'calm', 'dark', 'documentary'] 


def get_class_threshold(n_label):   
    if n_label == 59:
        return [0.6012993454933167, 0.3382226824760437, 0.31850722432136536, 0.17198117077350616, 0.2098543792963028, 
        0.23739218711853027, 0.23753446340560913, 0.5619033575057983, 0.3189554810523987, 0.519826352596283, 
        0.0584104023873806, 0.3821624517440796, 0.5602417588233948, 0.6844568848609924, 0.21136997640132904, 
        0.27292805910110474, 0.3718380331993103, 0.29905015230178833, 0.4771442711353302, 0.52419513463974, 
        0.7821753621101379, 0.19788958132266998, 0.5585399270057678, 0.2699269950389862, 0.3003271520137787, 
        0.6350449323654175, 0.1581130176782608, 0.6577697992324829, 0.36202090978622437, 0.07300777733325958, 
        0.09684327244758606, 0.19699722528457642, 0.3914148211479187, 0.22362738847732544, 0.38036009669303894, 
        0.19834712147712708, 0.044998519122600555, 0.3200647234916687, 0.28658244013786316, 0.2686421573162079, 
        0.1646978110074997, 0.23529843986034393, 0.40697768330574036, 0.18040449917316437, 0.3629751205444336, 
        0.2326177954673767, 0.25954103469848633, 0.3039984405040741, 0.0865105390548706, 0.2566741108894348, 
        0.3077268600463867, 0.4728788137435913, 0.219923198223114, 0.43041327595710754, 0.5963019728660583, 
        0.5283707976341248, 0.11069964617490768, 0.23919276893138885, 0.40541982650756836]

    elif n_label == 10:
         return [0.5487613081932068, 0.6980000138282776, 0.6750208735466003, 0.7213850021362305, 0.33256226778030396, 
         0.5687593221664429, 0.46707355976104736, 0.3864097595214844, 0.7889232039451599, 0.4387631118297577]

    
    elif n_label == 8:
        return [0.6718928217887878, 0.6334792375564575, 0.5778589844703674, 0.21438048779964447, 0.20546191930770874, 
        0.5962992906570435, 0.6785698533058167, 0.6272515654563904]

    
    elif n_label == 5:
        return [0.5840750336647034, 0.727165162563324, 0.8148972988128662, 0.49536406993865967, 0.5207918882369995]

