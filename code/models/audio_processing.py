
import soundfile as sf
import numpy as np
import os
from pydub import AudioSegment

import vggish.vggish_slim as vggish_slim
import vggish.vggish_params as vggish_params
import vggish.vggish_input as vggish_input
import vggish.vggish_postprocess as vggish_postprocess
import tensorflow.compat.v1 as tf

from pathlib import Path



def CreateVGGishNetwork(hop_size, sess, checkpoint_path):   # Hop size is in seconds.
  """Define VGGish model, load the checkpoint, and return a dictionary that points
  to the different tensors defined by the model.
  """
  vggish_slim.define_vggish_slim()
  vggish_params.EXAMPLE_HOP_SECONDS = hop_size
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)
  layers = {'conv1': 'vggish/conv1/Relu',
            'pool1': 'vggish/pool1/MaxPool',
            'conv2': 'vggish/conv2/Relu',
            'pool2': 'vggish/pool2/MaxPool',
            'conv3': 'vggish/conv3/conv3_2/Relu',
            'pool3': 'vggish/pool3/MaxPool',
            'conv4': 'vggish/conv4/conv4_2/Relu',
            'pool4': 'vggish/pool4/MaxPool',
            'fc1': 'vggish/fc1/fc1_2/Relu',
            #'fc2': 'vggish/fc2/Relu',
            'embedding': 'vggish/embedding',
            'features': 'vggish/input_features',
         }
  g = tf.get_default_graph()
  for k in layers:
    layers[k] = g.get_tensor_by_name( layers[k] + ':0')
  return {'features': features_tensor,
          'embedding': embedding_tensor,
          'layers': layers,
         }


def get_feature_vggish(wav_path):

    audio_id = wav_path.split('/')[-1].split('.')[0]

    # load vggish network
    root_folder = Path(__file__).parents[2]

    checkpoint_path = root_folder /'code/models/vggish/vggish_model.ckpt'
    pca_params_path = root_folder /'code/models/vggish/vggish_pca_params.npz'
    tf.compat.v1.disable_eager_execution()
    tf.reset_default_graph()
    sess = tf.Session()
    vgg = CreateVGGishNetwork(hop_size=0.02, sess=sess, checkpoint_path=checkpoint_path)
    
    # take centered 30 seconds of audios
    f = sf.SoundFile(wav_path)
    sr = f.samplerate  # Sample rate of the audio file
    data = f.read(dtype='int16')  # Read the entire file
    num_samples = data.shape[0]  # Total number of samples
    samples_30s = int(sr * 29.1)  # Number of samples for 30 seconds
    start_sample = int((num_samples - samples_30s) / 2)  # Start sample for the centered 30 seconds
    wav_data = data[start_sample:start_sample + samples_30s, :]
    
    # vggish features
    samples = wav_data / 32768.0 # Convert to [-1.0, +1.0]
    input_batch = vggish_input.waveform_to_examples(samples, sr)
    [embedding_batch] = sess.run([vgg['embedding']], feed_dict={vgg['features']: input_batch})
    pproc = vggish_postprocess.Postprocessor(pca_params_path)
    postprocessed_batch = pproc.postprocess(embedding_batch)

    return audio_id, postprocessed_batch 


def generate_wav(file_path_mp3):
    name, ext = os.path.splitext(file_path_mp3)

    if ext == ".mp3":
        mp3_sound = AudioSegment.from_mp3(file_path_mp3)
        mp3_sound.export("{0}.wav".format(name), format="wav")
    
    return "{0}.wav".format(name)

    


