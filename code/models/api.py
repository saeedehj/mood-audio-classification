import streamlit as st
import os.path
import numpy as np
import pandas as pd
from PIL import Image

from predict import predict_mood

from pathlib import Path




def save_file(directory, sound_file):
    print(os.path.join(directory, sound_file.name))
    # save your sound file in the right folder by following the path
    with open(os.path.join(directory, sound_file.name), 'wb') as f:
         f.write(sound_file.getbuffer())
         f.close()
    return sound_file.name


def choice_prediction():
    st.write('# Prediction')
    st.write('### Choose a music file in .mp3 format')
   
    # upload sound
    uploaded_file = st.file_uploader(' ', type='mp3')

    
   
    if uploaded_file is not None:  
        # view details
        file_details = {'filename':uploaded_file.name, 'filetype':uploaded_file.type, 'filesize':uploaded_file.size}
        st.write(file_details)

        # read and play the audio file
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/mpeg')
        
        # save_file function
        root_folder = Path(__file__).parents[2]

        audio_file_name = save_file(directory=root_folder /'data/mp3/', sound_file=uploaded_file)
        
        # if you select the predict button
        st.write('### ')
        st.markdown(
                    """<style>
                        div.stButton > button:first-child { 
                            background-color: #4CAF50;
                            height: 4em;
                            width: 13em; 
                            text-align: center;
                            position: center;
                            border: 3px solid green; 
                            font-weight: bold;
                            margin: 0;
                            position: absolute;
                            top: 50%;
                            left: 50%;
                            -ms-transform: translate(-50%, -50%);
                            transform: translate(-50%, -50%);
                            }
                        </style>""",
                    unsafe_allow_html=True)
        if st.button('Predict'):
            result = str(predict_mood(8, root_folder /'data/mp3/'+audio_file_name)).replace('[', ' ').replace(']', ' ').replace("'", ' ').replace('"', ' ')
            
            st.write('### ')
            st.write('### ')

            # write the prediction: the prediction of the last sound sent corresponds to the first column
            st.markdown("<h2 style='color: black;'> The mood of this music is : " + result + "</h2>", unsafe_allow_html=True)

    else:
        st.write('The file has not been uploaded yet')
    return



if __name__ == '__main__':
    root_folder = Path(__file__).parents[2]

    st.image(Image.open(root_folder /'data/music-mood.jpg'), width=400, use_column_width=True)
    st.write('___')
    # create a sidebar
    st.sidebar.title('Music Emotion Recognition')
    select = st.sidebar.selectbox('', ['Home', 'Prediction'], key='1')
    st.sidebar.write(select)
    # if sidebar selection is "Prediction"
    if select=='Prediction':
        # choice_prediction function
        choice_prediction()
    # else: stay on the home page
    else:
        st.markdown("<h1 style='text-align: center; color: black;'>Welcome to the Music Emotion Recognition System</h1>", unsafe_allow_html=True)
