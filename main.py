import numpy as np

from keras import models

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer

commands = ['right' ,'yes' ,'left' ,'stop' ,'up' ,'go' ,'no' ,'down']

loaded_model = models.load_model("saved_model")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("Predicted label:", command)
    return command

    
if __name__ == "__main__":
    from turtle_helper import move_turtle
    while True:
        command = predict_mic()
        move_turtle(command)
        if command == "stop":
            terminate()
            break