import wget
from PIL import Image
import torch
import cv2
import os
import time
import numpy as np
import torchvision

import torchvision.transforms.functional as F
from math import ceil

from detect import Detector
from t_utils import xywh2xyxy, prepare4streamlit
from infer import video_input, image_input


cfg_small_model_path = 'models/yolov5s.pt'
cfg_med_model_path = 'models/yolov5s.pt'
cfg_big_model_path = 'models/yolov5s.pt'


model = None
confidence = .25
batch_size = 4


def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_


def main():
    # global variables
    global model, confidence, cfg_model_path

    model_src = ''
    # upload model
    # URL, upload file (max 200 mb)
    if model_src == "small":
        cfg_model_path = cfg_small_model_path
    elif model_src == "med":
        cfg_model_path = cfg_med_model_path
    else:
        cfg_model_path = cfg_big_model_path


    if not os.path.isfile(cfg_model_path):
        print("Модель не доступна!!!, пожалуйста обратитесь к авторам. Это треш, такого вообще не должно было быть.")
    else:
        # load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(cfg_model_path, device)

        # confidence slider

        # custom classes
        if st.sidebar.checkbox("Пользовательские классы"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Выберите классы", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())


        # input options
        input_option = 'video'

        # input src option
        # data_src = st.sidebar.radio("Выберите источник: ", ['Выбрать из примера', 'Загрузить данные'])

        if input_option == 'photo':
            image_input(model, confidence, batch_size)
        else:
            video_input(model, confidence, batch_size)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
