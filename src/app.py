import streamlit as st
import wget
import torch
import os
from infer import video_input, image_input, folder_video_input, folder_image_input
from ultralytics import YOLO


st.set_page_config(layout="wide")

cfg_small_model_path = 'models/fast.pt'
cfg_med_model_path = 'models/yolov9m.pt'
cfg_big_model_path = 'models/yolov5x6.pt'
root_dir = '/workspace/local_data'

model = None
confidence = .25
batch_size = 4





def load_model(path, device):
    if "yolov5x6" in path:
        model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True, trust_repo=True)
        model_.to(device).half().eval()
    else:
        model_ = YOLO(path)
    model_.to(device)
    print("model to ", device)
    return model_



def main():
    # global variables
    global model, confidence, cfg_model_path

    st.title("Детекция летательных аппаратов")

    st.sidebar.title("Настройки")

    # upload model
    model_src = st.sidebar.radio("Выберите модель", ["Легкая", "Средняя", "Большая"])
    # URL, upload file (max 200 mb)
    if model_src == "Легкая":
        cfg_model_path = cfg_small_model_path
    elif model_src == "Средняя":
        cfg_model_path = cfg_med_model_path
    else:
        cfg_model_path = cfg_big_model_path

        # st.sidebar.text(cfg_model_path.split("/")[-1])
    st.sidebar.markdown("---")

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Модель не доступна!!!, пожалуйста обратитесь к авторам. Это треш, такого вообще не должно было быть.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Выберите устройство", ['ГПУ', 'ЦПУ'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Выберите устройство", ['ГПУ', 'ЦПУ'], disabled=True, index=0)

        # load model
        device = 'cuda' if device_option == 'ГПУ' else 'cpu'
        model = load_model(cfg_model_path, device)

        # confidence slider
        confidence = st.sidebar.slider('Уверенность', min_value=0.1, max_value=1.0, value=confidence)
        # custom classes
        if st.sidebar.checkbox("Пользовательские классы"):
            model_names = list(model.names.values())
            assigned_class = st.sidebar.multiselect("Выберите классы", model_names, default=[model_names[0]])
            classes = [model_names.index(name) for name in assigned_class]
            model.classes = classes
        else:
            model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        # input options
        input_option = st.sidebar.radio("Тип файла: ", ['фото', 'видео', 'папка с фото', 'папка с видео'])

        # input src option
        # data_src = st.sidebar.radio("Выберите источник: ", ['Выбрать из примера', 'Загрузить данные'])

        if input_option == 'фото':
            image_input(model, confidence, batch_size, cfg_model_path)
        elif input_option == 'видео':
            video_input(model, confidence, batch_size, cfg_model_path)
        elif input_option == 'папка с видео':
            folder_video_input(model, confidence, batch_size, root_dir, cfg_model_path)
        elif input_option == 'папка с фото':
            folder_image_input(model, confidence, batch_size, root_dir, cfg_model_path)



if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
