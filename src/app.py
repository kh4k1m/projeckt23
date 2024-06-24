import streamlit as st
import wget
import torch
import os
from infer import video_input, image_input, folder_video_input, folder_image_input
from ultralytics import YOLO


st.set_page_config(layout="wide")

cfg_small_model_path = 'models/yolov8n.pt'
cfg_med_model_path = 'models/yolov9e.pt'
cfg_big_model_path = 'models/yolov5x6.pt'
root_dir = '/workspace/local_data'

model = None
confidence = .25
batch_size = 4




@st.cache_resource 
def load_model(path, device):
        # modell = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_small_model_path, force_reload=True, trust_repo=True)
        # modell.to(device).half().eval()
    names_list = ["Коптер", "Самолёт", "Вертолёт", "Птица", "БПЛА_С"]
    modell = YOLO(cfg_small_model_path)
    modell.to(device)
    # modelm = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_med_model_path, force_reload=True, trust_repo=True)
    # modelm.to(device).half().eval()
    modelm = YOLO(cfg_med_model_path)
    modelm.to(device)
    modelx = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_big_model_path, force_reload=True, trust_repo=True)
    modelx.to(device).half().eval()
    for i in range(5):
        modell.names[i] = names_list[i]
        modelm.names[i] = names_list[i]
        modelx.names[i] = names_list[i]
    # modell.names = names_list
    # modelm.names = names_list
    # model_.to(device)
    print("model to ", device)
    return modell, modelm, modelx



def main():
    # global variables
    global model, confidence, cfg_model_path

    st.title("Детекция летательных аппаратов")

    st.sidebar.title("Настройки")

    # upload model
    model_src = st.sidebar.radio("Выберите модель", ["Легкая", "Средняя", "Большая"])
    # URL, upload file (max 200 mb)
    st.sidebar.markdown("---")
    
    # device options
    if torch.cuda.is_available():
        device_option = st.sidebar.radio("Выберите устройство", ['ГПУ', 'ЦПУ'], disabled=False, index=0)
    else:
        device_option = st.sidebar.radio("Выберите устройство", ['ГПУ', 'ЦПУ'], disabled=True, index=0)

    # load model
    device = 'cuda' if device_option == 'ГПУ' else 'cpu'
    modell, modelm, modelx = load_model(cfg_big_model_path, device)
    if model_src == "Легкая":
        model = modell
        print('model', 'L'*40)
    elif model_src == "Средняя":
        model = modelm
        print('model', 'M'*40)
    else:
        model = modelx
        print('model', 'X'*40)

        # st.sidebar.text(cfg_model_path.split("/")[-1])
    

    # check if model file is available
    # if not os.path.isfile(cfg_model_path):
    #     st.warning("Модель не доступна!!!, пожалуйста обратитесь к авторам. Это треш, такого вообще не должно было быть.", icon="⚠️")
    # else:
        

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
        image_input(model, confidence, batch_size)
    elif input_option == 'видео':
        video_input(model, confidence, batch_size)
    elif input_option == 'папка с видео':
        folder_video_input(model, confidence, batch_size, root_dir)
    elif input_option == 'папка с фото':
        folder_image_input(model, confidence, batch_size, root_dir)



if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
