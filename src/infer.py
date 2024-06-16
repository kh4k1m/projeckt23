import streamlit as st
from PIL import Image
from turbojpeg import TurboJPEG, TJCS_RGB
import torch
import os
import time
import torchvision
import pathlib

from math import ceil

from detect import Detector
from t_utils import prepare4streamlit, save_auto_annot


jpeg = TurboJPEG()


def image_input(model, confidence, batch_size, cfg_model_path):
    img_file = None
    infer_image = Detector(model, confidence=confidence, nms_thresh=0.35, num_classes=80, draw_bbox=True)
    img_bytes = st.sidebar.file_uploader("Загрузить изображение", type=['png', 'jpeg', 'jpg', 'JPG', 'JPEG', 'PNG'])
    if img_bytes:
        img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
        Image.open(img_bytes).save(img_file)

    if img_file:
        drawed_img, boxes, scores, labels = infer_image(img_file)
        st.image(drawed_img, caption="Предсказание")


def folder_image_input(model, confidence, batch_size, root_dir, cfg_model_path):
    infer_image = Detector(model, confidence=confidence, nms_thresh=0.35, num_classes=80, draw_bbox=False)
    st.sidebar.markdown("Началась оффлайн обработка фото с папки!")
    height = 640
    width = 640
    prev_time = 0
    curr_time = 0
    fps = 0
    imgs_list = []
    lbls_list = []
    for address, dirs, files in os.walk(root_dir):
        for name in files:
            endswith = pathlib.PurePosixPath(name).suffix
            if endswith.lower() in ['.png', '.jpeg', '.jpg']:
                img_file = os.path.join(address, name)
                lbl_file = os.path.join(address, name.replace(endswith, '.txt'))
                imgs_list += [img_file]
                lbls_list += [lbl_file]
    len_files = len(imgs_list)
    st1, st2, st3, st4 = st.columns(4)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{height}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{width}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown(f"{fps}")
    with st4:
        st.markdown("## Process")
        st4_text = st.markdown(f"{0}/{len_files} кадров")
    start_time = time.time()
    for i, (img_file, lbl_file) in enumerate(zip(imgs_list, lbls_list)):
        with open(str(img_file), 'rb') as fl:
            np_img = jpeg.decode(fl.read(), TJCS_RGB)
        batch = torch.from_numpy(np_img).permute(2, 0, 1)[None, ...]
        _, _, height, width = batch.shape
        height = ceil(height / 64) * 64
        width = ceil(width / 64) * 64
        batch = torch.nn.functional.interpolate(batch, size=(height, width))
        output_imgs, boxes, scores, labels = infer_image(batch)
        curr_time = time.time()
        save_auto_annot(lbl_file, boxes, labels)
        fps = (1 / (curr_time - prev_time))
        prev_time = curr_time
        st1_text.markdown(f"**{height}**")
        st2_text.markdown(f"**{width}**")
        st3_text.markdown(f"**{fps:.2f}**")
        st4_text.markdown(f"**{i}/{len_files} кадров**")
    full_time = round(time.time() - start_time, 4)
    st4_text.markdown(f"**{len_files}/{len_files} кадров**")
    print('Time:', full_time)
    st.markdown(f"### Обработка фото завершена. Ушло на обработку {full_time} секунд. В локальной папке с фотографиями появились файлы их разметки.")



def video_input(model, confidence, batch_size, cfg_model_path):
    infer_image = Detector(model, confidence=confidence, nms_thresh=0.35, num_classes=80, draw_bbox=True)
    vid_file = None
    vid_bytes = st.sidebar.file_uploader("Загрузить видео", type=['mp4', 'mpv', 'avi', 'mov'])
    if not vid_bytes:
        return
    if vid_bytes:
        vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
        with open(vid_file, 'wb') as out:
            out.write(vid_bytes.read())
        del vid_bytes
    if vid_file:
        # output = st.empty()
        vframes, aframes, vmetadata = torchvision.io.read_video(vid_file)
        # reader = torchvision.io.VideoReader(vid_file, "video")

        video_fps = vmetadata['video_fps']
        audio_fps = vmetadata['audio_fps']
        _, height, width, _ = vframes.shape
        height = ceil(height / 64) * 64
        width = ceil(width / 64) * 64
        len_f = len(vframes)
        prev_time = 0
        curr_time = 0
        fps = 0
        st1, st2, st3, st4 = st.columns(4)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")
        with st4:
            st.markdown("## Process")
            st4_text = st.markdown(f"{0}/{len_f} кадров")
        vframes = torch.nn.functional.interpolate(vframes.permute((0, 3, 1, 2)), size=(height, width))
        video_array = []
        start_time = time.time()
        for i in range(0, len_f, batch_size):
            batch = vframes[i:i+batch_size] 
            output_imgs, boxes, scores, labels = infer_image(batch)
            if height > width:
                output_imgs = prepare4streamlit(output_imgs, height, width)
            video_array += output_imgs
            curr_time = time.time()
            fps = (1 / (curr_time - prev_time)) * len(batch)
            prev_time = curr_time
            st3_text.markdown(f"**{fps:.2f}**")
            st4_text.markdown(f"**{i}/{len_f} кадров**")
        st4_text.markdown(f"**{len_f}/{len_f} кадров**")
        print('Time:', time.time() - start_time)
        del vframes
        
        video_array = torch.stack(video_array)  # .permute((0, 2, 3, 1))
        torchvision.io.write_video(
            filename=vid_file,
            video_array=video_array,
            fps=video_fps,
            audio_array=aframes,
            audio_fps=audio_fps,
            audio_codec='mp3'
            )

        st.markdown("---")

        video_file = open(vid_file, 'rb')
        video_bytes = video_file.read()
        st.sidebar.download_button('Сохранить видео', video_file, file_name=os.path.basename(vid_file))
        st.video(video_bytes)

def folder_video_input(model, confidence, batch_size, root_dir, cfg_model_path):
    infer_image = Detector(model, confidence=confidence, nms_thresh=0.35, num_classes=80, draw_bbox=True)
    st.sidebar.markdown("Началась оффлайн обработка видео с папки!")
    start_time = time.time()
    for address, dirs, files in os.walk(root_dir):
        for name in files:
            endswith = pathlib.PurePosixPath(name).suffix
            if endswith.lower() in ['.mp4', '.mpv', '.avi', '.mov']:
                vid_file = os.path.join(address, name)
                out_vif_file = os.path.join(address, 'predicted_' + name)
                vframes, aframes, vmetadata = torchvision.io.read_video(vid_file)
                # reader = torchvision.io.VideoReader(vid_file, "video")

                video_fps = vmetadata['video_fps']
                audio_fps = vmetadata['audio_fps']
                _, height, width, _ = vframes.shape
                height = ceil(height / 64) * 64
                width = ceil(width / 64) * 64
                len_f = len(vframes)
                prev_time = 0
                curr_time = 0
                fps = 0
                st1, st2, st3, st4 = st.columns(4)
                with st1:
                    st.markdown("## Height")
                    st1_text = st.markdown(f"{height}")
                with st2:
                    st.markdown("## Width")
                    st2_text = st.markdown(f"{width}")
                with st3:
                    st.markdown("## FPS")
                    st3_text = st.markdown(f"{fps}")
                with st4:
                    st.markdown("## Process")
                    st4_text = st.markdown(f"{0}/{len_f} кадров")
                vframes = torch.nn.functional.interpolate(vframes.permute((0, 3, 1, 2)), size=(height, width))
                video_array = []
                
                for i in range(0, len_f, batch_size):
                    batch = vframes[i:i+batch_size] 
                    output_imgs, boxes, scores, labels = infer_image(batch)
                    if height > width:
                        output_imgs = prepare4streamlit(output_imgs, height, width)
                    video_array += output_imgs
                    curr_time = time.time()
                    fps = (1 / (curr_time - prev_time)) * len(batch)
                    prev_time = curr_time
                    st3_text.markdown(f"**{fps:.2f}**")
                    st4_text.markdown(f"**{i}/{len_f} кадров**")
                st4_text.markdown(f"**{len_f}/{len_f} кадров**")
                
                del vframes
                
                video_array = torch.stack(video_array)  # .permute((0, 2, 3, 1))
                torchvision.io.write_video(
                    filename=out_vif_file,
                    video_array=video_array,
                    fps=video_fps,
                    audio_array=aframes,
                    audio_fps=audio_fps,
                    audio_codec='mp3'
                    )
    full_time = round(time.time() - start_time, 4)
    print('Time:', full_time)
    st.markdown(f"### Обработка видео завершена. Ушло на обработку {full_time} секунд. В локальной папке с видео появились видеофайлы с разметкой.")


