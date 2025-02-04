import streamlit as st
import torch
import os
import time
from PIL import Image
from torchvision.utils import save_image
from saved_models.pretrained_models import PRETRAINED_MODELS
from stylize_image import stylize_image
from stylize_video import stylize_video


device = "cuda" if torch.cuda.is_available() else "cpu"

st.title("🎨 Fast Style Transfer")
st.write("Выберите тип контента для стилизации.")

content_type = st.radio("Выберите тип:", ["Изображение", "Видео"])

if "selected_style" not in st.session_state:
    st.session_state.selected_style = None

# --- 🖼 **Стилизация изображения** ---
if content_type == "Изображение":
    uploaded_file = st.file_uploader("Загрузите изображение...",
                                     type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Исходное изображение",
                 use_container_width=True)

        st.write("Выберите стиль:")
        col1, col2, col3, col4 = st.columns(4)

        style_images = {
            "starry_night": "images/style_images/starry-night.jpg",
            "rain_princess": "images/style_images/rain-princess.jpg",
            "abstract": "images/style_images/abstract-dalle.png",
            "mosaic": "images/style_images/mosaic.jpg"
        }

        def select_style(style_name):
            st.session_state.selected_style = style_name  # Сохраняем стиль

        with col1:
            if st.button("Starry Night"):
                select_style("starry_night")
        with col2:
            if st.button("Rain Princess"):
                select_style("rain_princess")
        with col3:
            if st.button("Abstract"):
                select_style("abstract")
        with col4:
            if st.button("Mosaic"):
                select_style("mosaic")

        # Показываем текущий выбор
        if st.session_state.selected_style:
            st.write(f"Выбран стиль: {st.session_state.selected_style}")
            st.image(style_images[st.session_state.selected_style],
                     use_container_width=True)

            model_path = PRETRAINED_MODELS[st.session_state.selected_style]

            if st.button("Применить стиль"):
                stylized_image = stylize_image(image, image_size=512,
                                               model_path=model_path)

                # Сохраняем изображение в session_state
                st.session_state["stylized_image"] = stylized_image

                # Сохранение результата
                output_path = f"images/generated_images/\
                    stylized_{uploaded_file.name}"
                save_image(stylized_image, output_path)
                # Сохраняем путь к файлу
                st.session_state["output_path"] = output_path

        # ✅ Показываем изображение только из session_state
        if "stylized_image" in st.session_state:
            st.image(st.session_state["output_path"],
                     caption="Стилизованное изображение",
                     use_container_width=True)

            # Кнопка для скачивания результата
            with open(st.session_state["output_path"], "rb") as file:
                st.download_button(label="Скачать изображение",
                                   data=file,
                                   file_name=st.session_state["output_path"],
                                   mime="image/png")

# --- 🎥 **Стилизация видео** ---
elif content_type == "Видео":
    uploaded_video = st.file_uploader("Загрузите видео (MP4)...", type=["mp4"])

    if uploaded_video is not None:
        # Сохраняем оригинальное видео во временную папку
        input_video_path = f"videos/{uploaded_video.name}"
        with open(input_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(input_video_path)
        st.write(f"Видео загружено: {uploaded_video.name}")

        # Выбор стиля
        selected_style = st.selectbox("Выберите стиль:",
                                      list(PRETRAINED_MODELS.keys()))
        model_path = PRETRAINED_MODELS[selected_style]

        if st.button("Применить стиль"):
            output_video_path = f"videos/stylized_{uploaded_video.name}"

            # Создаем прогресс-бар
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Запуск стилизации
            msg = "Стилизация видео... Это может занять несколько минут ⏳"
            with st.spinner(msg):
                output_video_path = stylize_video(
                    video_path=input_video_path,
                    model_path=model_path,
                    save_path=output_video_path,
                    frames_per_step=8,  # Количество кадров за один шаг
                    image_size=128,
                    progress_bar=progress_bar,
                    status_text=status_text
                )

            # ✅ Ждём, пока файл действительно появится
            while (not os.path.exists(output_video_path) or
                   os.path.getsize(output_video_path) == 0):
                time.sleep(1)

            st.success("✅ Стилизация завершена!")

            # ✅ **Сохраняем результат в session_state**
            st.session_state["stylized_video"] = output_video_path

        if "stylized_video" in st.session_state:
            video_path = st.session_state["stylized_video"]

            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                st.video(os.path.abspath(video_path))
            else:
                st.warning("⚠️ Видео не загружено или повреждено.")

            # ✅ Кнопка для скачивания файла
            with open(video_path, "rb") as f:
                st.download_button(
                    label="📥 Скачать стилизованное видео",
                    data=f,
                    file_name=os.path.basename(video_path),
                    mime="video/mp4",
                )
