import os
import numpy as np
import time
import subprocess
import cv2
import torch
from models import loss_models, transformation_models
from utils import preprocess_batch, deprocess_batch
from torchvision.transforms.functional import resize


device = "cuda" if torch.cuda.is_available() else "cpu"


def stylize_video(video_path, model_path, save_path, frames_per_step, image_size, progress_bar=None, status_text=None):
    # load the video
    video = cv2.VideoCapture(video_path)

    # get the video's dimensions and frame count
    width_original = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frames_to_capture = frame_count

    print(f"source video dimensions: {width_original}x{height_original}")
    print(f"source video frame count: {frame_count}")
    print(f"source video fps: {fps}\n")

    # get the video output dimensions
    width = width_original
    height = height_original
    if image_size:
        min_dim = min(width_original, height_original)
        width = int(width_original / min_dim * image_size)
        height = int(height_original / min_dim * image_size)

    print(f"output video dimensions: {width}x{height}")
    print(f"output video frame count: {frames_to_capture}")
    print(f"output video fps: {fps}\n")

    # setting up the model
    transformation_model = transformation_models.TransformationModel().to(device).eval()
    # loading weights of pretrained model
    checkpoint = torch.load(model_path, map_location=device)
    transformation_model.load_state_dict(checkpoint["model_state_dict"])
    transformation_model.requires_grad_(False)

    # partition the frames into batches of size 64
    frames_batch_size = 16

    # save the frames as a video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        save_path,
        fourcc,
        float(fps),
        (width, height),
    )

    # total_batches = frame_count // frames_batch_size + (1 if frame_count % frames_batch_size != 0 else 0)

    start_time = time.time()  # Засекаем время

    # use the first iteration to get the frame sizes
    for i in range(0, frames_to_capture, frames_batch_size):
        # make sure the last batch has the correct size
        batch_size = frames_batch_size
        if i + frames_batch_size > frames_to_capture:
            batch_size = frames_to_capture - i
        # create a batch of empty frames
        frames_batch = np.empty(
            (batch_size, height_original, width_original, 3), dtype=np.uint8
        )
        print(f"stylizing frames <{i}-{i + batch_size}/{frames_to_capture}>")

        # read the frames
        frame_index = 0
        ret = True
        while video.isOpened() and frame_index < frames_batch_size:
            ret, frame = video.read()
            if ret:
                frames_batch[frame_index] = frame
                frame_index += 1
            else:
                # end of frames batch
                break

        stylized_batch = stylize_frames_batch(
            frames_batch, transformation_model, frames_per_step, image_size
        )
        for styled_frame in stylized_batch:
            out.write(styled_frame)

        # **Обновляем прогресс-бар**
        if progress_bar and status_text:
            elapsed_time = time.time() - start_time
            progress = (i + batch_size) / frame_count
            progress_bar.progress(progress)

            estimated_time_left = (elapsed_time / progress) * (1 - progress) if progress > 0 else 0
            status_text.text(f"Прогресс: {int(progress * 100)}% ⏳ Осталось: {int(estimated_time_left)} сек.")

    print(f"✅ Стилизованное видео сохранено в {save_path}")
    out.release()
    time.sleep(2)  # ⏳ Даем время файлу завершить запись

    if status_text:
        status_text.text("✅ Стилизация завершена!")
        progress_bar.progress(1.0)

    # to add the audio back to the video, run this command in the terminal:
    # ffmpeg -i {save_path} -i {video_path} -c copy -map 0:v:0 -map 1:a:0 {save_with_audio_path}

    # # Конвертируем в H.264
    fixed_video_path = save_path.replace(".mp4", "_fixed.mp4")
    ffmpeg_cmd = [
        "ffmpeg", "-y",  # Перезаписывать файл без запроса
        "-i", save_path,  # Входной файл
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",  # Кодек видео
        "-c:a", "aac", "-b:a", "128k",  # Аудиокодек
        fixed_video_path  # Выходной файл
    ]

    print(f"🔄 Конвертация видео в H.264: {fixed_video_path}")
    subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Проверяем, создалось ли видео
    if os.path.exists(fixed_video_path) and os.path.getsize(fixed_video_path) > 0:
        print(f"✅ Видео успешно конвертировано: {fixed_video_path}")
        return fixed_video_path  # ✅ Возвращаем путь к файлу!
    else:
        print("❌ Ошибка: Видео не сконвертировано!")
        return save_path  # Если что-то пошло не так, используем оригинальный файл


def stylize_frames_batch(
    frames, transformation_model, frames_per_step, image_size=None
):
    """
    Stylize a batch of frames
    """
    # change the frames into torch tensors
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

    # preprocess the frames to what the model expects
    mean = loss_models.VGG16Loss.MEAN
    std = loss_models.VGG16Loss.STD
    frames = preprocess_batch(frames, mean, std)
    if image_size:
        frames = resize(frames, image_size)
    width, height = frames.shape[3], frames.shape[2]
    mean = mean.to(device)
    std = std.to(device)

    frames_to_capture = frames.shape[0]
    # stylize the frames in batches
    stylized_frames = torch.empty_like(frames)
    for i in range(0, frames_to_capture, frames_per_step):
        # get the batch
        section = frames[i : i + frames_per_step].to(device)
        # stylize the batch
        stylized_section = transformation_model(section)

        # depreprocess the batch
        stylized_section = deprocess_batch(stylized_section, mean, std)

        # for some reason the transformed image ends up having slightly different dimensions
        # so we resize it to the right dimensions
        stylized_section = resize(
            stylized_section, (section.shape[2], section.shape[3])
        )
        # save the batch
        stylized_frames[i : i + frames_per_step] = stylized_section

        # print progress every 24 frames
        if i % 24 == 0:
            print(f"from batch, stylized frame [{i}/{frames_to_capture}]")

    print("styled frames successfully\n")

    # convert the frames back to numpy arrays
    stylized_frames = (
        stylized_frames.detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .mul(255)
        .numpy()
        .astype("uint8")
    )
    # colors channel is in BGR, so we convert it to RGB
    stylized_frames = stylized_frames[:, :, :, ::-1]

    return stylized_frames
