# fast_style_transfer_GAN
This is the study project within DLS courses dedicated to inference the GAN model of stylizing images (and video as bonus) via streamlit interface.

The main part of code was taken from [here](https://github.com/igreat/fast-style-transfer/tree/main) where author mentioned the original [paper](https://cs.stanford.edu/people/jcjohns/eccv16/)

The feature of this method is its fast time of inference.

# Structure

In the folder `gifs` there are gif-images from original video and styilized generated videos. In order to view in github directly.

In the folder `images` there are content images, style images (4 styles are pretrained) and generated images.

In the folder `models` there are two code files. The `loss_models.py` involves the VGG16 model from torchvision.models and the content, style and TV (total variation) losses. The `transformation_models.py` file contains the main deeplearning model for applying style transfer.

In the folder `reports` there are the demo video of inference and some files for grading the project.

In the folder `saved_models` there are the pretrained models (weights) and the mapping PRETRAINED_MODELS in the file `pretrained_models.py` corresponding for 4 styles.

In the folder `tests` there are tests covering some functions from the code of the project.

In the folder `videos` there are an example video `flag_kg_winter.mp4` with the stylized ones with names starting with "stylized_".

The file `app.py` is the main streamlit file.

The file `argument_parsers.py` contains the function training_parser() required in the running the `train_model.py` through the command line using argparser. In the original [work](https://github.com/igreat/fast-style-transfer/tree/main) the stylizations of image and video are to be done via command line. But in this work I deprecated these options because of the streamlit interface.

The file `Dockerfile` contains docker instructions for building the docker image.

The file `requirements.txt` contains the packages required to the streamlit cloud app.

The file `stylize_image.py` contains the function for image stylization.

The file `stylize_video.py` contains the function for video stylization.

The file `train_model.py` is to be launched through command line with arguments. It loads the dataset and prepared the images, initializes the model and runs training cycle. Here is the example how to run:

`python train_model.py --style_image_path {PATH TO YOUR STYLE IMAGE} --train_dataset_path {PATH TO DATASET}`

The file `utils.py` contains functions and classes required for stylization and model training.

# Styles

<p align="center">
  <img src="images/style_images/starry-night.jpg" height="150">
  <img src="images/style_images/rain-princess.jpg" height="150">
  <img src="images/style_images/abstract-dalle.png" height="150">
  <img src="images/style_images/mosaic.jpg" height="150">
</p>



# Results

<p align="center">
  <img src="images/content_images/bahla-fort.jpg" height="150">
</p>
<p align="center">
  <img src="images/generated_images/bahla-fort-starry-night.png" height="150">
  <img src="images/generated_images/bahla-fort-rain-princess.png" height="150">
  <img src="images/generated_images/bahla-fort-abstract.png" height="150">
  <img src="images/generated_images/bahla-fort-mosaic.png" height="150">
</p>

<hr>

<p align="center">
  <img src="images/content_images/fruits.jpg" height="150">
</p>
<p align="center">
  <img src="images/generated_images/fruits-starry-night.png" height="150">
  <img src="images/generated_images/fruits-rain-princess.png" height="150">
  <img src="images/generated_images/fruits-abstract.png" height="150">
  <img src="images/generated_images/fruits-mosaic.png" height="150">
</p>

<hr>

<p align="center">
  <img src="images/content_images/three_girls.jpg" height="150">
</p>
<p align="center">
  <img src="images/generated_images/three_girls_starry_night.jpg" height="150">
  <img src="images/generated_images/three_girls_rain.jpg" height="150">
  <img src="images/generated_images/three_girls_abstract.jpg" height="150">
  <img src="images/generated_images/three_girls_mozaic.jpg" height="150">
</p>

<hr>

<p align="center">
  <img src="images/content_images/photo_1.jpg" height="150">
</p>
<p align="center">
  <img src="images/generated_images/stylized_photo_1_starry_night.jpg" height="150">
  <img src="images/generated_images/stylized_photo_1_rain.jpg" height="150">
  <img src="images/generated_images/stylized_photo_1_abstract.jpg" height="150">
  <img src="images/generated_images/stylized_photo_1_mosaic.jpg" height="150">
</p>

<hr>

<p align="center">
  <img src="images/content_images/photo_2.jpg" height="150">
</p>
<p align="center">
  <img src="images/generated_images/stylized_photo_2_starry_night.jpg" height="150">
  <img src="images/generated_images/stylized_photo_2_rain.jpg" height="150">
  <img src="images/generated_images/stylized_photo_2_abstract.jpg" height="150">
  <img src="images/generated_images/stylized_photo_2_mosaic.jpg" height="150">
</p>

<hr>

# Video

Videos are located in the folder `videos`. Here I show the gifs:

<p align="center">
  <img src="gifs/flag_kg_winter.gif" width="800">
</p>
<p align="center">
  <img src="gifs/flag_kg_winter_starry_night.gif" height="150">
  <img src="gifs/flag_kg_winter_rain.gif" height="150">
  <img src="gifs/flag_kg_winter_abstract.gif" height="150">
  <img src="gifs/flag_kg_winter_mosaic.gif" height="150">
</p>

<hr>

# Docker usage

There are 2 ways to use the docker of the project:

1) One can run the ready docker image from my dockerhub: run the command `docker run -p 8501:8501 omurkanov/fast-style-transfer` and then open in browswer by address `http://localhost:8501`

2) One can build the docker image locally. For this one should clone the repo:
```git clone https://github.com/turgunbek/fast_style_transfer_GAN.git
cd fast_style_transfer_GAN```
Then build the image: `docker build -t fast-style-transfer .`
Then run the container: `docker run -p 8501:8501 fast-style-transfer`
Then open it in the browser: `http://localhost:8501`


# Reports (Отчёт для разбалловки)

* Есть работающий инференс (можно увидеть в reports\demo.mp4 или на сервере https://faststyletransfergan.streamlit.app/ или запустить самому: склонировать репозиторий, далее в терминале набрать `streamlit run app.py`).
* Есть работающее приложение на сервере по адресу: https://faststyletransfergan.streamlit.app/.
* Запись демо - файл `demo.mp4` находится в папке `reports`.
* Адекватный результат на тестовых данных - взял первые 2 фотки из тестовых (images\content_images\photo_1.jpg и images\content_images\photo_2.jpg) и для них сделал стилизиацию со всеми 4мя стилями, результаты находятся в папке images\generated_images\: stylized_photo_1_starry_night.jpg, stylized_photo_2_starry_night.jpg, ..., stylized_photo_1_mosaic.jpg, stylized_photo_2_mosaic.jpg, а также для других картинок есть результаты стилизаций.
* Описание репозитория довольно понятное, функции с докстрингами, ассертами и т.д.
* Линтеры - прогнал через flake8 - всё ок (скриншот виден в reports\linters_and_tests.png).
* Тесты - составил в папке tests тесты на некоторые функции, прогнал тесты - всё ок (скриншот виден в reports\linters_and_tests.png).
* Дополнительная фича - возможность стилизовать видео. Сделал в интерфейсе streamlit возможность выбора видео. На демо в конце видно, что работает с видео. Результаты находятся в папке videos.
* Докер-контейнер - собрал докер образ и выложил на докерхаб (https://hub.docker.com/r/omurkanov/fast-style-transfer), можно подтянуть командой `docker pull omurkanov/fast-style-transfer`. Также есть докерфайл `Dockerfile`, с помощью которого можно собрать докер-образ локально (см. инструкцию выше).

