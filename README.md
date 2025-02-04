# fast_style_transfer_GAN
This is the study project within DLS courses dedicated to inference the GAN model of stylizing images (and video as bonus) via streamlit interface.

The main part of code was taken from [here](https://github.com/igreat/fast-style-transfer/tree/main) where author mentioned the original [paper](https://cs.stanford.edu/people/jcjohns/eccv16/)

The feature of this method is its fast time of inference.

In the folder `models` there are two code files. The `loss_models.py` involves the VGG16 model from torchvision.models and the content, style and TV (total variation) losses. The `transformation_models.py` file contains the main deeplearning model for applying style transfer.

In the folder `saved_models` there are the pretrained models (weights) and the mapping PRETRAINED_MODELS in the file `pretrained_models.py` corresponding for 4 styles.

In the folder `images` there are content images, style images (4 styles are pretrained) and generated images.

In the folder `videos` there are an example video `flag_kg_winter.mp4` with the stylized ones with names starting with "stylized_".

The file `app.py` is the main streamlit file.

The file `argument_parsers.py` contains the function training_parser() required in the running the `train_model.py` through the command line using argparser. In the original [work](https://github.com/igreat/fast-style-transfer/tree/main) the stylizations of image and video are to be done via command line. But in this work I deprecated these options because of the streamlit interface.

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






