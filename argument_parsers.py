# this is where all argument parsing should be done
import argparse
import torch


def training_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a model")
    parser.add_argument(
        "--style_image_path",
        type=str,
        required=True,
        help="path to the style image",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="data/mscoco-new",
        help="path to the training dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="saved_models/trained_model.pth",
        help="path to save the trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of epochs to train the model for",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size to train the model with",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="image size to train the model with",
    )
    parser.add_argument(
        "--style_size",
        type=int,
        default=None,
        help=(
            "style size to train the model with. "
            "if not specified, the orignal size will be used"
        ),
    )
    parser.add_argument(
        "--style_weight",
        type=float,
        default=5e7,
        help="weight of the style loss",
    )
    parser.add_argument(
        "--content_weight",
        type=float,
        default=1e2,
        help="weight of the content loss",
    )
    parser.add_argument(
        "--tv_weight",
        type=float,
        default=0,
        help="weight of the total variation loss",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="learning rate to train the model with",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help=(
            "path to the checkpoint to resume training from."
            "If not specified, training will start from scratch"
        ),
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=2000,
        help=(
            "number of images to train on before saving a checkpoint."
            "keep it a multiple of the batch size"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="device to train the model on",
    )
    args = parser.parse_args()

    if not args.device:
        args.device = {torch.has_cuda: "cuda",
                       torch.has_mps: "mps"}.get(True, "cpu")

    return args
