import torch
from models.loss_models import StyleLoss, ContentLoss, VGG16Loss


def test_style_loss():
    """Проверяем работу StyleLoss."""
    tensor = torch.rand(4, 3, 64, 64)  # Батч из 4 изображений
    loss_fn = StyleLoss(tensor, weight=1.0)
    output = loss_fn(tensor)

    assert output.shape == tensor.shape, "StyleLoss не сохраняет форму входа!"


def test_content_loss():
    """Проверяем работу ContentLoss."""
    tensor = torch.rand(4, 3, 64, 64)  # Батч из 4 изображений
    loss_fn = ContentLoss(weight=1.0)
    output = loss_fn(tensor)

    assert_msg = "ContentLoss не сохраняет форму входа!"
    assert output.shape == tensor.shape, assert_msg


def test_vgg16_loss():
    """Проверяем, создается ли модель VGG16Loss без ошибок."""
    tensor = torch.rand(1, 3, 256, 256)  # 1 изображение
    loss_fn = VGG16Loss(style_img=tensor, device="cpu")

    assert isinstance(loss_fn, VGG16Loss), "Ошибка при создании VGG16Loss!"
