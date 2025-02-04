import torch
from utils import preprocess_image, deprocess_image, get_gram_matrix


# Фиксированные параметры нормализации
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def test_preprocess_deprocess():
    """Тестируем преобразование изображения туда и обратно."""
    tensor = torch.rand(3, 256, 256)  # Фейковое изображение
    processed = preprocess_image(tensor, mean, std)
    restored = deprocess_image(processed, mean, std)

    assert_msg = "Ошибка в преобразовании!"
    assert torch.allclose(tensor, restored, atol=1e-2), assert_msg


def test_gram_matrix():
    """Проверяем, что Gram Matrix имеет правильные размеры."""
    tensor = torch.rand(1, 3, 64, 64)  # Батч из 1 изображения 3x64x64
    gram = get_gram_matrix(tensor)

    assert gram.shape == (1, 3, 3), "Размер Gram Matrix неверный!"
