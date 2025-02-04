import torch
from models.transformation_models import TransformationModel


def test_transformation_model():
    """Проверяем, работает ли TransformationModel."""
    model = TransformationModel()
    input_tensor = torch.rand(1, 3, 256, 256)  # Батч из 1 изображения
    output_tensor = model(input_tensor)

    assert_msg = "Выход TransformationModel имеет неправильную форму!"
    assert output_tensor.shape == input_tensor.shape, assert_msg
