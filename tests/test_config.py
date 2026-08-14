from src import config


def test_paths_are_under_project_root():
    assert config.DATA_DIR.is_relative_to(config.ROOT) or str(config.DATA_DIR).startswith(str(config.ROOT))
    assert config.KAGGLE_DIR == config.DATA_DIR / "chest_xray"


def test_image_size_is_positive():
    assert config.IMAGE_SIZE > 0
