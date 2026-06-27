"""Basic smoke tests. Real tests come in next commits."""

from tehran_house_price import __version__
from tehran_house_price.settings import get_config
from tehran_house_price.utils.paths import project_root


def test_version():
    assert __version__ == "0.1.0"


def test_config_loads():
    cfg = get_config()
    assert cfg["project"]["name"] == "tehran_house_price"


def test_project_root_exists():
    assert project_root().exists()
