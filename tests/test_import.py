from importlib.metadata import version

import tinynlp


def test_import_exposes_package_version() -> None:
    assert tinynlp.__version__ == version("tinynlp-opt")
    assert tinynlp.__all__ == ["__version__"]
