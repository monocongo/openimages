import logging
import os

import pytest

from openimages import download

# ------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_dir",
)
def test_download_dataset(
        data_dir,
):
    """
    Test for the openimages.download.download_dataset() function

    :param data_dir: temporary directory into which test files will be loaded
    """

    pass
