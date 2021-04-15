import os
import zipfile
from typing import Dict

from helpers import download_file


def download_segmentation_zipfiles(base_url: str, section: str, meta_dir: str):
    """
    Downloads segmentation mask archive files from OpenImages dataset.

    :param base_url: OpenImages URL location
    :param section: split section (train, validation, or test) for which to
        download the archives
    :param meta_dir: directory which we should download the archive files into
    """
    # make the metadata directory if it doesn't exist
    if meta_dir is not None:
        os.makedirs(meta_dir, exist_ok=True)

    for i in range(16):
        bin = format(i, "x")
        mask_filename = _construct_archive_filename(section, bin)
        url = f"{base_url}{section}-masks/{mask_filename}"
        dest_path = f"{meta_dir}/{mask_filename}"

        if not os.path.exists(dest_path):
            try:
                download_file(url, dest_path)
            except ValueError as e:
                raise ValueError(
                    f"Failed to get segmentation mask archive (bin {bin}) for split section {section}.",
                    e,
                )


def extract_segmentation_mask(arguments: Dict):
    """
    Extracts segmentation mask images from previously downloaded archive files.

    :param arguments: dictionary containing the following arguments:
        handle_map: dictionary with mask archive filenames as keys and
            associated zipfile objects as values
        section: split section (train, validation, or test) to which the
            mask belongs
        mask_filename: name of the mask image file
        dest_file_path: path to where the mask image will be extracted
    """
    archive_filename = _construct_archive_filename(arguments["section"], arguments["mask_filename"][0])
    zf_handle = arguments["handle_map"][archive_filename]

    zf_handle.extract(arguments["mask_filename"], arguments["dest_file_path"])


def open_segmentation_zipfiles(section: str, meta_dir: str) -> Dict:
    """
    Opens the segmentation mask archive files (as downloaded by
    `download_segmentation_zipfiles`) and returns a dictionary mapping the
    archive filenames to the associated zipfile objects.

    :param section: split section (train, validation, or test) for which to
        download the archives
    :param meta_dir: directory which we should download the archive files into
    :return A dictionary mapping segmentation archive filenames to their
        associated zipfile objects.
    """
    handle_map = {}
    for i in range(16):
        bin = format(i, "x")
        filename = _construct_archive_filename(section, bin)
        handle_map[filename] = zipfile.ZipFile(
            os.path.join(meta_dir, filename), mode="r")

    return handle_map


def close_segmentation_zipfiles(handle_map: Dict):
    """
    Closes the zipfile objects in the given handleMap, as opened by `open_segmentation_zipfiles`.

    :param handle_map: the dictionary with zipfile handles to close.
    """
    for hnd in handle_map.values():
        hnd.close()


def _construct_archive_filename(section: str, bin: str):
    return f"{section}-masks-{bin}.zip"
