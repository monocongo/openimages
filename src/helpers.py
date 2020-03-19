import os

import requests
from tqdm import tqdm


# Modified from https://stackoverflow.com/a/37573701.
def download_file(url: str, dest_path: str = None):
    """
    Downloads file at the given URL and stores it in dest_path if given, or
    returns the contents if dest_path is None (default).

    :param url URL to download.
    :param dest_path Location to store downloaded file at, or None to return
        contents instead.
    :return Downloaded file contents if dest_path is None, otherwise None.
    """
    response = requests.get(url, allow_redirects=True, stream=True)

    if response.status_code != 200:
        raise ValueError(
            f"Failed to download file from {url}. "
            f"-- Invalid response (status code: {response.status_code}).",
        )

    total_size = int(response.headers.get('content-length', 0))
    block_size = 100 * 1024  # 100 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True,
             desc=f"GET {os.path.basename(os.path.normpath(url))}")

    if dest_path is not None:
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
    else:
        file_contents_blocks = []
        for data in response.iter_content(block_size):
            t.update(len(data))
            file_contents_blocks.append(data)

    t.close()

    if total_size != 0 and t.n != total_size:
        raise ValueError(
            f"Download interrupted (received {t.n} of {total_size} bytes)")

    return bytes().join(file_contents_blocks) if dest_path is None else None
