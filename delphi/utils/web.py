""" Helper functions for web activity. """

from tqdm import tqdm
import urllib.request as request

def tqdm_reporthook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def get_data_from_url(url: str):
    return request.urlopen(url)


def download_file(url: str, filename: str):
    print(f"Downloading {url} to {filename}")
    with tqdm(ncols=80, unit="bytes", unit_scale=True, unit_divisor=1024) as t:
        reporthook = tqdm_reporthook(t)
        request.urlretrieve(url, filename, reporthook)
