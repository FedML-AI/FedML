import wget


def download_and_extract_archive(
    url: str,
    download_root: str,
) -> None:
    filename = wget.download(url, out=download_root)
    print(filename)
