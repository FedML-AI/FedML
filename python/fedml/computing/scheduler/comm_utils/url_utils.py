from urllib.parse import urlparse, urlunparse


def replace_inference_port(inference_url, worker_proxy_port):
    # Parse the URL
    parsed_url = urlparse(inference_url)

    # Replace the port in the netloc
    new_netloc = f"{parsed_url.hostname}:{worker_proxy_port}"

    # Reconstruct the URL with the new port
    new_url = urlunparse((
        parsed_url.scheme,
        new_netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment
    ))
    return new_url


def remove_url_path(inference_url):
    parsed_url = urlparse(inference_url)
    new_url = urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        "",
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment
    ))
    return new_url
