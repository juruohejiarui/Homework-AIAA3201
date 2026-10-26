"""Parse the official TUM page and classify archives without duplicating ROS bags."""
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

PAGE = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download"
HOSTS = {"cvg.cit.tum.de", "vision.in.tum.de"}


def category(name):
    if "_validation" in name:
        return "validation"
    if "calibration" in name or "checkerboard" in name:
        return "calibration"
    return "ordinary"


class ArchiveLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = {}

    def handle_starttag(self, tag, attributes):
        if tag != "a":
            return
        href = dict(attributes).get("href", "")
        url = urljoin(PAGE, href)
        parsed = urlparse(url)
        name = PurePosixPath(parsed.path).name
        if not name.startswith("rgbd_dataset_freiburg") or not name.endswith(".tgz"):
            return
        if parsed.hostname not in HOSTS:
            raise ValueError(f"Unexpected archive host on TUM page: {url}")
        url = parsed._replace(scheme="https").geturl()
        entry = {"name": name, "url": url, "category": category(name)}
        if name in self.links and self.links[name]["url"] != url:
            raise ValueError(f"Conflicting archive URLs: {name}")
        self.links[name] = entry


def crawl():
    with urlopen(PAGE, timeout=45) as response:
        html = response.read().decode("utf-8")
    parser = ArchiveLinks()
    parser.feed(html)
    if not parser.links:
        raise ValueError("No TUM archives found; official page layout may have changed")
    return sorted(parser.links.values(), key=lambda entry: entry["name"])


def get_size(entry):
    for attempt in range(3):
        try:
            with urlopen(Request(entry["url"], method="HEAD"), timeout=45) as response:
                size = int(response.headers["Content-Length"])
                if response.status != 200 or size <= 0:
                    raise ValueError(f"Invalid archive response: {entry['url']}")
                return {**entry, "bytes": size}
        except (OSError, ValueError, TypeError):
            if attempt == 2:
                raise
