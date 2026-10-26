"""Discover every official NYUv2 Raw ZIP and its published MD5, never archive payloads."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import PurePosixPath
import re
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from depth_project.io import read_json
from depth_project.paths import PROJECT

PAGE = "https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html"
RAW_PREFIX = "/mit/silberman/nyu_depth_v2/"
KNOWN_ARCHIVES = PROJECT / "storage/nyuv2_full_download_manifest.json"


def get_text(url):
    with urlopen(url, timeout=45) as response:
        data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError(f"Unexpectedly large metadata response: {url}")
    return data.decode("utf-8-sig")


class RawLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.archives, self.checksums = {}, {}

    def handle_starttag(self, tag, attributes):
        if tag != "a":
            return
        parsed = urlparse(urljoin(PAGE, dict(attributes).get("href", "")))
        if parsed.hostname != "horatio.cs.nyu.edu" or not parsed.path.startswith(RAW_PREFIX):
            return
        name = PurePosixPath(parsed.path).name
        url = parsed._replace(scheme="https").geturl()
        if name.endswith(".zip"):
            self.archives[name] = url
        elif name.endswith(".md5"):
            self.checksums[name] = url

    def entries(self):
        if not self.archives:
            raise ValueError("No official Raw ZIP links found on the NYUv2 page")
        result = []
        for name, url in sorted(self.archives.items()):
            md5_name = name.removesuffix(".zip") + ".md5"
            if md5_name not in self.checksums:
                raise ValueError(f"Official MD5 link is missing: {name}")
            result.append({"name": name, "url": url, "md5_url": self.checksums[md5_name],
                           "discovery": "current_official_page"})
        return result


def parse_md5(text, name):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected one official MD5 record for {name}")
    match = re.fullmatch(r"([0-9a-fA-F]{32})\s+\*?(.+)", lines[0])
    if not match or match[2] != name:
        raise ValueError(f"Official checksum filename disagrees with ZIP: {name}")
    return match[1].lower()


def resolve_entry(entry):
    for attempt in range(3):
        try:
            text = get_text(entry["md5_url"])
            digest = parse_md5(text, entry["name"])
            with urlopen(Request(entry["url"], method="HEAD"), timeout=45) as response:
                size = int(response.headers["Content-Length"])
                if response.status != 200 or size <= 0:
                    raise ValueError(f"Missing official ZIP size: {entry['name']}")
            return {**entry, "bytes": size, "md5": digest, "official_md5_text": text}
        except (OSError, ValueError, TypeError):
            if attempt == 2:
                raise


def discover(workers=8):
    parser = RawLinks()
    parser.feed(get_text(PAGE))
    entries = parser.entries()
    names = {entry["name"] for entry in entries}
    # The current table omits classrooms.zip. Retain previously verified official parts,
    # but re-fetch their publisher MD5 and size; never substitute a third-party mirror.
    for known in read_json(KNOWN_ARCHIVES)["archives"]:
        if known["name"] in names:
            continue
        parsed = urlparse(known["url"])
        if parsed.hostname != "horatio.cs.nyu.edu" or not parsed.path.startswith(RAW_PREFIX):
            raise ValueError("Supplemental Raw archive must be on the official NYU host")
        entries.append({"name": known["name"], "url": known["url"],
                        "md5_url": known["url"].removesuffix(".zip") + ".md5",
                        "discovery": "previous_official_manifest_not_on_current_page"})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(resolve_entry, sorted(entries, key=lambda entry: entry["name"])))
