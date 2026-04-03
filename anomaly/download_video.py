import requests
import os

url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/crowdhuman-test.mp4"  # Small crowd sample if available; fallback
# Alternative small crowd video from public source
urls = [
    "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",  # Generic, replace with crowd if known
]

for u in urls:
    try:
        r = requests.get(u, stream=True)
        if r.status_code
