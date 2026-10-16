import base64
import json
import os
import urllib.error
import urllib.request

with open("punjabi_test_16k.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

payload = json.dumps({"audio": audio_b64, "language": "pa"}).encode()
req = urllib.request.Request(
    "https://epunjabi-backend.fly.dev/speech-to-text",
    data=payload,
    headers={
        "Content-Type": "application/json",
        "X-API-Key": os.environ["SMOKE_TEST_API_KEY"],
    },
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        print(resp.status)
        print(resp.read().decode())
except urllib.error.HTTPError as e:
    print(e.code)
    print(e.read().decode())
