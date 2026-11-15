"""Download the authors' checkpoint; verify exact bytes before use."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
import tempfile

ROOT = Path(__file__).resolve().parent
INFO = json.loads((ROOT/'models/model_info.json').read_text())
URL = 'https://www.dropbox.com/s/ut3n356qv8pnzhg/weights.zip?dl=1'


def main():
    target = ROOT/'models/stgcn_500_5.pt'
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == INFO['checkpoint_sha256']:
        print('Published checkpoint already installed and verified.')
        return
    with tempfile.TemporaryDirectory(prefix='pose-model-') as directory:
        archive = Path(directory)/'weights.zip'
        print('Downloading published model archive (approximately 181 MiB)...',flush=True)
        with urllib.request.urlopen(URL,timeout=60) as response, archive.open('wb') as out:
            while chunk := response.read(1024*1024):
                out.write(chunk)
        with zipfile.ZipFile(archive) as z:
            payload = z.read('weights/stgcn_500_5.pt')
    if hashlib.sha256(payload).hexdigest() != INFO['checkpoint_sha256']:
        raise RuntimeError('Downloaded checkpoint does not match the verified release.')
    target.write_bytes(payload)
    print('Published ST-GCN weights installed and checksum verified.')


if __name__ == '__main__':
    main()
