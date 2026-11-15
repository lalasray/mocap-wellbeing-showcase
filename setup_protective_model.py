"""Download the pinned published checkpoint; verify before installing."""
import hashlib
import urllib.request
from protective_model import CHECKPOINT, URL, SHA256


def main():
    if CHECKPOINT.exists() and hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == SHA256:
        print('Protective-movement checkpoint verified.')
        return
    data = urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError('Downloaded checkpoint checksum mismatch.')
    CHECKPOINT.parent.mkdir(exist_ok=True)
    CHECKPOINT.write_bytes(data)
    print('Protective-movement checkpoint installed and verified.')


if __name__ == '__main__':
    main()
