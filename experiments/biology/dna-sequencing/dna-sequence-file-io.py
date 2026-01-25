import os
import tempfile
import random
from typing import Optional


class DNA_SEQUENCE_FILE:
    """Simple file-backed DNA sequence helper.

    Stores ASCII bytes 'A','T','G','C'. Provides chunked random-generation,
    read, write (overwrite/append/at-offset), and insertion (streaming via temp file).
    """

    BASES = b'ATGC'

    def __init__(self, filename: str, filesize: int = 0):
        self.filename = filename
        self.filesize = int(filesize)

    def __str__(self) -> str:
        return f"DNA_SEQUENCE_FILE(filename={self.filename!r}, filesize={self.filesize})"

    def _fsync_file(self, f) -> None:
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # fallback/no-op on platforms that don't support fsync
            pass

    def read(self, offset: int = 0, length: Optional[int] = None) -> bytes:
        """Read up to `length` bytes from file starting at `offset`.

        If `length` is None, read to EOF.
        """
        with open(self.filename, 'rb') as f:
            f.seek(offset)
            data = f.read() if length is None else f.read(length)
        return data

    def write(self, data: bytes, offset: Optional[int] = None) -> None:
        """Write `data` to file. If `offset` is None, overwrite file; else write at offset.
        Use binary-safe operations and fsync after writing.
        """
        if offset is None:
            # overwrite entire file
            with open(self.filename, 'wb') as f:
                f.write(data)
                self._fsync_file(f)
            self.filesize = os.path.getsize(self.filename)
        else:
            # write in-place at offset (requires file exists or will be created)
            os.makedirs(os.path.dirname(self.filename) or '.', exist_ok=True)
            # open r+b if file exists else create then r+b
            mode = 'r+b' if os.path.exists(self.filename) else 'w+b'
            with open(self.filename, mode) as f:
                f.seek(offset)
                f.write(data)
                self._fsync_file(f)
            self.filesize = os.path.getsize(self.filename)

    def insert(self, offset: int, data: bytes) -> None:
        """Insert `data` at byte `offset` by streaming into a temporary file.

        For large files this is IO heavy but safe and memory-efficient.
        """
        # If file doesn't exist, insertion is just a write
        if not os.path.exists(self.filename):
            self.write(data, offset=0)
            return

        with open(self.filename, 'rb') as src:
            # Create temp file in same directory to allow atomic replace
            dirpath = os.path.dirname(self.filename) or '.'
            fd, tmp_path = tempfile.mkstemp(dir=dirpath)
            with os.fdopen(fd, 'wb') as dst:
                # copy up to offset
                to_copy = offset
                chunk = 1 << 20
                while to_copy > 0:
                    read_len = min(chunk, to_copy)
                    chunk_data = src.read(read_len)
                    if not chunk_data:
                        break
                    dst.write(chunk_data)
                    to_copy -= len(chunk_data)

                # write inserted data
                dst.write(data)

                # copy remainder
                while True:
                    chunk_data = src.read(chunk)
                    if not chunk_data:
                        break
                    dst.write(chunk_data)

                self._fsync_file(dst)

        # replace original file atomically
        os.replace(tmp_path, self.filename)
        self.filesize = os.path.getsize(self.filename)

    def generate_random(self, size: int, chunk_size: int = 1 << 20, seed: Optional[int] = None) -> None:
        """Generate a random DNA sequence of exactly `size` bytes and write to the file.

        Generation is chunked to avoid large memory usage. Each output byte is one of 'A','T','G','C'.
        """
        if seed is not None:
            random.seed(seed)

        bases = self.BASES
        # Precompute a mapping for 0-255 -> base using modulo for speed
        table = bytes(bytes([bases[b % 4] for b in range(256)]))

        with open(self.filename, 'wb') as f:
            remaining = int(size)
            while remaining > 0:
                this = min(remaining, chunk_size)
                # get cryptographic random bytes and map via table
                rnd = os.urandom(this)
                out = rnd.translate(table)
                f.write(out)
                remaining -= this
            self._fsync_file(f)

        self.filesize = os.path.getsize(self.filename)

    def insert_random_bases(self, count: int, seed: Optional[int] = None) -> None:
        """Insert `count` random single-base bytes at random offsets throughout the file.

        This performs `count` insertions; for large counts this may be slow.
        """
        if seed is not None:
            random.seed(seed)

        for _ in range(count):
            base = random.choice(self.BASES)
            # choose offset in [0, filesize]
            off = random.randint(0, max(0, self.filesize))
            self.insert(off, bytes([base]))


def _human_readable(n: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DNA sequence file utility (safe, chunked)')
    parser.add_argument('file', help='path to sequence file to create or operate on')
    parser.add_argument('--size', type=str, default=None,
                        help='generate random sequence of given size, e.g. 6GB, 1048576B')
    parser.add_argument('--generate', action='store_true', help='generate random sequence of given size')
    parser.add_argument('--insert-random', type=int, default=0, help='insert N random bases at random offsets')
    parser.add_argument('--seed', type=int, default=None, help='seed for deterministic generation')
    args = parser.parse_args()

    def parse_size(size: str) -> int:
        """Given a human-readable byte string (e.g. 2G, 10GB, 30MB, 20KB),
            return the number of bytes.  Will return 0 if the argument has
            unexpected form.
            from https://gist.github.com/beugley/ccd69945346759eb6142272a6d69b4e0
        """
        if (size[-1] == 'B'):
            size = size[:-1]
        if (size.isdigit()):
            bytes = int(size)
        else:
            bytes = size[:-1]
            unit = size[-1]
            if (bytes.isdigit()):
                bytes = int(bytes)
                if (unit == 'G'):
                    bytes *= 1073741824
                elif (unit == 'M'):
                    bytes *= 1048576
                elif (unit == 'K'):
                    bytes *= 1024
                else:
                    bytes = 0
            else:
                bytes = 0
        return bytes

    obj = DNA_SEQUENCE_FILE(args.file)
    if args.generate:
        if not args.size:
            parser.error('--generate requires --size')
        size_bytes = parse_size(args.size)
        print(f'Generating {size_bytes} bytes ({_human_readable(size_bytes)}) into {args.file}')
        obj.generate_random(size_bytes, seed=args.seed)
        print('Done:', obj)

    if args.insert_random:
        print(f'Inserting {args.insert_random} random bases into {args.file}')
        obj.filesize = os.path.getsize(args.file) if os.path.exists(args.file) else 0
        obj.insert_random_bases(args.insert_random, seed=args.seed)
        print('Done:', obj)


