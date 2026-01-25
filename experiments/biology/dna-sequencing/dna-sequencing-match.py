"""
DNA Sequence Search System
Searches for gene sequences in complementary DNA strands stored in dna sequence 6GB
simulate DB with connection pooling (max 10 connections per database)

"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable, List, Optional


def _human_readable(n: int) -> str:
	for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
		if n < 1024:
			return f"{n:.2f}{unit}"
		n /= 1024
	return f"{n:.2f}PB"


def reverse_complement(seq: bytes) -> bytes:
	"""Return the reverse-complement of a DNA sequence bytes (A,T,G,C)."""
	table = bytes.maketrans(b'ATGCatgc', b'TACGtacg')
	return seq.translate(table)[::-1]


def normalize_pattern(p: str) -> bytes:
	return p.strip().upper().encode('ascii')


def stream_search(file_path: str, pattern: bytes, chunk_size: int = 1 << 20) -> Iterable[int]:
	"""Yield file offsets where `pattern` occurs in `file_path` using streaming reads.

	This is memory efficient and handles matches that span chunk boundaries by
	carrying an overlap of len(pattern)-1 bytes between chunks.
	"""
	if not pattern:
		return

	patlen = len(pattern)
	overlap = patlen - 1

	with open(file_path, 'rb') as f:
		pos = 0
		tail = b''
		while True:
			chunk = f.read(chunk_size)
			if not chunk:
				break
			buf = tail + chunk
			start = 0
			while True:
				idx = buf.find(pattern, start)
				if idx == -1:
					break
				yield pos - len(tail) + idx
				start = idx + 1

			# prepare tail for next chunk
			if overlap > 0:
				tail = buf[-overlap:]
			else:
				tail = b''
			pos += len(chunk)


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description='Streaming DNA sequence search')
	parser.add_argument('--file', '-f', help='Path to DNA sequence file', required=False)
	parser.add_argument('--pattern', '-p', help='DNA pattern to search for (e.g. ATG)', required=False)
	parser.add_argument('--pattern-file', help='File containing pattern (first line used)')
	parser.add_argument('--reverse', '-r', action='store_true', help='Also search reverse-complement')
	parser.add_argument('--chunk-size', type=int, default=1 << 20, help='Read chunk size in bytes')
	parser.add_argument('--max-results', type=int, default=0, help='Stop after this many results (0 = unlimited)')
	parser.add_argument('--example', action='store_true', help='Create a tiny example file and run a sample search')

	args = parser.parse_args(argv)

	if args.example:
		sample = b'AAATGCCCATGTTTATGCGTATG'
		sample_path = os.path.join(os.getcwd(), 'sample_dna.txt')
		with open(sample_path, 'wb') as f:
			f.write(sample)
		size = os.path.getsize(sample_path)
		print(f'Wrote example file: {sample_path} (size: {_human_readable(size)})')
		args.file = sample_path
		if not args.pattern:
			args.pattern = 'ATG'

	if not args.file:
		parser.error('--file is required (or use --example)')

	if args.pattern_file:
		with open(args.pattern_file, 'r', encoding='utf-8') as pf:
			args.pattern = pf.readline().strip()

	if not args.pattern:
		parser.error('--pattern is required (or use --pattern-file)')

	pattern = normalize_pattern(args.pattern)
	if any(c not in b'ATGC' for c in pattern):
		print('Warning: pattern contains non-ATGC characters; results may be unexpected')

	matches_found = 0
	file_size = os.path.getsize(args.file) if os.path.exists(args.file) else 0
	print(f"Searching {args.file} for pattern {pattern.decode()} (chunk={args.chunk_size}, size={_human_readable(file_size)})")
	start_time = time.time()

	for off in stream_search(args.file, pattern, chunk_size=args.chunk_size):
		matches_found += 1
		if args.max_results and matches_found >= args.max_results:
			break

	if args.reverse:
		rc = reverse_complement(pattern)
		print(f"Searching reverse-complement: {rc.decode()}")
		for off in stream_search(args.file, rc, chunk_size=args.chunk_size):
			matches_found += 1
			if args.max_results and matches_found >= args.max_results:
				break

	duration = time.time() - start_time
	if duration > 0:
		rate = matches_found / duration
		print(f"Search complete: {matches_found} matches in {duration:.2f}s ({rate:.2f} matches/s)")
	else:
		print(f"Search complete: {matches_found} matches (duration too short to measure)")
	return 0


if __name__ == '__main__':
	raise SystemExit(main())

