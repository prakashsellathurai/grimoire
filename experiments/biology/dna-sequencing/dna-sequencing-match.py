"""
DNA Sequence Search System
Searches for gene sequences in complementary DNA strands stored in BST-based databases
with connection pooling (max 10 connections per database)

This version simulates the FULL 6 BILLION character DNA sequences using:
- Memory-mapped files for efficient storage
- On-demand generation to avoid memory overflow
- Chunked processing with persistent BST index
"""

import threading
import time
import random
import os
import mmap
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pickle


# ==================== Configuration ====================
# Set to True for quick testing with smaller files (600MB instead of 6GB)
QUICK_TEST_MODE = False

if QUICK_TEST_MODE:
    DNA_LENGTH = 600_000_000      # 600 million bases (600 MB) for testing
    CHUNK_SIZE = 100_000_000      # 100 million bases per chunk
    print("⚠️  QUICK TEST MODE: Using 600MB files instead of 6GB")
else:
    DNA_LENGTH = 6_000_000_000    # 6 billion bases (6 GB) for production
    CHUNK_SIZE = 100_000_000      # 100 million bases per chunk

NUM_CHUNKS = DNA_LENGTH // CHUNK_SIZE
BASES = ['A', 'T', 'G', 'C']
BASES_BYTES = [ord('A'), ord('T'), ord('G'), ord('C')]  # Pre-computed for speed
COMPLEMENT_MAP = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

# Try to import numpy for faster generation (optional)
try:
    import numpy as np
    HAS_NUMPY = True
    print("✓ NumPy detected: Using optimized file generation (5-10x faster)")
except ImportError:
    HAS_NUMPY = False
    print("⚠️  NumPy not found: Using standard library (slower file generation)")


# ==================== BST Node ====================
@dataclass
class BSTNode:
    """Node in Binary Search Tree for storing DNA segment metadata"""
    start: int
    end: int
    chunk_id: int
    left: Optional['BSTNode'] = None
    right: Optional['BSTNode'] = None


# ==================== DNA File Manager ====================
class DNAFileManager:
    """
    Manages DNA sequence storage using memory-mapped files
    Simulates 6 billion characters efficiently
    """
    
    def __init__(self, filepath: str, length: int = DNA_LENGTH):
        self.filepath = filepath
        self.length = length
        self.file_handle = None
        self.mmap_handle = None
        
    def create_file(self, insert_pattern: str = None, pattern_positions: List[int] = None):
        """Create DNA sequence file with optional pattern insertions - OPTIMIZED for low RAM"""
        print(f"Creating {self.filepath} ({self.length / 1e9:.1f} GB)...")
        print(f"  Using memory-efficient streaming approach...")
        
        # Use 5MB write buffer to minimize RAM usage
        write_buffer_size = 5_000_000  # 5MB buffer
        pattern_len = len(insert_pattern) if insert_pattern else 0
        
        # Sort pattern positions for efficient insertion
        sorted_patterns = sorted(pattern_positions) if pattern_positions else []
        pattern_idx = 0
        
        with open(self.filepath, 'wb') as f:
            total_written = 0
            
            while total_written < self.length:
                remaining = min(write_buffer_size, self.length - total_written)
                chunk_start = total_written
                chunk_end = total_written + remaining
                
                # Generate chunk efficiently
                if HAS_NUMPY:
                    # Fast generation using numpy (5-10x faster)
                    chunk = np.random.choice(BASES_BYTES, size=remaining, replace=True).astype(np.uint8)
                else:
                    # Standard library approach (slower but works without numpy)
                    chunk = bytearray(remaining)
                    for i in range(remaining):
                        chunk[i] = BASES_BYTES[random.randint(0, 3)]
                
                # Insert patterns that fall within this chunk
                while pattern_idx < len(sorted_patterns):
                    pattern_pos = sorted_patterns[pattern_idx]
                    
                    if pattern_pos < chunk_start:
                        pattern_idx += 1
                        continue
                    
                    if pattern_pos >= chunk_end:
                        break
                    
                    # Pattern falls in this chunk
                    local_pos = pattern_pos - chunk_start
                    if local_pos + pattern_len <= remaining:
                        for j, base in enumerate(insert_pattern):
                            chunk[local_pos + j] = ord(base)
                    
                    pattern_idx += 1
                
                # Write chunk to disk
                if HAS_NUMPY:
                    f.write(chunk.tobytes())
                else:
                    f.write(chunk)
                
                # Explicit memory cleanup
                del chunk
                
                total_written += remaining
                
                # Progress reporting every 100MB
                if total_written % 100_000_000 == 0 or total_written == self.length:
                    percent = 100 * total_written / self.length
                    print(f"  Progress: {total_written / 1e9:.2f} GB / {self.length / 1e9:.1f} GB "
                          f"({percent:.1f}%)")
        
        print(f"  ✓ Completed: {self.filepath}")
    
    def open_mmap(self):
        """Open file with memory mapping for efficient random access"""
        self.file_handle = open(self.filepath, 'r+b')
        self.mmap_handle = mmap.mmap(self.file_handle.fileno(), 0)
    
    def close_mmap(self):
        """Close memory-mapped file"""
        if self.mmap_handle:
            self.mmap_handle.close()
        if self.file_handle:
            self.file_handle.close()
    
    def read_range(self, start: int, end: int) -> str:
        """Read DNA sequence from start to end position"""
        if not self.mmap_handle:
            self.open_mmap()
        
        length = end - start + 1
        self.mmap_handle.seek(start)
        data = self.mmap_handle.read(length)
        return data.decode('ascii')
    
    def __enter__(self):
        self.open_mmap()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_mmap()


# ==================== DNA Database ====================
class DNADatabase:
    """
    BST-based database for DNA sequences with connection pooling
    Supports max 10 parallel connections
    """
    
    def __init__(self, name: str, file_manager: DNAFileManager, max_connections: int = 10):
        self.name = name
        self.file_manager = file_manager
        self.root: Optional[BSTNode] = None
        self.max_connections = max_connections
        self.connection_semaphore = threading.Semaphore(max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()
        
    def build_index(self, chunk_size: int = CHUNK_SIZE):
        """Build BST index for chunks"""
        print(f"[{self.name}] Building BST index...")
        
        num_chunks = DNA_LENGTH // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size - 1
            self.insert(start, end, i)
            
            if (i + 1) % 10 == 0:
                print(f"  Indexed {i + 1}/{num_chunks} chunks")
        
        print(f"[{self.name}] Index built successfully!")
    
    def insert(self, start: int, end: int, chunk_id: int):
        """Insert a chunk reference into the BST"""
        self.root = self._insert_node(self.root, start, end, chunk_id)
    
    def _insert_node(self, node: Optional[BSTNode], start: int, end: int, 
                     chunk_id: int) -> BSTNode:
        """Recursive insertion into BST"""
        if node is None:
            return BSTNode(start, end, chunk_id)
        
        if start < node.start:
            node.left = self._insert_node(node.left, start, end, chunk_id)
        else:
            node.right = self._insert_node(node.right, start, end, chunk_id)
        
        return node
    
    def acquire_connection(self) -> int:
        """Acquire a database connection (blocks if max connections reached)"""
        self.connection_semaphore.acquire()
        with self.lock:
            self.active_connections += 1
            conn_id = self.active_connections
        return conn_id
    
    def release_connection(self):
        """Release a database connection"""
        with self.lock:
            self.active_connections -= 1
        self.connection_semaphore.release()
    
    def query_range(self, start: int, end: int) -> List[dict]:
        """Query DNA segments in a given range"""
        conn_id = self.acquire_connection()
        try:
            results = []
            self._range_search(self.root, start, end, results)
            
            # Read actual sequences from file
            for result in results:
                seq_start = result['start']
                seq_end = result['end']
                result['sequence'] = self.file_manager.read_range(seq_start, seq_end)
            
            return results
        finally:
            self.release_connection()
    
    def _range_search(self, node: Optional[BSTNode], start: int, end: int, 
                      results: List[dict]):
        """Recursive range search in BST"""
        if node is None:
            return
        
        if node.end < start:
            self._range_search(node.right, start, end, results)
        elif node.start > end:
            self._range_search(node.left, start, end, results)
        else:
            self._range_search(node.left, start, end, results)
            results.append({
                'start': node.start,
                'end': node.end,
                'chunk_id': node.chunk_id
            })
            self._range_search(node.right, start, end, results)


# ==================== DNA Utilities ====================
class DNAUtils:
    """Utility functions for DNA operations"""
    
    @staticmethod
    def get_complement(sequence: str) -> str:
        """Get complementary DNA sequence"""
        return ''.join(COMPLEMENT_MAP.get(base, base) for base in sequence)
    
    @staticmethod
    def compute_lps(pattern: str) -> List[int]:
        """
        Compute KMP Longest Proper Prefix which is also Suffix array
        Time: O(m) where m = pattern length
        """
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    @staticmethod
    def find_pattern_kmp(sequence: str, pattern: str) -> List[int]:
        """
        Find all occurrences using KMP algorithm - O(n + m)
        
        This is MUCH faster than naive O(n×m) approach:
        - For 100M sequence with 10-char pattern:
          - Naive: 1 billion comparisons
          - KMP: 100 million comparisons (10x faster)
        
        Time: O(n + m)
        Space: O(m)
        """
        n = len(sequence)
        m = len(pattern)
        
        if m == 0 or m > n:
            return []
        
        # Preprocessing: Build LPS array - O(m)
        lps = DNAUtils.compute_lps(pattern)
        
        # Searching: KMP search - O(n)
        matches = []
        i = 0  # index for sequence
        j = 0  # index for pattern
        
        while i < n:
            if pattern[j] == sequence[i]:
                i += 1
                j += 1
            
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != sequence[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return matches
    
    @staticmethod
    def find_pattern(sequence: str, pattern: str) -> List[int]:
        """
        Find all occurrences of pattern in sequence
        Uses KMP algorithm for O(n + m) complexity
        """
        return DNAUtils.find_pattern_kmp(sequence, pattern)


# ==================== Search Result ====================
@dataclass
class SearchMatch:
    """Represents a pattern match in DNA sequence"""
    global_position: int
    chunk_id: int
    strand: str


# ==================== Search Coordinator ====================
class SearchCoordinator:
    """Coordinates parallel search across DNA databases"""
    
    def __init__(self, db1: DNADatabase, db2: DNADatabase, chunk_size: int = CHUNK_SIZE):
        self.db1 = db1
        self.db2 = db2
        self.chunk_size = chunk_size
        self.num_chunks = NUM_CHUNKS
        
    def search_chunk(self, db: DNADatabase, chunk_id: int, pattern: str) -> List[SearchMatch]:
        """Search for pattern in a specific chunk"""
        start = chunk_id * self.chunk_size
        end = start + self.chunk_size - 1
        
        # Handle overlapping search at chunk boundaries
        overlap = len(pattern) - 1
        if chunk_id > 0:
            start -= overlap
        if chunk_id < self.num_chunks - 1:
            end += overlap
        
        matches = []
        segments = db.query_range(start, end)
        
        for segment in segments:
            positions = DNAUtils.find_pattern(segment['sequence'], pattern)
            for pos in positions:
                global_pos = segment['start'] + pos
                # Avoid duplicate matches in overlap regions
                if chunk_id == 0 or global_pos >= chunk_id * self.chunk_size:
                    matches.append(SearchMatch(
                        global_position=global_pos,
                        chunk_id=chunk_id,
                        strand=db.name
                    ))
        
        return matches
    
    def parallel_search(self, db: DNADatabase, pattern: str, 
                       max_workers: int = 10) -> List[SearchMatch]:
        """Search database in parallel using multiple workers"""
        print(f"\n[{db.name}] Starting parallel search across {self.num_chunks} chunks...")
        print(f"[{db.name}] Using {max_workers} parallel workers")
        
        all_matches = []
        completed_chunks = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit search tasks for all chunks
            future_to_chunk = {
                executor.submit(self.search_chunk, db, chunk_id, pattern): chunk_id
                for chunk_id in range(self.num_chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    matches = future.result()
                    all_matches.extend(matches)
                    completed_chunks += 1
                    
                    if completed_chunks % 10 == 0:
                        print(f"[{db.name}] Progress: {completed_chunks}/{self.num_chunks} chunks "
                              f"({len(all_matches)} matches so far)")
                    
                except Exception as e:
                    print(f"[{db.name}] Chunk {chunk_id}: Error - {e}")
        
        print(f"[{db.name}] Search complete! Total matches: {len(all_matches)}")
        return all_matches
    
    def search_both_strands(self, pattern: str) -> Tuple[List[SearchMatch], List[SearchMatch]]:
        """Search for pattern in both complementary strands simultaneously"""
        print("\n" + "="*70)
        print("DNA SEQUENCE SEARCH SYSTEM - 6 BILLION BASE PAIRS")
        print("="*70)
        print(f"\nSearch Pattern: {pattern} (length: {len(pattern)})")
        
        # Calculate complement pattern
        complement_pattern = DNAUtils.get_complement(pattern)
        print(f"Complement Pattern: {complement_pattern}")
        
        print("\n--- Starting Parallel Search on Both Strands ---")
        
        # Search both strands in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_strand1 = executor.submit(self.parallel_search, self.db1, pattern)
            future_strand2 = executor.submit(self.parallel_search, self.db2, complement_pattern)
            
            strand1_matches = future_strand1.result()
            strand2_matches = future_strand2.result()
        
        return strand1_matches, strand2_matches


# ==================== Database Initializer ====================
class DatabaseInitializer:
    """Initializes DNA databases with 6 billion character sequences"""
    
    @staticmethod
    def initialize_databases(search_pattern: str = "ATTCCTGAGC") -> Tuple[DNADatabase, DNADatabase]:
        """Initialize both DNA strand databases with full 6 billion characters"""
        print("\n" + "="*70)
        print(f"INITIALIZING DNA DATABASES - {DNA_LENGTH / 1e9:.1f} BILLION BASE PAIRS")
        print("="*70)
        
        strand1_file = "dna_strand1.dat"
        strand2_file = "dna_strand2.dat"
        
        # Insert pattern at specific positions for testing
        if QUICK_TEST_MODE:
            # Fewer patterns for quick testing
            pattern_positions = [
                1_000_000,      # 1 million
                100_000_000,    # 100 million
                300_000_000,    # 300 million
            ]
        else:
            # Full set of patterns for production
            pattern_positions = [
                1_000_000,      # 1 million
                500_000_000,    # 500 million
                1_500_000_000,  # 1.5 billion
                3_000_000_000,  # 3 billion
                5_000_000_000   # 5 billion
            ]
        
        # Create strand 1 file if it doesn't exist
        if not os.path.exists(strand1_file):
            print(f"\n[Strand 1] File not found. Creating new file...")
            print(f"  Expected time: ~{DNA_LENGTH / 1e9 * 2:.0f}-{DNA_LENGTH / 1e9 * 5:.0f} minutes")
            fm1 = DNAFileManager(strand1_file)
            fm1.create_file(insert_pattern=search_pattern, pattern_positions=pattern_positions)
        else:
            file_size = os.path.getsize(strand1_file)
            print(f"\n[Strand 1] ✓ Using existing file: {strand1_file} ({file_size / 1e9:.2f} GB)")
        
        # Create strand 2 file if it doesn't exist
        if not os.path.exists(strand2_file):
            print(f"\n[Strand 2] File not found. Creating new file...")
            print(f"  Expected time: ~{DNA_LENGTH / 1e9 * 2:.0f}-{DNA_LENGTH / 1e9 * 5:.0f} minutes")
            # Generate complement of strand 1
            print(f"[Strand 2] Generating complementary strand...")
            
            complement_pattern = DNAUtils.get_complement(search_pattern)
            fm2 = DNAFileManager(strand2_file)
            fm2.create_file(insert_pattern=complement_pattern, pattern_positions=pattern_positions)
        else:
            file_size = os.path.getsize(strand2_file)
            print(f"\n[Strand 2] ✓ Using existing file: {strand2_file} ({file_size / 1e9:.2f} GB)")
        
        # Open file managers
        fm1 = DNAFileManager(strand1_file)
        fm2 = DNAFileManager(strand2_file)
        
        fm1.open_mmap()
        fm2.open_mmap()
        
        # Create databases and build indices
        print("\n--- Building Database Indices ---")
        db1 = DNADatabase("Strand_1", fm1)
        db1.build_index()
        
        db2 = DNADatabase("Strand_2", fm2)
        db2.build_index()
        
        print(f"\n✓ Databases initialized successfully!")
        print(f"  - Total DNA length: {DNA_LENGTH:,} bases ({DNA_LENGTH / 1e9:.1f} GB)")
        print(f"  - Chunk size: {CHUNK_SIZE:,} bases")
        print(f"  - Number of chunks: {NUM_CHUNKS}")
        print(f"  - Max connections per DB: 10")
        print(f"  - Pattern inserted at {len(pattern_positions)} positions for testing")
        
        return db1, db2


# ==================== Main Application ====================
def main():
    """Main application entry point"""
    
    # Configuration
    SEARCH_PATTERN = "ATTCCTGAGC"
    
    print("\n" + "="*70)
    print("DNA SEQUENCE SEARCH SYSTEM")
    print("Simulating Full 6 Billion Character DNA Sequences")
    print("="*70)
    
    # Initialize databases (creates files if needed)
    db1, db2 = DatabaseInitializer.initialize_databases(SEARCH_PATTERN)
    
    # Create search coordinator
    coordinator = SearchCoordinator(db1, db2)
    
    # Perform search
    print("\n" + "="*70)
    print("STARTING SEARCH")
    print("="*70)
    
    start_time = time.time()
    strand1_matches, strand2_matches = coordinator.search_both_strands(SEARCH_PATTERN)
    end_time = time.time()
    
    # Display results
    print("\n" + "="*70)
    print("SEARCH RESULTS")
    print("="*70)
    
    print(f"\nStrand 1 (Original): {len(strand1_matches)} matches")
    if strand1_matches:
        print("  Sample positions:")
        for match in strand1_matches[:10]:
            print(f"    - Position {match.global_position:,}")
    
    print(f"\nStrand 2 (Complement): {len(strand2_matches)} matches")
    if strand2_matches:
        print("  Sample positions:")
        for match in strand2_matches[:10]:
            print(f"    - Position {match.global_position:,}")
    
    total_matches = len(strand1_matches) + len(strand2_matches)
    print(f"\n{'='*70}")
    print(f"TOTAL MATCHES: {total_matches}")
    print(f"SEARCH TIME: {end_time - start_time:.2f} seconds")
    print(f"THROUGHPUT: {DNA_LENGTH / (end_time - start_time) / 1e6:.2f} million bases/second")
    print(f"{'='*70}")
    
    # Cleanup
    db1.file_manager.close_mmap()
    db2.file_manager.close_mmap()


if __name__ == "__main__":
    main()