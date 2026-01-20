"""
Advanced Pattern Matching Algorithms for DNA Search
Implements multiple O(n) and near-O(n) algorithms for efficient searching

Algorithms Included:
1. Knuth-Morris-Pratt (KMP) - O(n + m)
2. Boyer-Moore - O(n/m) average, O(n×m) worst
3. Rabin-Karp - O(n + m) average with rolling hash
4. Aho-Corasick - O(n + m + z) for multiple patterns
5. Suffix Array - O(n log n) preprocessing, O(m log n) search
"""

from typing import List, Dict, Tuple, Set
from collections import deque, defaultdict
import hashlib


# ==================== 1. Knuth-Morris-Pratt (KMP) Algorithm ====================
class KMPMatcher:
    """
    KMP Algorithm - O(n + m) time complexity
    Best for: Single pattern search with guaranteed linear time
    
    Preprocessing: O(m)
    Search: O(n)
    Total: O(n + m)
    """
    
    @staticmethod
    def compute_lps(pattern: str) -> List[int]:
        """
        Compute Longest Proper Prefix which is also Suffix (LPS) array
        Time: O(m)
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
    def search(text: str, pattern: str) -> List[int]:
        """
        KMP pattern search
        Time: O(n + m)
        Space: O(m)
        """
        n = len(text)
        m = len(pattern)
        
        if m == 0 or m > n:
            return []
        
        # Preprocessing
        lps = KMPMatcher.compute_lps(pattern)
        
        # Searching
        matches = []
        i = 0  # index for text
        j = 0  # index for pattern
        
        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return matches


# ==================== 2. Boyer-Moore Algorithm ====================
class BoyerMooreMatcher:
    """
    Boyer-Moore Algorithm - O(n/m) average case, O(n×m) worst case
    Best for: Long patterns, especially with large alphabet (DNA has only 4 chars)
    
    Average: O(n/m) - can skip characters
    Worst: O(n×m)
    """
    
    @staticmethod
    def build_bad_char_table(pattern: str) -> Dict[str, int]:
        """
        Build bad character heuristic table
        Time: O(m)
        """
        m = len(pattern)
        bad_char = {}
        
        for i in range(m):
            bad_char[pattern[i]] = i
        
        return bad_char
    
    @staticmethod
    def search(text: str, pattern: str) -> List[int]:
        """
        Boyer-Moore pattern search with bad character heuristic
        Average: O(n/m), Worst: O(n×m)
        """
        n = len(text)
        m = len(pattern)
        
        if m == 0 or m > n:
            return []
        
        bad_char = BoyerMooreMatcher.build_bad_char_table(pattern)
        matches = []
        s = 0  # shift of the pattern
        
        while s <= n - m:
            j = m - 1
            
            # Keep reducing j while characters match
            while j >= 0 and pattern[j] == text[s + j]:
                j -= 1
            
            if j < 0:
                # Pattern found
                matches.append(s)
                # Shift pattern to align with next character
                s += m if s + m >= n else m - bad_char.get(text[s + m], -1)
            else:
                # Shift pattern based on bad character
                s += max(1, j - bad_char.get(text[s + j], -1))
        
        return matches


# ==================== 3. Rabin-Karp Algorithm ====================
class RabinKarpMatcher:
    """
    Rabin-Karp Algorithm - O(n + m) average case
    Best for: Multiple pattern search, uses rolling hash
    
    Average: O(n + m)
    Worst: O(n×m) - rare hash collisions
    """
    
    def __init__(self, base: int = 256, prime: int = 101):
        """
        Initialize with base and prime for hashing
        For DNA (4 characters), base=4 works well
        """
        self.base = base
        self.prime = prime
    
    def hash(self, s: str, m: int) -> int:
        """Compute hash value for string of length m"""
        h = 0
        for i in range(m):
            h = (h * self.base + ord(s[i])) % self.prime
        return h
    
    def search(self, text: str, pattern: str) -> List[int]:
        """
        Rabin-Karp pattern search using rolling hash
        Time: O(n + m) average
        """
        n = len(text)
        m = len(pattern)
        
        if m == 0 or m > n:
            return []
        
        pattern_hash = self.hash(pattern, m)
        text_hash = self.hash(text, m)
        
        # Precompute base^(m-1) % prime
        h = pow(self.base, m - 1, self.prime)
        
        matches = []
        
        for i in range(n - m + 1):
            # Check if hashes match
            if pattern_hash == text_hash:
                # Verify actual string (avoid false positives)
                if text[i:i + m] == pattern:
                    matches.append(i)
            
            # Calculate rolling hash for next window
            if i < n - m:
                text_hash = (self.base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % self.prime
                # Handle negative values
                if text_hash < 0:
                    text_hash += self.prime
        
        return matches


# ==================== 4. Aho-Corasick Algorithm ====================
class AhoCorasickMatcher:
    """
    Aho-Corasick Algorithm - O(n + m + z) where z = number of matches
    Best for: Searching multiple patterns simultaneously
    
    Preprocessing: O(m) where m = sum of all pattern lengths
    Search: O(n + z)
    """
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.output = []  # Patterns that end here
            self.fail = None  # Failure link
    
    def __init__(self):
        self.root = self.TrieNode()
    
    def add_pattern(self, pattern: str, pattern_id: int = 0):
        """Add pattern to trie"""
        node = self.root
        for char in pattern:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        node.output.append((pattern, pattern_id))
    
    def build_failure_links(self):
        """Build failure links using BFS"""
        queue = deque()
        
        # Initialize failure links for root's children
        for child in self.root.children.values():
            child.fail = self.root
            queue.append(child)
        
        # BFS to build failure links
        while queue:
            current = queue.popleft()
            
            for char, child in current.children.items():
                queue.append(child)
                
                # Find failure link
                fail_node = current.fail
                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail
                
                child.fail = fail_node.children[char] if fail_node else self.root
                
                # Merge outputs
                child.output.extend(child.fail.output)
    
    def search(self, text: str) -> Dict[str, List[int]]:
        """
        Search for all patterns in text
        Returns: {pattern: [positions]}
        Time: O(n + z)
        """
        results = defaultdict(list)
        node = self.root
        
        for i, char in enumerate(text):
            # Follow failure links if character not found
            while node is not None and char not in node.children:
                node = node.fail
            
            if node is None:
                node = self.root
                continue
            
            node = node.children[char]
            
            # Report all patterns ending at this position
            for pattern, pattern_id in node.output:
                position = i - len(pattern) + 1
                results[pattern].append(position)
        
        return dict(results)


# ==================== 5. Optimized DNA-Specific Search ====================
class DNAOptimizedMatcher:
    """
    DNA-specific optimizations using 2-bit encoding
    DNA has only 4 bases: A, T, G, C
    Can encode each base in 2 bits for faster comparison
    
    Time: O(n + m)
    Space: O(n/4) - compressed representation
    """
    
    DNA_ENCODE = {'A': 0b00, 'T': 0b01, 'G': 0b10, 'C': 0b11}
    DNA_DECODE = {0b00: 'A', 0b01: 'T', 0b10: 'G', 0b11: 'C'}
    
    @staticmethod
    def encode_sequence(sequence: str) -> int:
        """
        Encode DNA sequence into integer using 2 bits per base
        Example: "ATGC" -> 0b00011011
        """
        encoded = 0
        for base in sequence:
            encoded = (encoded << 2) | DNAOptimizedMatcher.DNA_ENCODE.get(base, 0)
        return encoded
    
    @staticmethod
    def search_encoded(text: str, pattern: str) -> List[int]:
        """
        Search using bit-encoded DNA sequences
        Time: O(n + m)
        """
        n = len(text)
        m = len(pattern)
        
        if m == 0 or m > n:
            return []
        
        # Encode pattern
        pattern_encoded = DNAOptimizedMatcher.encode_sequence(pattern)
        mask = (1 << (2 * m)) - 1  # Mask to keep only m bases
        
        # Encode first window
        window_encoded = DNAOptimizedMatcher.encode_sequence(text[:m])
        
        matches = []
        if window_encoded == pattern_encoded:
            matches.append(0)
        
        # Rolling window with bit operations
        for i in range(1, n - m + 1):
            # Shift left by 2 bits (remove leftmost base)
            window_encoded = ((window_encoded << 2) & mask) | DNAOptimizedMatcher.DNA_ENCODE.get(text[i + m - 1], 0)
            
            if window_encoded == pattern_encoded:
                matches.append(i)
        
        return matches


# ==================== Benchmark and Comparison ====================
def benchmark_algorithms(text: str, pattern: str):
    """
    Benchmark all algorithms on given text and pattern
    """
    import time
    
    algorithms = {
        'Naive': lambda t, p: naive_search(t, p),
        'KMP': lambda t, p: KMPMatcher.search(t, p),
        'Boyer-Moore': lambda t, p: BoyerMooreMatcher.search(t, p),
        'Rabin-Karp': lambda t, p: RabinKarpMatcher().search(t, p),
        'DNA-Optimized': lambda t, p: DNAOptimizedMatcher.search_encoded(t, p),
    }
    
    print("="*70)
    print("PATTERN MATCHING ALGORITHM BENCHMARK")
    print("="*70)
    print(f"Text length: {len(text):,} bases")
    print(f"Pattern length: {len(pattern)} bases")
    print(f"Pattern: {pattern}")
    print()
    
    results = {}
    for name, algorithm in algorithms.items():
        start = time.time()
        matches = algorithm(text, pattern)
        end = time.time()
        
        elapsed = end - start
        results[name] = {
            'matches': len(matches),
            'time': elapsed,
            'speed': len(text) / elapsed / 1e6 if elapsed > 0 else float('inf')
        }
        
        print(f"{name:15} | {elapsed*1000:8.2f} ms | {results[name]['speed']:8.2f} MB/s | {len(matches):6} matches")
    
    print("="*70)
    
    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    print(f"\n🏆 Fastest: {fastest[0]} ({fastest[1]['time']*1000:.2f} ms)")
    
    return results


def naive_search(text: str, pattern: str) -> List[int]:
    """Naive O(n×m) search for comparison"""
    matches = []
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            matches.append(i)
    return matches


# ==================== Recommended Algorithm Selector ====================
def get_recommended_algorithm(pattern_length: int, num_patterns: int = 1) -> str:
    """
    Recommend best algorithm based on use case
    
    Args:
        pattern_length: Length of search pattern
        num_patterns: Number of different patterns to search
    
    Returns:
        Recommended algorithm name and rationale
    """
    if num_patterns > 1:
        return "Aho-Corasick", "Multiple patterns - build trie once, search all simultaneously"
    
    if pattern_length <= 5:
        return "DNA-Optimized", "Very short pattern - bit encoding is fastest"
    
    if pattern_length <= 20:
        return "KMP", "Medium pattern - guaranteed O(n+m) with low overhead"
    
    if pattern_length > 20:
        return "Boyer-Moore", "Long pattern - can skip characters efficiently"
    
    return "KMP", "Default choice - reliable O(n+m) performance"


# ==================== Example Usage ====================
if __name__ == "__main__":
    # Generate test DNA sequence
    import random
    
    bases = ['A', 'T', 'G', 'C']
    test_size = 10_000_000  # 10 MB test
    
    print("Generating test DNA sequence...")
    text = ''.join(random.choice(bases) for _ in range(test_size))
    
    # Insert pattern at known positions for verification
    pattern = "ATTCCTGAGC"
    insert_positions = [100000, 500000, 1000000, 5000000]
    
    for pos in insert_positions:
        if pos + len(pattern) <= len(text):
            text = text[:pos] + pattern + text[pos + len(pattern):]
    
    print(f"Test sequence: {test_size:,} bases")
    print(f"Pattern: {pattern}")
    print(f"Known positions: {insert_positions}")
    print()
    
    # Run benchmark
    results = benchmark_algorithms(text, pattern)
    
    # Verify all algorithms found same matches
    print("\nVerification:")
    match_counts = [r['matches'] for r in results.values()]
    if len(set(match_counts)) == 1:
        print("✓ All algorithms found the same number of matches")
    else:
        print("⚠ Mismatch in results!")
        for name, result in results.items():
            print(f"  {name}: {result['matches']} matches")
    
    # Get recommendation
    algo_name, rationale = get_recommended_algorithm(len(pattern))
    print(f"\n💡 Recommended for this pattern: {algo_name}")
    print(f"   Reason: {rationale}")