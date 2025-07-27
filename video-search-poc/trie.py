# trie.py
"""
Enhanced Trie implementation for Video Search Platform - Phase 3 Optimizations
Supports prefix matching, fuzzy search, auto-complete, and wildcard searching
Includes optimized algorithms, result caching, and performance improvements
"""

import time
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


class TrieNode:
    """Node class for the Trie data structure with optimizations"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.video_ids = []  # Store video IDs for words that end here
        self.frequency = 0   # Track frequency of searches for this word
        self.last_accessed = 0  # Track when this node was last accessed


class OptimizedTrie:
    """Enhanced Trie with fuzzy matching, caching, and performance optimizations"""
    
    def __init__(self, cache_size=1000):
        self.root = TrieNode()
        self.word_count = 0
        
        # Caching system for expensive operations
        self.cache_size = cache_size
        self.prefix_cache = {}  # Cache for prefix search results
        self.fuzzy_cache = {}   # Cache for fuzzy search results
        self.wildcard_cache = {} # Cache for wildcard search results
        self.cache_access_order = {
            'prefix': [],
            'fuzzy': [],
            'wildcard': []
        }
        
        # Performance statistics
        self.stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_search_time': 0.0,
            'fuzzy_searches': 0,
            'prefix_searches': 0,
            'wildcard_searches': 0
        }
        
        # Optimization flags
        self.enable_caching = True
        self.max_fuzzy_distance = 3  # Limit fuzzy search distance for performance
    
    def insert(self, word, video_id=None):
        """Insert word into trie with optional video ID association"""
        if not word:
            return
        
        word = word.lower().strip()
        node = self.root
        
        # Update access time
        current_time = time.time()
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.last_accessed = current_time
        
        if not node.is_end:
            self.word_count += 1
        
        node.is_end = True
        node.frequency += 1
        node.last_accessed = current_time
        
        if video_id and video_id not in node.video_ids:
            node.video_ids.append(video_id)
        
        # Clear related caches when new words are added
        self._invalidate_caches()
    
    def search_exact(self, word):
        """Search for exact word match"""
        if not word:
            return False
        
        word = word.lower().strip()
        node = self.root
        
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        return node.is_end
    
    def search_prefix(self, prefix, limit=10):
        """Find all words with given prefix, limited by count with caching"""
        if not prefix:
            return []
        
        start_time = time.time()
        prefix = prefix.lower().strip()
        
        # Check cache first
        cache_key = f"{prefix}:{limit}"
        if self.enable_caching and cache_key in self.prefix_cache:
            self.stats['cache_hits'] += 1
            self._update_cache_access('prefix', cache_key)
            self._update_stats('prefix', time.time() - start_time)
            return self.prefix_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Navigate to the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                self._update_stats('prefix', time.time() - start_time)
                return []
            node = node.children[char]
        
        # Collect words using optimized BFS
        words = self._collect_words_optimized(node, prefix, limit)
        
        # Sort by frequency and recency
        words.sort(key=lambda x: (x[1], x[2]), reverse=True)  # frequency, recency
        result = [word for word, freq, recency in words[:limit]]
        
        # Cache the result
        if self.enable_caching:
            self._update_cache('prefix', cache_key, result)
        
        self._update_stats('prefix', time.time() - start_time)
        return result
    
    def _collect_words_optimized(self, node, prefix, limit, collected=None):
        """Optimized word collection using BFS with early termination"""
        if collected is None:
            collected = []
        
        if len(collected) >= limit:
            return collected
        
        # Use BFS for better performance with large tries
        queue = [(node, prefix)]
        
        while queue and len(collected) < limit:
            current_node, current_prefix = queue.pop(0)
            
            if current_node.is_end:
                collected.append((
                    current_prefix, 
                    current_node.frequency,
                    current_node.last_accessed
                ))
            
            # Add children to queue, prioritizing frequently accessed nodes
            children = sorted(
                current_node.children.items(),
                key=lambda x: x[1].frequency,
                reverse=True
            )
            
            for char, child_node in children:
                if len(collected) < limit:
                    queue.append((child_node, current_prefix + char))
        
        return collected
    
    def auto_complete(self, partial_word, limit=5):
        """Provide auto-complete suggestions"""
        return self.search_prefix(partial_word, limit)
    
    def fuzzy_search(self, word, max_distance=None):
        """Optimized fuzzy search using dynamic programming"""
        if not word:
            return []
        
        start_time = time.time()
        word = word.lower().strip()
        
        # Use instance max_distance if not provided
        if max_distance is None:
            max_distance = min(self.max_fuzzy_distance, len(word) // 2 + 1)
        
        # Check cache first
        cache_key = f"{word}:{max_distance}"
        if self.enable_caching and cache_key in self.fuzzy_cache:
            self.stats['cache_hits'] += 1
            self._update_cache_access('fuzzy', cache_key)
            self._update_stats('fuzzy', time.time() - start_time)
            return self.fuzzy_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Use optimized Levenshtein distance with early termination
        results = self._fuzzy_search_optimized(word, max_distance)
        
        # Sort by edit distance and frequency
        results.sort(key=lambda x: (x[1], -self._get_frequency(x[0])))
        final_results = [word for word, distance in results[:20]]  # Limit results
        
        # Cache the result
        if self.enable_caching:
            self._update_cache('fuzzy', cache_key, final_results)
        
        self._update_stats('fuzzy', time.time() - start_time)
        return final_results
    
    def _fuzzy_search_optimized(self, target, max_distance):
        """Optimized fuzzy search using iterative deepening and pruning"""
        results = []
        target_len = len(target)
        
        def _dfs_with_pruning(node, current_word, pos, edits):
            # Early termination if edit distance exceeds threshold
            if edits > max_distance:
                return
            
            # If we've processed all characters and within edit distance
            if pos >= target_len:
                if node.is_end and edits <= max_distance:
                    results.append((current_word, edits))
                return
            
            # Exact match
            target_char = target[pos]
            if target_char in node.children:
                _dfs_with_pruning(
                    node.children[target_char], 
                    current_word + target_char, 
                    pos + 1, 
                    edits
                )
            
            # Only explore edit operations if we haven't exceeded the limit
            if edits < max_distance:
                # Insertion (add character from trie)
                for char, child_node in node.children.items():
                    if char != target_char:  # Don't repeat exact match
                        _dfs_with_pruning(
                            child_node, 
                            current_word + char, 
                            pos, 
                            edits + 1
                        )
                
                # Deletion (skip character in target)
                _dfs_with_pruning(node, current_word, pos + 1, edits + 1)
                
                # Substitution
                for char, child_node in node.children.items():
                    if char != target_char:
                        _dfs_with_pruning(
                            child_node, 
                            current_word + char, 
                            pos + 1, 
                            edits + 1
                        )
        
        _dfs_with_pruning(self.root, "", 0, 0)
        return results
    
    def _get_frequency(self, word):
        """Get frequency of a word"""
        node = self.root
        for char in word:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.frequency if node.is_end else 0
    
    def wildcard_search(self, pattern):
        """Optimized wildcard search with caching"""
        if not pattern:
            return []
        
        start_time = time.time()
        pattern = pattern.lower().strip()
        
        # Check cache first
        if self.enable_caching and pattern in self.wildcard_cache:
            self.stats['cache_hits'] += 1
            self._update_cache_access('wildcard', pattern)
            self._update_stats('wildcard', time.time() - start_time)
            return self.wildcard_cache[pattern]
        
        self.stats['cache_misses'] += 1
        
        results = set()  # Use set to avoid duplicates
        self._wildcard_search_optimized(self.root, pattern, 0, "", results)
        
        final_results = list(results)[:50]  # Limit results for performance
        
        # Cache the result
        if self.enable_caching:
            self._update_cache('wildcard', pattern, final_results)
        
        self._update_stats('wildcard', time.time() - start_time)
        return final_results
    
    def _wildcard_search_optimized(self, node, pattern, pattern_idx, current_word, results):
        """Optimized wildcard search with pruning"""
        if len(results) >= 50:  # Early termination for performance
            return
        
        if pattern_idx >= len(pattern):
            if node.is_end:
                results.add(current_word)
            return
        
        char = pattern[pattern_idx]
        
        if char == '*':
            # Match zero characters (skip wildcard)
            self._wildcard_search_optimized(node, pattern, pattern_idx + 1, current_word, results)
            
            # Match one or more characters
            for child_char, child_node in node.children.items():
                # Continue with wildcard (match more characters)
                self._wildcard_search_optimized(
                    child_node, pattern, pattern_idx, current_word + child_char, results
                )
                # Move past wildcard (match exactly one character)
                self._wildcard_search_optimized(
                    child_node, pattern, pattern_idx + 1, current_word + child_char, results
                )
        else:
            # Exact character match
            if char in node.children:
                self._wildcard_search_optimized(
                    node.children[char], pattern, pattern_idx + 1, current_word + char, results
                )
    
    def get_video_ids_for_word(self, word):
        """Get all video IDs associated with a word"""
        if not word:
            return []
        
        word = word.lower().strip()
        node = self.root
        
        for char in word:
            if char not in node.children:
                return []
            node = node.children[char]
        
        return node.video_ids if node.is_end else []
    
    def _update_cache(self, cache_type, key, value):
        """Update cache with LRU eviction"""
        cache = getattr(self, f'{cache_type}_cache')
        access_order = self.cache_access_order[cache_type]
        
        if key in cache:
            access_order.remove(key)
        elif len(cache) >= self.cache_size // 3:  # Divide cache among three types
            if access_order:
                lru_key = access_order.pop(0)
                del cache[lru_key]
        
        cache[key] = value
        access_order.append(key)
    
    def _update_cache_access(self, cache_type, key):
        """Update cache access order"""
        access_order = self.cache_access_order[cache_type]
        if key in access_order:
            access_order.remove(key)
            access_order.append(key)
    
    def _invalidate_caches(self):
        """Clear all caches when trie is modified"""
        self.prefix_cache.clear()
        self.fuzzy_cache.clear()
        self.wildcard_cache.clear()
        for cache_type in self.cache_access_order:
            self.cache_access_order[cache_type].clear()
    
    def _update_stats(self, operation_type, duration):
        """Update performance statistics"""
        self.stats['total_operations'] += 1
        self.stats[f'{operation_type}_searches'] += 1
        
        # Update average search time
        current_avg = self.stats['average_search_time']
        total_ops = self.stats['total_operations']
        
        self.stats['average_search_time'] = (
            (current_avg * (total_ops - 1) + duration) / total_ops
        )
    
    def optimize_memory(self):
        """Perform memory optimization operations"""
        # Clear caches
        self._invalidate_caches()
        
        # Could add node compression here for production use
        print("Memory optimization completed")
    
    def get_stats(self):
        """Get comprehensive trie statistics"""
        node_count = self._count_nodes(self.root)
        depth = self._max_depth(self.root)
        
        cache_hit_rate = (
            self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
            if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
        )
        
        return {
            'word_count': self.word_count,
            'node_count': node_count,
            'max_depth': depth,
            'memory_efficiency': self.word_count / node_count if node_count > 0 else 0,
            'cache_hit_rate': cache_hit_rate,
            'average_search_time': self.stats['average_search_time'],
            'total_operations': self.stats['total_operations'],
            'operation_breakdown': {
                'prefix_searches': self.stats['prefix_searches'],
                'fuzzy_searches': self.stats['fuzzy_searches'],
                'wildcard_searches': self.stats['wildcard_searches']
            }
        }
    
    def _count_nodes(self, node):
        """Count total nodes in trie"""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count
    
    def _max_depth(self, node, current_depth=0):
        """Find maximum depth of trie"""
        if not node.children:
            return current_depth
        
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth


# Keep original Trie class for backward compatibility
class Trie(OptimizedTrie):
    """Backward compatibility alias for OptimizedTrie"""
    pass


class VideoTrieSystem:
    """Comprehensive trie system for video search with multiple search categories and optimizations"""
    
    def __init__(self, cache_size=1000):
        # Use optimized tries for all categories
        self.title_trie = OptimizedTrie(cache_size)
        self.year_trie = OptimizedTrie(cache_size)
        self.actor_trie = OptimizedTrie(cache_size)
        self.genre_trie = OptimizedTrie(cache_size)
        self.keyword_trie = OptimizedTrie(cache_size)
        self.director_trie = OptimizedTrie(cache_size)
        
        # Performance tracking
        self.system_stats = {
            'videos_indexed': 0,
            'total_system_operations': 0,
            'last_optimization': time.time()
        }
    
    def add_video_to_tries(self, video):
        """Add video information to all relevant tries with optimization"""
        video_id = video.video_id
        
        # Add title words with better tokenization
        title_words = self._tokenize_text(video.title)
        for word in title_words:
            self.title_trie.insert(word, video_id)
        
        # Add full title
        self.title_trie.insert(video.title, video_id)

        # Add video year
        self.year_trie.insert(str(video.year), video_id)
        
        # Add actors with name parts
        for actor in video.actors:
            self.actor_trie.insert(actor, video_id)
            # Add individual name parts for better search
            name_parts = self._tokenize_text(actor)
            for part in name_parts:
                if len(part) > 2:  # Ignore very short name parts
                    self.actor_trie.insert(part, video_id)
        
        # Add genres
        for genre in video.genre:
            self.genre_trie.insert(genre, video_id)
        
        # Add keywords with stemming-like processing
        for keyword in video.keywords:
            self.keyword_trie.insert(keyword, video_id)
            # Add variations for better matching
            if len(keyword) > 4:
                self.keyword_trie.insert(keyword[:-1], video_id)  # Remove last char
        
        # Add directors with name parts
        for director in video.directors:
            self.director_trie.insert(director, video_id)
            name_parts = self._tokenize_text(director)
            for part in name_parts:
                if len(part) > 2:
                    self.director_trie.insert(part, video_id)
        
        self.system_stats['videos_indexed'] += 1
        
        # Periodic optimization
        if self.system_stats['videos_indexed'] % 100 == 0:
            self._optimize_system()
    
    def _tokenize_text(self, text):
        """Improved text tokenization"""
        import re
        # Split on spaces, punctuation, and common separators
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [word for word in words if len(word) > 2 and word not in stop_words]
    
    def search_titles(self, query, search_type='prefix'):
        """Enhanced title search with better performance"""
        if search_type == 'exact':
            video_ids = self.title_trie.get_video_ids_for_word(query)
            return [(vid, 1.0) for vid in video_ids]  # Return with relevance scores
        elif search_type == 'prefix':
            titles = self.title_trie.search_prefix(query)
            results = []
            for title in titles:
                video_ids = self.title_trie.get_video_ids_for_word(title)
                for vid in video_ids:
                    results.append((vid, 0.8))
            return results
        elif search_type == 'fuzzy':
            titles = self.title_trie.fuzzy_search(query)
            results = []
            for title in titles:
                video_ids = self.title_trie.get_video_ids_for_word(title)
                for vid in video_ids:
                    results.append((vid, 0.6))
            return results
        elif search_type == 'wildcard':
            titles = self.title_trie.wildcard_search(query)
            results = []
            for title in titles:
                video_ids = self.title_trie.get_video_ids_for_word(title)
                for vid in video_ids:
                    results.append((vid, 0.7))
            return results
        else:
            return []
    
    def search_actors(self, query, search_type='prefix'):
        """Enhanced actor search"""
        if search_type == 'exact':
            return self.actor_trie.get_video_ids_for_word(query)
        elif search_type == 'prefix':
            return self.actor_trie.search_prefix(query)
        elif search_type == 'fuzzy':
            return self.actor_trie.fuzzy_search(query)
        elif search_type == 'wildcard':
            return self.actor_trie.wildcard_search(query)
        else:
            return []


    def search_year(self, query, search_type='prefix'):
        """Enhanced actor search"""
        if search_type == 'exact':
            return self.year_trie.get_video_ids_for_word(query)
        elif search_type == 'prefix':
            return self.year_trie.search_prefix(query)
        elif search_type == 'fuzzy':
            return self.year_trie.fuzzy_search(query)
        elif search_type == 'wildcard':
            return self.year_trie.wildcard_search(query)
        else:
            return []
    
    def get_auto_complete_suggestions(self, query, category='all', limit=5):
        """Enhanced auto-complete with relevance scoring"""
        suggestions = []
        
        if category in ['all', 'title']:
            title_suggestions = self.title_trie.auto_complete(query, limit)
            suggestions.extend([('title', s) for s in title_suggestions])
        
        if category in ['all', 'actor']:
            actor_suggestions = self.actor_trie.auto_complete(query, limit)
            suggestions.extend([('actor', s) for s in actor_suggestions])
        
        if category in ['all', 'genre']:
            genre_suggestions = self.genre_trie.auto_complete(query, limit)
            suggestions.extend([('genre', s) for s in genre_suggestions])
        
        if category in ['all', 'keyword']:
            keyword_suggestions = self.keyword_trie.auto_complete(query, limit)
            suggestions.extend([('keyword', s) for s in keyword_suggestions])
        
        # Sort by relevance (frequency and recency)
        return suggestions[:limit]
    
    def _optimize_system(self):
        """Periodic system optimization"""
        print(f"Optimizing trie system (videos indexed: {self.system_stats['videos_indexed']})")
        
        # Optimize memory usage for all tries
        for trie in [self.title_trie, self.actor_trie, self.genre_trie, 
                     self.keyword_trie, self.director_trie]:
            trie.optimize_memory()
        
        self.system_stats['last_optimization'] = time.time()
    
    def get_system_stats(self):
        """Get comprehensive trie system statistics"""
        return {
            'title_trie': self.title_trie.get_stats(),
            'actor_trie': self.actor_trie.get_stats(),
            'genre_trie': self.genre_trie.get_stats(),
            'keyword_trie': self.keyword_trie.get_stats(),
            'director_trie': self.director_trie.get_stats(),
            'system_stats': self.system_stats
        }
    
    def clear_all_caches(self):
        """Clear all caches across all tries"""
        for trie in [self.title_trie, self.actor_trie, self.genre_trie, 
                     self.keyword_trie, self.director_trie]:
            trie._invalidate_caches()
