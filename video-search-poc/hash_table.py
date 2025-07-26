# hash_table.py
"""
Enhanced Hash Table implementation for Video Search Platform - Phase 3 Optimizations
Supports video metadata storage, actor indexing, genre indexing, and keyword indexing
Includes dynamic resizing, caching, and performance optimizations
"""

import time
from typing import Dict, List, Any, Optional, Tuple


class Video:
    """Represents a video with comprehensive metadata"""
    def __init__(self, video_id, title, year, genre, actors, directors, keywords, rating=0.0, description=""):
        self.video_id = video_id
        self.title = title
        self.year = year
        self.genre = genre if isinstance(genre, list) else [genre]
        self.actors = actors if isinstance(actors, list) else [actors]
        self.directors = directors if isinstance(directors, list) else [directors]
        self.keywords = keywords if isinstance(keywords, list) else [keywords]
        self.rating = rating
        self.description = description
    
    def __str__(self):
        return f"{self.title} ({self.year}) - {', '.join(self.genre)} - Rating: {self.rating}"
    
    def to_dict(self):
        """Convert video object to dictionary for easier handling"""
        return {
            'video_id': self.video_id,
            'title': self.title,
            'year': self.year,
            'genre': self.genre,
            'actors': self.actors,
            'directors': self.directors,
            'keywords': self.keywords,
            'rating': self.rating,
            'description': self.description
        }
    
    def __hash__(self):
        """Enable Video objects to be used as hash keys"""
        return hash(self.video_id)
    
    def __eq__(self, other):
        """Enable equality comparison for Video objects"""
        if not isinstance(other, Video):
            return False
        return self.video_id == other.video_id


class OptimizedHashTable:
    """Enhanced Hash Table with dynamic resizing, caching, and performance optimizations"""
    
    def __init__(self, initial_size=1000, load_factor_threshold=0.75, cache_size=100):
        self.size = initial_size
        self.table = [[] for _ in range(self.size)]
        self.count = 0
        self.load_factor_threshold = load_factor_threshold
        
        # Performance optimizations
        self.cache_size = cache_size
        self.cache = {}  # LRU cache for frequently accessed items
        self.cache_access_order = []  # Track access order for LRU
        
        # Performance metrics
        self.stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'resize_operations': 0,
            'collision_count': 0,
            'average_lookup_time': 0.0
        }
    
    def _hash(self, key):
        """Improved hash function with better distribution"""
        if isinstance(key, str):
            # Use polynomial rolling hash for better distribution
            hash_value = 0
            prime = 31
            for i, char in enumerate(key):
                hash_value = (hash_value + ord(char) * (prime ** i)) % self.size
            return hash_value
        elif isinstance(key, int):
            # Use multiplicative hashing for integers
            A = 0.6180339887  # (sqrt(5) - 1) / 2
            return int(self.size * ((key * A) % 1))
        else:
            return hash(key) % self.size
    
    def _resize(self):
        """Dynamically resize the hash table when load factor exceeds threshold"""
        old_table = self.table
        old_size = self.size
        
        # Double the size
        self.size = old_size * 2
        self.table = [[] for _ in range(self.size)]
        old_count = self.count
        self.count = 0
        
        # Clear cache as hash values will change
        self.cache.clear()
        self.cache_access_order.clear()
        
        # Rehash all existing items
        for bucket in old_table:
            for key, value in bucket:
                self._insert_without_resize(key, value)
        
        self.stats['resize_operations'] += 1
        print(f"Hash table resized from {old_size} to {self.size} (items: {old_count})")
    
    def _insert_without_resize(self, key, value):
        """Insert without triggering resize (used during resize operation)"""
        index = self._hash(key)
        bucket = self.table[index]
        
        # Check for existing key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                self._update_cache(key, value)
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self.count += 1
        
        if len(bucket) > 1:
            self.stats['collision_count'] += 1
        
        self._update_cache(key, value)
    
    def insert(self, key, value):
        """Insert key-value pair with dynamic resizing and caching"""
        start_time = time.time()
        
        # Check if resize is needed
        if self.count / self.size > self.load_factor_threshold:
            self._resize()
        
        self._insert_without_resize(key, value)
        
        # Update performance metrics
        self.stats['total_operations'] += 1
        lookup_time = time.time() - start_time
        self._update_average_lookup_time(lookup_time)
    
    def search(self, key):
        """Search for value by key with caching"""
        start_time = time.time()
        
        # Check cache first
        if key in self.cache:
            self.stats['cache_hits'] += 1
            self._update_cache_access(key)
            value = self.cache[key]
            self._update_average_lookup_time(time.time() - start_time)
            return value
        
        # Cache miss - search in table
        self.stats['cache_misses'] += 1
        index = self._hash(key)
        
        for k, v in self.table[index]:
            if k == key:
                self._update_cache(key, v)
                self._update_average_lookup_time(time.time() - start_time)
                return v
        
        self._update_average_lookup_time(time.time() - start_time)
        return None
    
    def delete(self, key):
        """Delete key-value pair"""
        index = self._hash(key)
        bucket = self.table[index]
        
        for i, (k, _) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.count -= 1
                
                # Remove from cache
                if key in self.cache:
                    del self.cache[key]
                    self.cache_access_order.remove(key)
                
                return True
        return False
    
    def _update_cache(self, key, value):
        """Update LRU cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache_access_order.remove(key)
        elif len(self.cache) >= self.cache_size:
            # Remove least recently used item
            lru_key = self.cache_access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.cache_access_order.append(key)
    
    def _update_cache_access(self, key):
        """Update cache access order for LRU"""
        if key in self.cache_access_order:
            self.cache_access_order.remove(key)
            self.cache_access_order.append(key)
    
    def _update_average_lookup_time(self, lookup_time):
        """Update average lookup time statistics"""
        current_avg = self.stats['average_lookup_time']
        total_ops = self.stats['total_operations']
        
        if total_ops > 0:
            self.stats['average_lookup_time'] = (
                (current_avg * (total_ops - 1) + lookup_time) / total_ops
            )
    
    def get_all_values(self):
        """Get all values stored in the hash table"""
        values = []
        for bucket in self.table:
            for _, value in bucket:
                values.append(value)
        return values
    
    def get_all_keys(self):
        """Get all keys stored in the hash table"""
        keys = []
        for bucket in self.table:
            for key, _ in bucket:
                keys.append(key)
        return keys
    
    def get_stats(self):
        """Get comprehensive hash table statistics"""
        used_buckets = sum(1 for bucket in self.table if bucket)
        max_bucket_size = max(len(bucket) for bucket in self.table) if self.table else 0
        avg_bucket_size = self.count / used_buckets if used_buckets > 0 else 0
        
        cache_hit_rate = (
            self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
            if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
        )
        
        return {
            'total_items': self.count,
            'table_size': self.size,
            'used_buckets': used_buckets,
            'load_factor': self.count / self.size,
            'max_bucket_size': max_bucket_size,
            'avg_bucket_size': avg_bucket_size,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache),
            'resize_count': self.stats['resize_operations'],
            'collision_count': self.stats['collision_count'],
            'average_lookup_time': self.stats['average_lookup_time']
        }
    
    def clear_cache(self):
        """Clear the cache (useful for memory management)"""
        self.cache.clear()
        self.cache_access_order.clear()


# Keep the original HashTable class for backward compatibility
class HashTable(OptimizedHashTable):
    """Backward compatibility alias for OptimizedHashTable"""
    pass


class VideoMetadataStore:
    """Comprehensive video metadata storage system using optimized hash tables"""
    
    def __init__(self, cache_enabled=True):
        # Use optimized hash tables with different sizes based on expected load
        self.videos = OptimizedHashTable(1000, cache_size=200)  # Larger cache for videos
        
        # Index tables with appropriate sizes
        self.actor_index = OptimizedHashTable(500, cache_size=100)
        self.genre_index = OptimizedHashTable(100, cache_size=50)
        self.director_index = OptimizedHashTable(300, cache_size=75)
        self.keyword_index = OptimizedHashTable(800, cache_size=150)
        self.year_index = OptimizedHashTable(200, cache_size=50)
        
        # Performance tracking
        self.cache_enabled = cache_enabled
        self.operation_stats = {
            'total_videos_added': 0,
            'total_searches': 0,
            'average_add_time': 0.0,
            'average_search_time': 0.0
        }
    
    def add_video(self, video):
        """Add a video and update all relevant indexes with performance tracking"""
        if not isinstance(video, Video):
            raise TypeError("Expected Video object")
        
        start_time = time.time()
        
        # Store the video object
        self.videos.insert(video.video_id, video)
        
        # Update actor index
        for actor in video.actors:
            actor_lower = actor.lower()
            video_list = self.actor_index.search(actor_lower) or []
            if video.video_id not in video_list:
                video_list.append(video.video_id)
            self.actor_index.insert(actor_lower, video_list)
        
        # Update genre index
        for genre in video.genre:
            genre_lower = genre.lower()
            video_list = self.genre_index.search(genre_lower) or []
            if video.video_id not in video_list:
                video_list.append(video.video_id)
            self.genre_index.insert(genre_lower, video_list)
        
        # Update director index
        for director in video.directors:
            director_lower = director.lower()
            video_list = self.director_index.search(director_lower) or []
            if video.video_id not in video_list:
                video_list.append(video.video_id)
            self.director_index.insert(director_lower, video_list)
        
        # Update keyword index
        for keyword in video.keywords:
            keyword_lower = keyword.lower()
            video_list = self.keyword_index.search(keyword_lower) or []
            if video.video_id not in video_list:
                video_list.append(video.video_id)
            self.keyword_index.insert(keyword_lower, video_list)
        
        # Update year index
        year_list = self.year_index.search(video.year) or []
        if video.video_id not in year_list:
            year_list.append(video.video_id)
        self.year_index.insert(video.year, year_list)
        
        # Update performance statistics
        add_time = time.time() - start_time
        self.operation_stats['total_videos_added'] += 1
        self._update_average_add_time(add_time)
    
    def get_video(self, video_id):
        """Retrieve video by ID with performance tracking"""
        start_time = time.time()
        result = self.videos.search(video_id)
        search_time = time.time() - start_time
        
        self.operation_stats['total_searches'] += 1
        self._update_average_search_time(search_time)
        
        return result
    
    def search_by_actor(self, actor_name):
        """Find all videos with a specific actor"""
        start_time = time.time()
        video_ids = self.actor_index.search(actor_name.lower()) or []
        results = [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
        
        search_time = time.time() - start_time
        self.operation_stats['total_searches'] += 1
        self._update_average_search_time(search_time)
        
        return results
    
    def search_by_genre(self, genre):
        """Find all videos of a specific genre"""
        start_time = time.time()
        video_ids = self.genre_index.search(genre.lower()) or []
        results = [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
        
        search_time = time.time() - start_time
        self.operation_stats['total_searches'] += 1
        self._update_average_search_time(search_time)
        
        return results
    
    def search_by_director(self, director_name):
        """Find all videos by a specific director"""
        start_time = time.time()
        video_ids = self.director_index.search(director_name.lower()) or []
        results = [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
        
        search_time = time.time() - start_time
        self.operation_stats['total_searches'] += 1
        self._update_average_search_time(search_time)
        
        return results
    
    def search_by_keyword(self, keyword):
        """Find all videos with a specific keyword"""
        start_time = time.time()
        video_ids = self.keyword_index.search(keyword.lower()) or []
        results = [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
        
        search_time = time.time() - start_time
        self.operation_stats['total_searches'] += 1
        self._update_average_search_time(search_time)
        
        return results
    
    def search_by_year(self, year):
        """Find all videos from a specific year"""
        start_time = time.time()
        video_ids = self.year_index.search(year) or []
        results = [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
        
        search_time = time.time() - start_time
        self.operation_stats['total_searches'] += 1
        self._update_average_search_time(search_time)
        
        return results
    
    def search_by_year_range(self, start_year, end_year):
        """Find all videos within a year range (optimized)"""
        start_time = time.time()
        results = []
        
        for year in range(start_year, end_year + 1):
            video_ids = self.year_index.search(year) or []
            year_results = [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
            results.extend(year_results)
        
        search_time = time.time() - start_time
        self.operation_stats['total_searches'] += 1
        self._update_average_search_time(search_time)
        
        return results
    
    def get_all_videos(self):
        """Get all videos in the system"""
        return self.videos.get_all_values()
    
    def get_video_count(self):
        """Get total number of videos"""
        return self.videos.count
    
    def optimize_performance(self):
        """Perform optimization operations"""
        # Clear caches if they're getting too large
        if self.cache_enabled:
            total_cache_size = (
                len(self.videos.cache) + len(self.actor_index.cache) + 
                len(self.genre_index.cache) + len(self.director_index.cache) + 
                len(self.keyword_index.cache) + len(self.year_index.cache)
            )
            
            if total_cache_size > 1000:  # Arbitrary threshold
                print("Optimizing performance: clearing caches")
                self.videos.clear_cache()
                self.actor_index.clear_cache()
                self.genre_index.clear_cache()
                self.director_index.clear_cache()
                self.keyword_index.clear_cache()
                self.year_index.clear_cache()
    
    def _update_average_add_time(self, add_time):
        """Update average add time statistics"""
        current_avg = self.operation_stats['average_add_time']
        total_adds = self.operation_stats['total_videos_added']
        
        if total_adds > 0:
            self.operation_stats['average_add_time'] = (
                (current_avg * (total_adds - 1) + add_time) / total_adds
            )
    
    def _update_average_search_time(self, search_time):
        """Update average search time statistics"""
        current_avg = self.operation_stats['average_search_time']
        total_searches = self.operation_stats['total_searches']
        
        if total_searches > 0:
            self.operation_stats['average_search_time'] = (
                (current_avg * (total_searches - 1) + search_time) / total_searches
            )
    
    def get_storage_stats(self):
        """Get comprehensive storage statistics"""
        return {
            'videos': self.videos.get_stats(),
            'actor_index': self.actor_index.get_stats(),
            'genre_index': self.genre_index.get_stats(),
            'director_index': self.director_index.get_stats(),
            'keyword_index': self.keyword_index.get_stats(),
            'year_index': self.year_index.get_stats(),
            'operation_stats': self.operation_stats
        }
