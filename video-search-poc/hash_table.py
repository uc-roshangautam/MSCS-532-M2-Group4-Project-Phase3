# hash_table.py
"""
Enhanced Hash Table implementation for Video Search Platform
Supports video metadata storage, actor indexing, genre indexing, and keyword indexing
"""

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


class HashTable:
    """Enhanced Hash Table with collision handling using chaining"""
    def __init__(self, size=1000):
        self.size = size
        self.table = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key):
        """Improved hash function for better distribution"""
        if isinstance(key, str):
            hash_value = 0
            for char in key:
                hash_value = (hash_value * 31 + ord(char)) % self.size
            return hash_value
        return hash(key) % self.size

    def insert(self, key, value):
        """Insert key-value pair with update capability"""
        index = self._hash(key)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))
        self.count += 1

    def search(self, key):
        """Search for value by key"""
        index = self._hash(key)
        for k, v in self.table[index]:
            if k == key:
                return v
        return None

    def delete(self, key):
        """Delete key-value pair"""
        index = self._hash(key)
        for i, (k, _) in enumerate(self.table[index]):
            if k == key:
                del self.table[index][i]
                self.count -= 1
                return True
        return False

    def get_all_values(self):
        """Get all values stored in the hash table"""
        values = []
        for bucket in self.table:
            for _, value in bucket:
                values.append(value)
        return values

    def get_stats(self):
        """Get hash table statistics"""
        used_buckets = sum(1 for bucket in self.table if bucket)
        max_bucket_size = max(len(bucket) for bucket in self.table)
        avg_bucket_size = self.count / used_buckets if used_buckets > 0 else 0
        
        return {
            'total_items': self.count,
            'table_size': self.size,
            'used_buckets': used_buckets,
            'load_factor': self.count / self.size,
            'max_bucket_size': max_bucket_size,
            'avg_bucket_size': avg_bucket_size
        }


class VideoMetadataStore:
    """Comprehensive video metadata storage system using multiple hash tables"""
    
    def __init__(self):
        # Primary storage for video objects
        self.videos = HashTable(1000)
        
        # Index tables for different search criteria
        self.actor_index = HashTable(500)      # actor_name -> [video_ids]
        self.genre_index = HashTable(100)      # genre -> [video_ids]
        self.director_index = HashTable(300)   # director_name -> [video_ids]
        self.keyword_index = HashTable(800)    # keyword -> [video_ids]
        self.year_index = HashTable(200)       # year -> [video_ids]
        
    def add_video(self, video):
        """Add a video and update all relevant indexes"""
        if not isinstance(video, Video):
            raise TypeError("Expected Video object")
        
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
    
    def get_video(self, video_id):
        """Retrieve video by ID"""
        return self.videos.search(video_id)
    
    def search_by_actor(self, actor_name):
        """Find all videos with a specific actor"""
        video_ids = self.actor_index.search(actor_name.lower()) or []
        return [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
    
    def search_by_genre(self, genre):
        """Find all videos of a specific genre"""
        video_ids = self.genre_index.search(genre.lower()) or []
        return [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
    
    def search_by_director(self, director_name):
        """Find all videos by a specific director"""
        video_ids = self.director_index.search(director_name.lower()) or []
        return [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
    
    def search_by_keyword(self, keyword):
        """Find all videos with a specific keyword"""
        video_ids = self.keyword_index.search(keyword.lower()) or []
        return [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
    
    def search_by_year(self, year):
        """Find all videos from a specific year"""
        video_ids = self.year_index.search(year) or []
        return [self.videos.search(vid) for vid in video_ids if self.videos.search(vid)]
    
    def get_all_videos(self):
        """Get all videos in the system"""
        return self.videos.get_all_values()
    
    def get_storage_stats(self):
        """Get comprehensive storage statistics"""
        return {
            'videos': self.videos.get_stats(),
            'actor_index': self.actor_index.get_stats(),
            'genre_index': self.genre_index.get_stats(),
            'director_index': self.director_index.get_stats(),
            'keyword_index': self.keyword_index.get_stats(),
            'year_index': self.year_index.get_stats()
        }
