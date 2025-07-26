# video_search_system.py
"""
Comprehensive Video Search System
Integrates hash tables, tries, and graphs for advanced video content search
"""

from hash_table import Video, VideoMetadataStore
from trie import VideoTrieSystem
from graph import VideoContentGraph
import time
from typing import List, Dict, Tuple, Optional


class SearchResult:
    """Represents a search result with metadata"""
    def __init__(self, video: Video, relevance_score: float = 0.0, match_type: str = ""):
        self.video = video
        self.relevance_score = relevance_score
        self.match_type = match_type
    
    def __str__(self):
        return f"{self.video.title} ({self.video.year}) - Score: {self.relevance_score:.2f} [{self.match_type}]"


class VideoSearchSystem:
    """
    Main video search system integrating all data structures
    Provides comprehensive search functionality for video content discovery
    """
    
    def __init__(self):
        # Initialize all data structures
        self.metadata_store = VideoMetadataStore()
        self.trie_system = VideoTrieSystem()
        self.content_graph = VideoContentGraph()
        
        # Search performance tracking
        self.search_stats = {
            'total_searches': 0,
            'average_response_time': 0.0,
            'search_types': {}
        }
    
    def add_video(self, video: Video) -> bool:
        """
        Add a video to all data structures
        Returns True if successful, False otherwise
        """
        try:
            # Add to hash table storage
            self.metadata_store.add_video(video)
            
            # Add to trie system for text searching
            self.trie_system.add_video_to_tries(video)
            
            # Add to graph for relationship mapping
            self.content_graph.add_video_to_graph(video)
            
            return True
        except Exception as e:
            print(f"Error adding video {video.video_id}: {e}")
            return False
    
    def search_by_title(self, query: str, search_type: str = 'fuzzy', limit: int = 10) -> List[SearchResult]:
        """Search videos by title using trie-based searching"""
        start_time = time.time()
        results = []
        
        try:
            if search_type == 'exact':
                # Exact title match
                matching_titles = self.trie_system.title_trie.get_video_ids_for_word(query)
                for video_id in matching_titles:
                    video = self.metadata_store.get_video(video_id)
                    if video:
                        results.append(SearchResult(video, 1.0, "exact_title"))
            
            elif search_type == 'prefix':
                # Prefix matching
                matching_titles = self.trie_system.title_trie.search_prefix(query, limit)
                for title in matching_titles:
                    video_ids = self.trie_system.title_trie.get_video_ids_for_word(title)
                    for video_id in video_ids:
                        video = self.metadata_store.get_video(video_id)
                        if video:
                            score = 0.8  # High score for prefix match
                            results.append(SearchResult(video, score, "prefix_title"))
            
            elif search_type == 'fuzzy':
                # Fuzzy matching for typos and variations
                matching_titles = self.trie_system.title_trie.fuzzy_search(query)
                for title in matching_titles:
                    video_ids = self.trie_system.title_trie.get_video_ids_for_word(title)
                    for video_id in video_ids:
                        video = self.metadata_store.get_video(video_id)
                        if video:
                            score = 0.6  # Lower score for fuzzy match
                            results.append(SearchResult(video, score, "fuzzy_title"))
            
            elif search_type == 'wildcard':
                # Wildcard matching
                matching_titles = self.trie_system.title_trie.wildcard_search(query)
                for title in matching_titles:
                    video_ids = self.trie_system.title_trie.get_video_ids_for_word(title)
                    for video_id in video_ids:
                        video = self.metadata_store.get_video(video_id)
                        if video:
                            score = 0.7  # Medium score for wildcard match
                            results.append(SearchResult(video, score, "wildcard_title"))
        
        except Exception as e:
            print(f"Error in title search: {e}")
        
        self._update_search_stats('title', time.time() - start_time)
        return self._sort_and_limit_results(results, limit)
    
    def search_by_actor(self, actor_name: str, limit: int = 10) -> List[SearchResult]:
        """Search videos by actor name"""
        start_time = time.time()
        results = []
        
        try:
            # Use hash table for exact actor search
            videos = self.metadata_store.search_by_actor(actor_name)
            for video in videos:
                results.append(SearchResult(video, 1.0, "exact_actor"))
            
            # Also try fuzzy search for actor names
            fuzzy_actors = self.trie_system.actor_trie.fuzzy_search(actor_name)
            for actor in fuzzy_actors:
                video_ids = self.trie_system.actor_trie.get_video_ids_for_word(actor)
                for video_id in video_ids:
                    video = self.metadata_store.get_video(video_id)
                    if video and not any(r.video.video_id == video_id for r in results):
                        results.append(SearchResult(video, 0.7, "fuzzy_actor"))
        
        except Exception as e:
            print(f"Error in actor search: {e}")
        
        self._update_search_stats('actor', time.time() - start_time)
        return self._sort_and_limit_results(results, limit)
    
    def search_by_genre(self, genre: str, limit: int = 10) -> List[SearchResult]:
        """Search videos by genre"""
        start_time = time.time()
        results = []
        
        try:
            videos = self.metadata_store.search_by_genre(genre)
            for video in videos:
                # Score based on rating and recency
                score = min(1.0, (video.rating / 10.0) * 0.7 + 0.3)
                results.append(SearchResult(video, score, "genre"))
        
        except Exception as e:
            print(f"Error in genre search: {e}")
        
        self._update_search_stats('genre', time.time() - start_time)
        return self._sort_and_limit_results(results, limit)
    
    def search_by_year(self, year: int, limit: int = 10) -> List[SearchResult]:
        """Search videos by release year"""
        start_time = time.time()
        results = []
        
        try:
            videos = self.metadata_store.search_by_year(year)
            for video in videos:
                score = min(1.0, video.rating / 10.0)
                results.append(SearchResult(video, score, "year"))
        
        except Exception as e:
            print(f"Error in year search: {e}")
        
        self._update_search_stats('year', time.time() - start_time)
        return self._sort_and_limit_results(results, limit)
    
    def complex_search(self, criteria: Dict, limit: int = 10) -> List[SearchResult]:
        """
        Perform complex multi-criteria search
        criteria = {
            'title': 'partial title',
            'actor': 'actor name',
            'genre': 'genre name',
            'year_range': (start_year, end_year),
            'min_rating': 7.0
        }
        """
        start_time = time.time()
        all_videos = self.metadata_store.get_all_videos()
        results = []
        
        try:
            for video in all_videos:
                score = 0.0
                match_criteria = []
                
                # Title matching
                if 'title' in criteria:
                    title_query = criteria['title'].lower()
                    if title_query in video.title.lower():
                        score += 0.3
                        match_criteria.append("title")
                
                # Actor matching
                if 'actor' in criteria:
                    actor_query = criteria['actor'].lower()
                    if any(actor_query in actor.lower() for actor in video.actors):
                        score += 0.25
                        match_criteria.append("actor")
                
                # Genre matching
                if 'genre' in criteria:
                    genre_query = criteria['genre'].lower()
                    if any(genre_query in genre.lower() for genre in video.genre):
                        score += 0.2
                        match_criteria.append("genre")
                
                # Year range matching
                if 'year_range' in criteria:
                    start_year, end_year = criteria['year_range']
                    if start_year <= video.year <= end_year:
                        score += 0.15
                        match_criteria.append("year_range")
                
                # Rating filter
                if 'min_rating' in criteria:
                    if video.rating >= criteria['min_rating']:
                        score += 0.1
                        match_criteria.append("rating")
                
                # Only include if ALL specified criteria match
                criteria_count = len(criteria)
                if len(match_criteria) == criteria_count and score > 0:
                    # Boost score based on video rating
                    score += (video.rating / 10.0) * 0.1
                    match_type = f"complex({','.join(match_criteria)})"
                    results.append(SearchResult(video, score, match_type))
        
        except Exception as e:
            print(f"Error in complex search: {e}")
        
        self._update_search_stats('complex', time.time() - start_time)
        return self._sort_and_limit_results(results, limit)
    
    def get_similar_videos(self, video_id: int, limit: int = 10) -> List[SearchResult]:
        """Find videos similar to the given video using graph analysis"""
        start_time = time.time()
        results = []
        
        try:
            similar_video_data = self.content_graph.find_similar_videos(video_id, max_results=limit)
            
            for similar_video_id, similarity_score in similar_video_data:
                video = self.metadata_store.get_video(int(similar_video_id))
                if video:
                    results.append(SearchResult(video, similarity_score, "graph_similarity"))
        
        except Exception as e:
            print(f"Error finding similar videos: {e}")
        
        self._update_search_stats('similarity', time.time() - start_time)
        return self._sort_and_limit_results(results, limit)
    
    def get_recommendations_by_genre(self, genre: str, limit: int = 10) -> List[SearchResult]:
        """Get top-rated video recommendations for a genre using graph analysis"""
        start_time = time.time()
        results = []
        
        try:
            recommendations = self.content_graph.get_genre_recommendations(genre, limit)
            
            for video_id, rating in recommendations:
                video = self.metadata_store.get_video(int(video_id))
                if video:
                    score = rating / 10.0  # Normalize rating to 0-1 scale
                    results.append(SearchResult(video, score, "genre_recommendation"))
        
        except Exception as e:
            print(f"Error getting genre recommendations: {e}")
        
        self._update_search_stats('recommendation', time.time() - start_time)
        return results
    
    def get_auto_complete_suggestions(self, query: str, category: str = 'all', limit: int = 5) -> List[Tuple[str, str]]:
        """Get auto-complete suggestions for search queries"""
        try:
            return self.trie_system.get_auto_complete_suggestions(query, category, limit)
        except Exception as e:
            print(f"Error getting auto-complete suggestions: {e}")
            return []
    
    def get_actor_collaborations(self, actor_name: str) -> List[Tuple[str, int]]:
        """Find actors who have collaborated with the given actor"""
        try:
            return self.content_graph.get_actor_collaborations(actor_name)
        except Exception as e:
            print(f"Error finding actor collaborations: {e}")
            return []
    
    def _sort_and_limit_results(self, results: List[SearchResult], limit: int) -> List[SearchResult]:
        """Sort results by relevance score and limit the number returned"""
        # Remove duplicates based on video ID
        unique_results = {}
        for result in results:
            video_id = result.video.video_id
            if video_id not in unique_results or result.relevance_score > unique_results[video_id].relevance_score:
                unique_results[video_id] = result
        
        # Sort by relevance score (descending) and then by rating
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: (x.relevance_score, x.video.rating),
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def _update_search_stats(self, search_type: str, response_time: float):
        """Update search performance statistics"""
        self.search_stats['total_searches'] += 1
        
        # Update average response time
        total_time = (self.search_stats['average_response_time'] * 
                     (self.search_stats['total_searches'] - 1) + response_time)
        self.search_stats['average_response_time'] = total_time / self.search_stats['total_searches']
        
        # Update search type counts
        if search_type not in self.search_stats['search_types']:
            self.search_stats['search_types'][search_type] = 0
        self.search_stats['search_types'][search_type] += 1
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        return {
            'hash_table_stats': self.metadata_store.get_storage_stats(),
            'trie_stats': self.trie_system.get_system_stats(),
            'graph_stats': self.content_graph.get_graph_stats(),
            'search_performance': self.search_stats,
            'total_videos': len(self.metadata_store.get_all_videos())
        }
    
    def export_data(self) -> Dict:
        """Export all system data for backup or analysis"""
        return {
            'videos': [video.to_dict() for video in self.metadata_store.get_all_videos()],
            'graph_data': self.content_graph.export_graph_data(),
            'system_stats': self.get_system_statistics()
        }
    
    def search_with_spell_correction(self, query: str, search_type: str = 'title', limit: int = 10) -> List[SearchResult]:
        """Search with automatic spell correction using fuzzy matching"""
        start_time = time.time()
        
        if search_type == 'title':
            # Try exact search first
            results = self.search_by_title(query, 'exact', limit)
            
            # If no exact results, try fuzzy search
            if not results:
                results = self.search_by_title(query, 'fuzzy', limit)
                
        elif search_type == 'actor':
            results = self.search_by_actor(query, limit)
            
        else:
            results = []
        
        self._update_search_stats('spell_corrected', time.time() - start_time)
        return results
    
    def bulk_add_videos(self, videos: List[Video]) -> Dict[str, int]:
        """Add multiple videos and return success/failure counts"""
        success_count = 0
        failure_count = 0
        
        for video in videos:
            if self.add_video(video):
                success_count += 1
            else:
                failure_count += 1
        
        return {
            'success': success_count,
            'failures': failure_count,
            'total': len(videos)
        } 