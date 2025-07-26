# video_search_system.py
"""
Comprehensive Video Search System - Phase 3 Optimizations
Integrates hash tables, tries, and graphs for advanced video content search
Includes performance optimizations, memory management, and scalability improvements
"""

from hash_table import Video, VideoMetadataStore
from trie import VideoTrieSystem
from graph import VideoContentGraph
import time
import gc
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class SearchResult:
    """Represents a search result with metadata and enhanced scoring"""
    def __init__(self, video: Video, relevance_score: float = 0.0, match_type: str = "", 
                 match_details: Dict = None):
        self.video = video
        self.relevance_score = relevance_score
        self.match_type = match_type
        self.match_details = match_details or {}
        self.timestamp = time.time()
    
    def __str__(self):
        return f"{self.video.title} ({self.video.year}) - Score: {self.relevance_score:.2f} [{self.match_type}]"
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'video': self.video.to_dict(),
            'relevance_score': self.relevance_score,
            'match_type': self.match_type,
            'match_details': self.match_details,
            'timestamp': self.timestamp
        }


class PerformanceTracker:
    """Advanced performance tracking for the search system"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.start_time = time.time()
        
    def record_operation(self, operation_type: str, duration: float, **kwargs):
        """Record performance metrics for an operation"""
        self.metrics[f"{operation_type}_times"].append(duration)
        self.counters[f"{operation_type}_count"] += 1
        
        # Record additional context
        for key, value in kwargs.items():
            self.metrics[f"{operation_type}_{key}"].append(value)
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        summary = {
            'uptime_seconds': time.time() - self.start_time,
            'operation_counts': dict(self.counters),
            'average_times': {},
            'total_operations': sum(self.counters.values())
        }
        
        # Calculate averages
        for operation in ['search', 'add_video', 'similarity', 'recommendation']:
            times_key = f"{operation}_times"
            if times_key in self.metrics and self.metrics[times_key]:
                summary['average_times'][operation] = sum(self.metrics[times_key]) / len(self.metrics[times_key])
        
        return summary


class OptimizedVideoSearchSystem:
    """
    Enhanced video search system with Phase 3 optimizations
    Provides comprehensive search functionality with performance improvements
    """
    
    def __init__(self, enable_optimizations=True, cache_size=1000):
        # Initialize optimized data structures
        self.metadata_store = VideoMetadataStore(cache_enabled=enable_optimizations)
        self.trie_system = VideoTrieSystem(cache_size=cache_size)
        self.content_graph = VideoContentGraph()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.enable_optimizations = enable_optimizations
        
        # Search statistics with enhanced tracking
        self.search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'average_response_time': 0.0,
            'search_types': defaultdict(int),
            'result_counts': defaultdict(list),
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Memory management
        self.memory_management = {
            'auto_optimize_threshold': 1000,  # Operations before auto-optimization
            'operation_count': 0,
            'last_optimization': time.time()
        }
        
        # Advanced search features
        self.search_history = []
        self.popular_queries = defaultdict(int)
        
    def add_video(self, video: Video) -> bool:
        """
        Add a video to all data structures with performance tracking
        Returns True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            # Add to hash table storage
            self.metadata_store.add_video(video)
            
            # Add to trie system for text searching
            self.trie_system.add_video_to_tries(video)
            
            # Add to graph for relationship mapping
            self.content_graph.add_video_to_graph(video)
            
            # Track performance
            duration = time.time() - start_time
            self.performance_tracker.record_operation('add_video', duration, video_id=video.video_id)
            
            # Update memory management counters
            self._update_operation_count()
            
            return True
            
        except Exception as e:
            print(f"Error adding video {video.video_id}: {e}")
            return False
    
    def search_by_title(self, query: str, search_type: str = 'fuzzy', limit: int = 10) -> List[SearchResult]:
        """Enhanced title search with improved scoring and caching"""
        start_time = time.time()
        results = []
        
        try:
            # Track query popularity
            self.popular_queries[f"title:{query}"] += 1
            
            # Enhanced search with multiple approaches
            if search_type == 'exact':
                matching_titles = self.trie_system.title_trie.get_video_ids_for_word(query)
                for video_id in matching_titles:
                    video = self.metadata_store.get_video(video_id)
                    if video:
                        results.append(SearchResult(
                            video, 1.0, "exact_title",
                            {'query': query, 'match_position': 'exact'}
                        ))
            
            elif search_type == 'prefix':
                matching_titles = self.trie_system.title_trie.search_prefix(query, limit * 2)
                for title in matching_titles:
                    video_ids = self.trie_system.title_trie.get_video_ids_for_word(title)
                    for video_id in video_ids:
                        video = self.metadata_store.get_video(video_id)
                        if video:
                            # Score based on match position and popularity
                            score = self._calculate_prefix_score(query, title, video)
                            results.append(SearchResult(
                                video, score, "prefix_title",
                                {'query': query, 'matched_title': title}
                            ))
            
            elif search_type == 'fuzzy':
                matching_titles = self.trie_system.title_trie.fuzzy_search(query)
                for title in matching_titles:
                    video_ids = self.trie_system.title_trie.get_video_ids_for_word(title)
                    for video_id in video_ids:
                        video = self.metadata_store.get_video(video_id)
                        if video:
                            # Score based on edit distance and video rating
                            score = self._calculate_fuzzy_score(query, title, video)
                            results.append(SearchResult(
                                video, score, "fuzzy_title",
                                {'query': query, 'matched_title': title}
                            ))
            
            elif search_type == 'wildcard':
                matching_titles = self.trie_system.title_trie.wildcard_search(query)
                for title in matching_titles:
                    video_ids = self.trie_system.title_trie.get_video_ids_for_word(title)
                    for video_id in video_ids:
                        video = self.metadata_store.get_video(video_id)
                        if video:
                            score = 0.7 + (video.rating / 10.0) * 0.2
                            results.append(SearchResult(
                                video, score, "wildcard_title",
                                {'query': query, 'matched_title': title}
                            ))
        
        except Exception as e:
            print(f"Error in title search: {e}")
        
        # Track performance and statistics
        duration = time.time() - start_time
        self._update_search_stats('title', duration, len(results), query)
        
        return self._sort_and_limit_results(results, limit)
    
    def search_by_actor(self, actor_name: str, limit: int = 10) -> List[SearchResult]:
        """Enhanced actor search with improved performance"""
        start_time = time.time()
        results = []
        
        try:
            self.popular_queries[f"actor:{actor_name}"] += 1
            
            # Use hash table for exact actor search
            videos = self.metadata_store.search_by_actor(actor_name)
            for video in videos:
                score = 1.0 + (video.rating / 10.0) * 0.2  # Boost high-rated movies
                results.append(SearchResult(
                    video, score, "exact_actor",
                    {'query': actor_name, 'actor_match': actor_name}
                ))
            
            # Also try fuzzy search for actor names
            fuzzy_actors = self.trie_system.actor_trie.fuzzy_search(actor_name)
            for actor in fuzzy_actors:
                video_ids = self.trie_system.actor_trie.get_video_ids_for_word(actor)
                for video_id in video_ids:
                    video = self.metadata_store.get_video(video_id)
                    if video and not any(r.video.video_id == video_id for r in results):
                        score = 0.7 + (video.rating / 10.0) * 0.2
                        results.append(SearchResult(
                            video, score, "fuzzy_actor",
                            {'query': actor_name, 'matched_actor': actor}
                        ))
        
        except Exception as e:
            print(f"Error in actor search: {e}")
        
        duration = time.time() - start_time
        self._update_search_stats('actor', duration, len(results), actor_name)
        
        return self._sort_and_limit_results(results, limit)
    
    def search_by_genre(self, genre: str, limit: int = 10) -> List[SearchResult]:
        """Enhanced genre search with intelligent ranking"""
        start_time = time.time()
        results = []
        
        try:
            self.popular_queries[f"genre:{genre}"] += 1
            
            videos = self.metadata_store.search_by_genre(genre)
            for video in videos:
                # Enhanced scoring based on multiple factors
                base_score = video.rating / 10.0
                recency_bonus = self._calculate_recency_bonus(video.year)
                popularity_bonus = self._calculate_popularity_bonus(video)
                
                total_score = base_score * 0.6 + recency_bonus * 0.2 + popularity_bonus * 0.2
                
                results.append(SearchResult(
                    video, total_score, "genre",
                    {'query': genre, 'genre_match': genre, 'factors': {
                        'rating': base_score, 'recency': recency_bonus, 'popularity': popularity_bonus
                    }}
                ))
        
        except Exception as e:
            print(f"Error in genre search: {e}")
        
        duration = time.time() - start_time
        self._update_search_stats('genre', duration, len(results), genre)
        
        return self._sort_and_limit_results(results, limit)
    
    def complex_search(self, criteria: Dict, limit: int = 10) -> List[SearchResult]:
        """
        Enhanced multi-criteria search with optimized filtering
        """
        start_time = time.time()
        
        # Use optimized approach: start with most selective criteria
        candidate_videos = self._get_candidate_videos_optimized(criteria)
        results = []
        
        try:
            for video in candidate_videos:
                score, match_details = self._evaluate_complex_criteria(video, criteria)
                
                if score > 0:  # Only include matches
                    results.append(SearchResult(
                        video, score, f"complex({len(criteria)}_criteria)",
                        {'criteria': criteria, 'match_details': match_details}
                    ))
        
        except Exception as e:
            print(f"Error in complex search: {e}")
        
        duration = time.time() - start_time
        self._update_search_stats('complex', duration, len(results), str(criteria))
        
        return self._sort_and_limit_results(results, limit)
    
    def get_similar_videos(self, video_id: int, limit: int = 10) -> List[SearchResult]:
        """Enhanced similarity search with improved algorithms"""
        start_time = time.time()
        results = []
        
        try:
            similar_video_data = self.content_graph.find_similar_videos(
                video_id, max_results=limit * 2  # Get more candidates for better ranking
            )
            
            reference_video = self.metadata_store.get_video(video_id)
            
            for similar_video_id, similarity_score in similar_video_data:
                video = self.metadata_store.get_video(int(similar_video_id))
                if video:
                    # Enhanced similarity scoring
                    enhanced_score = self._enhance_similarity_score(
                        reference_video, video, similarity_score
                    )
                    
                    results.append(SearchResult(
                        video, enhanced_score, "graph_similarity",
                        {'reference_video_id': video_id, 'base_similarity': similarity_score}
                    ))
        
        except Exception as e:
            print(f"Error finding similar videos: {e}")
        
        duration = time.time() - start_time
        self._update_search_stats('similarity', duration, len(results), str(video_id))
        
        return self._sort_and_limit_results(results, limit)
    
    def get_personalized_recommendations(self, user_preferences: Dict, limit: int = 10) -> List[SearchResult]:
        """Generate personalized recommendations based on user preferences"""
        start_time = time.time()
        results = []
        
        try:
            # Combine different recommendation strategies
            genre_recs = self._get_genre_based_recommendations(user_preferences, limit // 2)
            popularity_recs = self._get_popularity_based_recommendations(limit // 2)
            similarity_recs = self._get_similarity_based_recommendations(user_preferences, limit // 2)
            
            # Merge and re-rank
            all_recommendations = genre_recs + popularity_recs + similarity_recs
            
            # Remove duplicates and re-score
            seen_videos = set()
            for rec in all_recommendations:
                if rec.video.video_id not in seen_videos:
                    seen_videos.add(rec.video.video_id)
                    
                    # Apply personalization boost
                    personalized_score = self._apply_personalization_boost(rec, user_preferences)
                    rec.relevance_score = personalized_score
                    rec.match_type = "personalized_recommendation"
                    rec.match_details['personalization'] = user_preferences
                    
                    results.append(rec)
        
        except Exception as e:
            print(f"Error generating recommendations: {e}")
        
        duration = time.time() - start_time
        self._update_search_stats('recommendation', duration, len(results), str(user_preferences))
        
        return self._sort_and_limit_results(results, limit)
    
    def get_trending_content(self, time_window_hours: int = 24, limit: int = 10) -> List[SearchResult]:
        """Get trending content based on search patterns"""
        start_time = time.time()
        results = []
        
        try:
            # Analyze search patterns to determine trending content
            trending_queries = self._analyze_trending_queries(time_window_hours)
            
            for query, frequency in trending_queries[:limit]:
                # Extract content from trending queries
                if query.startswith('title:'):
                    title_results = self.search_by_title(query[6:], 'fuzzy', 3)
                    for result in title_results:
                        result.relevance_score *= (1 + frequency / 100)  # Boost based on popularity
                        result.match_type = "trending_title"
                        result.match_details['trend_frequency'] = frequency
                        results.append(result)
                        
                elif query.startswith('genre:'):
                    genre_results = self.search_by_genre(query[6:], 3)
                    for result in genre_results:
                        result.relevance_score *= (1 + frequency / 100)
                        result.match_type = "trending_genre"
                        result.match_details['trend_frequency'] = frequency
                        results.append(result)
        
        except Exception as e:
            print(f"Error getting trending content: {e}")
        
        duration = time.time() - start_time
        self._update_search_stats('trending', duration, len(results), str(time_window_hours))
        
        return self._sort_and_limit_results(results, limit)
    
    def optimize_system_performance(self):
        """Perform comprehensive system optimization"""
        print("Performing system-wide optimization...")
        start_time = time.time()
        
        try:
            # Optimize metadata store
            self.metadata_store.optimize_performance()
            
            # Clear trie caches
            self.trie_system.clear_all_caches()
            
            # Optimize graph performance
            self.content_graph.optimize_performance()
            
            # Force garbage collection
            gc.collect()
            
            # Update optimization timestamp
            self.memory_management['last_optimization'] = time.time()
            self.memory_management['operation_count'] = 0
            
            optimization_time = time.time() - start_time
            print(f"System optimization completed in {optimization_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error during system optimization: {e}")
    
    def _calculate_prefix_score(self, query: str, title: str, video: Video) -> float:
        """Calculate score for prefix matches"""
        base_score = 0.8
        
        # Boost for exact prefix match
        if title.lower().startswith(query.lower()):
            base_score += 0.1
        
        # Add rating bonus
        rating_bonus = (video.rating / 10.0) * 0.2
        
        # Add recency bonus
        recency_bonus = self._calculate_recency_bonus(video.year) * 0.1
        
        return min(1.0, base_score + rating_bonus + recency_bonus)
    
    def _calculate_fuzzy_score(self, query: str, title: str, video: Video) -> float:
        """Calculate score for fuzzy matches"""
        # Simple edit distance approximation
        max_len = max(len(query), len(title))
        if max_len == 0:
            return 0.0
        
        # Count matching characters
        matching_chars = sum(1 for a, b in zip(query.lower(), title.lower()) if a == b)
        similarity = matching_chars / max_len
        
        base_score = 0.4 + similarity * 0.3
        rating_bonus = (video.rating / 10.0) * 0.2
        
        return min(1.0, base_score + rating_bonus)
    
    def _calculate_recency_bonus(self, year: int) -> float:
        """Calculate recency bonus (newer content gets higher score)"""
        current_year = 2024
        years_old = current_year - year
        
        if years_old <= 2:
            return 0.3
        elif years_old <= 5:
            return 0.2
        elif years_old <= 10:
            return 0.1
        else:
            return 0.0
    
    def _calculate_popularity_bonus(self, video: Video) -> float:
        """Calculate popularity bonus based on various factors"""
        # Use rating as proxy for popularity
        if video.rating >= 8.5:
            return 0.3
        elif video.rating >= 7.5:
            return 0.2
        elif video.rating >= 6.5:
            return 0.1
        else:
            return 0.0
    
    def _get_candidate_videos_optimized(self, criteria: Dict) -> List[Video]:
        """Get candidate videos using the most selective criteria first"""
        candidates = None
        
        # Start with most selective criteria
        if 'year_range' in criteria:
            start_year, end_year = criteria['year_range']
            candidates = self.metadata_store.search_by_year_range(start_year, end_year)
        
        if 'genre' in criteria:
            genre_videos = self.metadata_store.search_by_genre(criteria['genre'])
            if candidates is None:
                candidates = genre_videos
            else:
                # Intersection
                candidate_ids = {v.video_id for v in candidates}
                candidates = [v for v in genre_videos if v.video_id in candidate_ids]
        
        if 'actor' in criteria:
            actor_videos = self.metadata_store.search_by_actor(criteria['actor'])
            if candidates is None:
                candidates = actor_videos
            else:
                candidate_ids = {v.video_id for v in candidates}
                candidates = [v for v in actor_videos if v.video_id in candidate_ids]
        
        return candidates or self.metadata_store.get_all_videos()
    
    def _evaluate_complex_criteria(self, video: Video, criteria: Dict) -> Tuple[float, Dict]:
        """Evaluate how well a video matches complex criteria"""
        score = 0.0
        match_details = {}
        total_criteria = len(criteria)
        
        # Title matching
        if 'title' in criteria:
            title_query = criteria['title'].lower()
            if title_query in video.title.lower():
                score += 1.0 / total_criteria
                match_details['title_match'] = True
        
        # Actor matching
        if 'actor' in criteria:
            actor_query = criteria['actor'].lower()
            if any(actor_query in actor.lower() for actor in video.actors):
                score += 1.0 / total_criteria
                match_details['actor_match'] = True
        
        # Genre matching
        if 'genre' in criteria:
            genre_query = criteria['genre'].lower()
            if any(genre_query in genre.lower() for genre in video.genre):
                score += 1.0 / total_criteria
                match_details['genre_match'] = True
        
        # Year range matching
        if 'year_range' in criteria:
            start_year, end_year = criteria['year_range']
            if start_year <= video.year <= end_year:
                score += 1.0 / total_criteria
                match_details['year_match'] = True
        
        # Rating filter
        if 'min_rating' in criteria:
            if video.rating >= criteria['min_rating']:
                score += 1.0 / total_criteria
                match_details['rating_match'] = True
        
        # Only return videos that match ALL criteria
        if len(match_details) == total_criteria:
            # Boost score based on video rating
            score += (video.rating / 10.0) * 0.1
            return score, match_details
        else:
            return 0.0, {}
    
    def _enhance_similarity_score(self, reference_video: Video, candidate_video: Video, base_score: float) -> float:
        """Enhance similarity score with additional factors"""
        enhanced_score = base_score
        
        # Rating compatibility bonus
        rating_diff = abs(reference_video.rating - candidate_video.rating)
        if rating_diff <= 1.0:
            enhanced_score += 0.1
        
        # Genre overlap bonus
        common_genres = set(reference_video.genre) & set(candidate_video.genre)
        if common_genres:
            enhanced_score += len(common_genres) * 0.05
        
        # Year proximity bonus
        year_diff = abs(reference_video.year - candidate_video.year)
        if year_diff <= 5:
            enhanced_score += 0.05
        
        return min(1.0, enhanced_score)
    
    def _update_operation_count(self):
        """Update operation count and trigger auto-optimization if needed"""
        self.memory_management['operation_count'] += 1
        
        if (self.enable_optimizations and 
            self.memory_management['operation_count'] >= self.memory_management['auto_optimize_threshold']):
            self.optimize_system_performance()
    
    def _update_search_stats(self, search_type: str, response_time: float, result_count: int, query: str):
        """Enhanced search statistics tracking"""
        self.search_stats['total_searches'] += 1
        self.search_stats['search_types'][search_type] += 1
        self.search_stats['result_counts'][search_type].append(result_count)
        
        if result_count > 0:
            self.search_stats['successful_searches'] += 1
        
        # Update average response time
        total_time = (self.search_stats['average_response_time'] * 
                     (self.search_stats['total_searches'] - 1) + response_time)
        self.search_stats['average_response_time'] = total_time / self.search_stats['total_searches']
        
        # Record in performance tracker
        self.performance_tracker.record_operation('search', response_time, 
                                                 search_type=search_type, 
                                                 result_count=result_count)
        
        # Add to search history (keep last 100)
        self.search_history.append({
            'timestamp': time.time(),
            'type': search_type,
            'query': query,
            'results': result_count,
            'duration': response_time
        })
        
        if len(self.search_history) > 100:
            self.search_history.pop(0)
    
    def _sort_and_limit_results(self, results: List[SearchResult], limit: int) -> List[SearchResult]:
        """Enhanced result sorting and limiting"""
        # Remove duplicates based on video ID
        unique_results = {}
        for result in results:
            video_id = result.video.video_id
            if video_id not in unique_results or result.relevance_score > unique_results[video_id].relevance_score:
                unique_results[video_id] = result
        
        # Sort by relevance score, then by rating, then by recency
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: (x.relevance_score, x.video.rating, x.video.year),
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics with Phase 3 enhancements"""
        performance_summary = self.performance_tracker.get_performance_summary()
        
        return {
            'hash_table_stats': self.metadata_store.get_storage_stats(),
            'trie_stats': self.trie_system.get_system_stats(),
            'graph_stats': self.content_graph.get_graph_stats(),
            'search_performance': self.search_stats,
            'performance_tracking': performance_summary,
            'total_videos': len(self.metadata_store.get_all_videos()),
            'memory_management': self.memory_management,
            'popular_queries': dict(self.popular_queries),
            'optimization_status': {
                'optimizations_enabled': self.enable_optimizations,
                'last_optimization': self.memory_management['last_optimization'],
                'operations_since_optimization': self.memory_management['operation_count']
            }
        }


# Keep backward compatibility with original class name
class VideoSearchSystem(OptimizedVideoSearchSystem):
    """Backward compatibility alias for OptimizedVideoSearchSystem"""
    
    def __init__(self):
        super().__init__(enable_optimizations=True, cache_size=1000)
    
    # Maintain backward compatibility for methods with changed signatures
    def search_by_title(self, query: str, search_type: str = 'fuzzy', limit: int = 10) -> List[SearchResult]:
        return super().search_by_title(query, search_type, limit)
    
    def search_by_actor(self, actor_name: str, limit: int = 10) -> List[SearchResult]:
        return super().search_by_actor(actor_name, limit)
    
    def search_by_genre(self, genre: str, limit: int = 10) -> List[SearchResult]:
        return super().search_by_genre(genre, limit)
    
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
        
        self._update_search_stats('year', time.time() - start_time, len(results), str(year))
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
        
        self._update_search_stats('recommendation', time.time() - start_time, len(results), genre)
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
        
        self._update_search_stats('spell_corrected', time.time() - start_time, len(results), query)
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
    
    def export_data(self) -> Dict:
        """Export all system data for backup or analysis"""
        return {
            'videos': [video.to_dict() for video in self.metadata_store.get_all_videos()],
            'graph_data': self.content_graph.export_graph_data(),
            'system_stats': self.get_system_statistics(),
            'search_history': self.search_history[-50:],  # Last 50 searches
            'popular_queries': dict(self.popular_queries)
        } 