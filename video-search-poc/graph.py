# graph.py
"""
Enhanced Graph implementation for Video Search Platform - Phase 3 Optimizations
Models relationships between videos, actors, genres, and directors
Supports similarity analysis, content recommendations, and performance optimizations
Includes caching, optimized algorithms, and scalability improvements
"""

from collections import deque, defaultdict
import math
import time
from typing import List, Dict, Tuple, Optional, Set
import heapq


class Node:
    """Represents a node in the video content graph with optimization features"""
    def __init__(self, node_id, node_type, data=None):
        self.node_id = node_id
        self.node_type = node_type  # 'video', 'actor', 'director', 'genre', 'keyword'
        self.data = data or {}
        self.connections = set()
        self.weight_connections = {}  # For weighted edges
        self.centrality_cache = {}   # Cache centrality calculations
        self.last_accessed = time.time()
        
        # Performance tracking
        self.access_count = 0
        self.similarity_cache = {}   # Cache similarity calculations

    def __str__(self):
        return f"{self.node_type}:{self.node_id}"

    def __repr__(self):
        return self.__str__()
    
    def update_access(self):
        """Update access tracking for performance optimization"""
        self.access_count += 1
        self.last_accessed = time.time()


class OptimizedVideoContentGraph:
    """Enhanced graph for modeling video content relationships with performance optimizations"""
    
    def __init__(self, cache_size=5000):
        self.nodes = {}
        self.adjacency_list = defaultdict(set)
        self.weighted_edges = {}  # For storing edge weights
        self.node_types = {
            'video': set(),
            'actor': set(),
            'director': set(),
            'genre': set(),
            'keyword': set()
        }
        
        # Performance optimization features
        self.cache_size = cache_size
        self.similarity_cache = {}      # Cache for similarity calculations
        self.path_cache = {}           # Cache for shortest path calculations
        self.centrality_cache = {}     # Cache for centrality calculations
        self.recommendation_cache = {} # Cache for recommendations
        
        # Cache access tracking for LRU
        self.cache_access_order = {
            'similarity': [],
            'path': [],
            'centrality': [],
            'recommendation': []
        }
        
        # Performance statistics
        self.stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'similarity_calculations': 0,
            'traversal_operations': 0,
            'average_similarity_time': 0.0,
            'average_traversal_time': 0.0
        }
        
        # Optimization parameters
        self.max_traversal_depth = 4
        self.similarity_threshold = 0.1
        self.enable_caching = True

    def add_node(self, node_id, node_type, data=None):
        """Add a node to the graph with performance tracking"""
        if node_id not in self.nodes:
            node = Node(node_id, node_type, data)
            self.nodes[node_id] = node
            self.node_types[node_type].add(node_id)
            self.adjacency_list[node_id] = set()
            
            # Clear related caches when graph structure changes
            self._invalidate_structural_caches()

    def add_edge(self, node1_id, node2_id, weight=1.0):
        """Add an edge between two nodes with optional weight and caching invalidation"""
        if node1_id in self.nodes and node2_id in self.nodes:
            self.adjacency_list[node1_id].add(node2_id)
            self.adjacency_list[node2_id].add(node1_id)
            
            # Store edge weights
            edge_key = tuple(sorted([node1_id, node2_id]))
            self.weighted_edges[edge_key] = weight
            
            # Update node connections
            self.nodes[node1_id].connections.add(node2_id)
            self.nodes[node2_id].connections.add(node1_id)
            self.nodes[node1_id].weight_connections[node2_id] = weight
            self.nodes[node2_id].weight_connections[node1_id] = weight
            
            # Invalidate caches that depend on graph structure
            self._invalidate_structural_caches()

    def get_edge_weight(self, node1_id, node2_id):
        """Get weight of edge between two nodes with caching"""
        edge_key = tuple(sorted([node1_id, node2_id]))
        return self.weighted_edges.get(edge_key, 0.0)

    def add_video_to_graph(self, video):
        """Add a video and all its relationships to the graph with optimizations"""
        video_id = f"video_{video.video_id}"
        
        # Add video node
        self.add_node(video_id, 'video', {
            'title': video.title,
            'year': video.year,
            'rating': video.rating,
            'description': video.description
        })
        
        # Add and connect actors with enhanced weighting
        for actor in video.actors:
            actor_id = f"actor_{actor.lower().replace(' ', '_')}"
            self.add_node(actor_id, 'actor', {'name': actor})
            # Weight based on video rating and recency
            weight = 1.0 + (video.rating / 10.0) * 0.3 + self._calculate_recency_bonus(video.year)
            self.add_edge(video_id, actor_id, weight=weight)
        
        # Add and connect directors with higher weights
        for director in video.directors:
            director_id = f"director_{director.lower().replace(' ', '_')}"
            self.add_node(director_id, 'director', {'name': director})
            weight = 1.5 + (video.rating / 10.0) * 0.4 + self._calculate_recency_bonus(video.year)
            self.add_edge(video_id, director_id, weight=weight)
        
        # Add and connect genres
        for genre in video.genre:
            genre_id = f"genre_{genre.lower().replace(' ', '_')}"
            self.add_node(genre_id, 'genre', {'name': genre})
            weight = 1.2 + (video.rating / 10.0) * 0.2
            self.add_edge(video_id, genre_id, weight=weight)
        
        # Add and connect keywords with variable weights
        for keyword in video.keywords:
            keyword_id = f"keyword_{keyword.lower().replace(' ', '_')}"
            self.add_node(keyword_id, 'keyword', {'name': keyword})
            weight = 0.8 + (video.rating / 10.0) * 0.1
            self.add_edge(video_id, keyword_id, weight=weight)
    
    def _calculate_recency_bonus(self, year):
        """Calculate recency bonus for weighting (more recent gets higher weight)"""
        current_year = 2024  # Could be dynamic
        years_old = current_year - year
        # Decay function: newer movies get slight bonus
        return max(0, 0.2 * math.exp(-years_old / 20))

    def bfs_optimized(self, start_node, max_depth=3, limit=100):
        """Optimized BFS with caching and early termination"""
        start_time = time.time()
        
        cache_key = f"bfs_{start_node}_{max_depth}_{limit}"
        if self.enable_caching and cache_key in self.path_cache:
            self.stats['cache_hits'] += 1
            self._update_cache_access('path', cache_key)
            self._update_traversal_stats(time.time() - start_time)
            return self.path_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        if start_node not in self.nodes:
            return []
        
        visited = set()
        queue = deque([(start_node, 0)])
        result = []
        
        while queue and len(result) < limit:
            node, depth = queue.popleft()
            
            if node not in visited and depth <= max_depth:
                visited.add(node)
                result.append((node, depth))
                self.nodes[node].update_access()
                
                # Add neighbors to queue, prioritizing by edge weight
                neighbors = [(neighbor, self.get_edge_weight(node, neighbor)) 
                           for neighbor in self.adjacency_list[node] 
                           if neighbor not in visited]
                
                # Sort by weight (descending) for better traversal
                neighbors.sort(key=lambda x: x[1], reverse=True)
                
                for neighbor, weight in neighbors:
                    if len(result) < limit:
                        queue.append((neighbor, depth + 1))
        
        # Cache the result
        if self.enable_caching:
            self._update_cache('path', cache_key, result)
        
        self._update_traversal_stats(time.time() - start_time)
        return result

    def dfs_optimized(self, start_node, max_depth=3, limit=100):
        """Optimized DFS with caching and pruning"""
        start_time = time.time()
        
        cache_key = f"dfs_{start_node}_{max_depth}_{limit}"
        if self.enable_caching and cache_key in self.path_cache:
            self.stats['cache_hits'] += 1
            self._update_cache_access('path', cache_key)
            self._update_traversal_stats(time.time() - start_time)
            return self.path_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        if start_node not in self.nodes:
            return []
        
        visited = set()
        result = []
        
        def _dfs_helper(node, depth):
            if node in visited or depth > max_depth or len(result) >= limit:
                return
            
            visited.add(node)
            result.append((node, depth))
            self.nodes[node].update_access()
            
            # Sort neighbors by weight for better exploration
            neighbors = [(neighbor, self.get_edge_weight(node, neighbor)) 
                       for neighbor in self.adjacency_list[node]]
            neighbors.sort(key=lambda x: x[1], reverse=True)
            
            for neighbor, weight in neighbors:
                if len(result) < limit:
                    _dfs_helper(neighbor, depth + 1)
        
        _dfs_helper(start_node, 0)
        
        # Cache the result
        if self.enable_caching:
            self._update_cache('path', cache_key, result)
        
        self._update_traversal_stats(time.time() - start_time)
        return result

    def find_similar_videos(self, video_id, similarity_threshold=None, max_results=10):
        """Optimized similarity search with caching and improved algorithms"""
        start_time = time.time()
        
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        cache_key = f"similarity_{video_id}_{similarity_threshold}_{max_results}"
        if self.enable_caching and cache_key in self.similarity_cache:
            self.stats['cache_hits'] += 1
            self._update_cache_access('similarity', cache_key)
            self._update_similarity_stats(time.time() - start_time)
            return self.similarity_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        video_node_id = f"video_{video_id}"
        
        if video_node_id not in self.nodes:
            return []
        
        # Use optimized similarity calculation
        similar_videos = self._calculate_similarity_scores(video_node_id, similarity_threshold)
        
        # Use heap for efficient top-k selection
        if len(similar_videos) > max_results:
            top_similar = heapq.nlargest(max_results, similar_videos.items(), key=lambda x: x[1])
        else:
            top_similar = list(similar_videos.items())
        
        # Format results
        result = []
        for video_node, score in top_similar:
            if score >= similarity_threshold:
                video_id_only = video_node.replace('video_', '')
                result.append((video_id_only, score))
        
        # Cache the result
        if self.enable_caching:
            self._update_cache('similarity', cache_key, result)
        
        self._update_similarity_stats(time.time() - start_time)
        return result
    
    def _calculate_similarity_scores(self, video_node_id, threshold):
        """Optimized similarity calculation using multiple factors"""
        connected_nodes = self.adjacency_list[video_node_id]
        similar_videos = defaultdict(float)
        
        # Multi-factor similarity calculation
        for connected_node in connected_nodes:
            connection_weight = self.get_edge_weight(video_node_id, connected_node)
            node_type = self.nodes[connected_node].node_type
            
            # Get type-specific weight multiplier
            type_multiplier = self._get_type_importance(node_type)
            
            # Find other videos connected to this node
            for neighbor in self.adjacency_list[connected_node]:
                if neighbor.startswith('video_') and neighbor != video_node_id:
                    neighbor_weight = self.get_edge_weight(connected_node, neighbor)
                    
                    # Calculate similarity score with multiple factors
                    base_score = connection_weight * neighbor_weight * type_multiplier
                    
                    # Add collaborative filtering bonus
                    collaboration_bonus = self._calculate_collaboration_bonus(
                        video_node_id, neighbor, connected_node
                    )
                    
                    # Add rating compatibility bonus
                    rating_bonus = self._calculate_rating_compatibility(video_node_id, neighbor)
                    
                    total_score = base_score + collaboration_bonus + rating_bonus
                    similar_videos[neighbor] += total_score
        
        # Normalize scores and apply threshold
        max_score = max(similar_videos.values()) if similar_videos else 1.0
        normalized_scores = {
            video: score / max_score 
            for video, score in similar_videos.items()
            if score / max_score >= threshold
        }
        
        return normalized_scores
    
    def _get_type_importance(self, node_type):
        """Get importance multiplier for different node types"""
        importance_map = {
            'director': 1.8,
            'genre': 1.4,
            'actor': 1.2,
            'keyword': 0.9
        }
        return importance_map.get(node_type, 1.0)
    
    def _calculate_collaboration_bonus(self, video1, video2, shared_node):
        """Calculate bonus for shared high-importance connections"""
        # Bonus for sharing important people (directors, main actors)
        if self.nodes[shared_node].node_type in ['director', 'actor']:
            return 0.1
        return 0.0
    
    def _calculate_rating_compatibility(self, video1, video2):
        """Calculate bonus for similar ratings"""
        try:
            rating1 = self.nodes[video1].data.get('rating', 0)
            rating2 = self.nodes[video2].data.get('rating', 0)
            rating_diff = abs(rating1 - rating2)
            
            # Closer ratings get higher bonus
            if rating_diff <= 0.5:
                return 0.15
            elif rating_diff <= 1.0:
                return 0.1
            elif rating_diff <= 2.0:
                return 0.05
            return 0.0
        except:
            return 0.0

    def get_actor_collaborations(self, actor_name, max_depth=2, limit=20):
        """Optimized actor collaboration search with caching"""
        start_time = time.time()
        
        cache_key = f"collab_{actor_name}_{max_depth}_{limit}"
        if self.enable_caching and cache_key in self.path_cache:
            self.stats['cache_hits'] += 1
            self._update_cache_access('path', cache_key)
            self._update_traversal_stats(time.time() - start_time)
            return self.path_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        actor_id = f"actor_{actor_name.lower().replace(' ', '_')}"
        
        if actor_id not in self.nodes:
            return []
        
        # Use BFS to find collaborations with weight consideration
        visited = set()
        queue = deque([(actor_id, 0, [])])
        collaborators = []
        
        while queue and len(collaborators) < limit:
            node, depth, path = queue.popleft()
            
            if node in visited or depth > max_depth:
                continue
            
            visited.add(node)
            
            # If it's an actor (not the original) and not already found
            if (node.startswith('actor_') and node != actor_id and 
                depth > 0 and len(path) > 0):
                actor_data = self.nodes[node].data
                collaborator_name = actor_data.get('name', node)
                # Calculate collaboration strength based on shared movies/projects
                strength = self._calculate_collaboration_strength(actor_id, node, path)
                collaborators.append((collaborator_name, depth, strength))
            
            # Add neighbors to queue, prioritizing high-weight connections
            neighbors = [(neighbor, self.get_edge_weight(node, neighbor)) 
                        for neighbor in self.adjacency_list[node] 
                        if neighbor not in visited]
            
            neighbors.sort(key=lambda x: x[1], reverse=True)
            
            for neighbor, weight in neighbors:
                if len(collaborators) < limit:
                    new_path = path + [node] if depth > 0 else [node]
                    queue.append((neighbor, depth + 1, new_path))
        
        # Sort by collaboration strength
        collaborators.sort(key=lambda x: x[2], reverse=True)
        result = [(name, depth) for name, depth, strength in collaborators]
        
        # Cache the result
        if self.enable_caching:
            self._update_cache('path', cache_key, result)
        
        self._update_traversal_stats(time.time() - start_time)
        return result
    
    def _calculate_collaboration_strength(self, actor1_id, actor2_id, path):
        """Calculate strength of collaboration between two actors"""
        # Count shared movies/projects
        shared_projects = 0
        for intermediate_node in path:
            if intermediate_node.startswith('video_'):
                shared_projects += 1
        
        # More shared projects = stronger collaboration
        return shared_projects + 1

    def get_genre_recommendations(self, genre_name, limit=10):
        """Optimized genre recommendations with caching and intelligent ranking"""
        start_time = time.time()
        
        cache_key = f"genre_rec_{genre_name}_{limit}"
        if self.enable_caching and cache_key in self.recommendation_cache:
            self.stats['cache_hits'] += 1
            self._update_cache_access('recommendation', cache_key)
            return self.recommendation_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        genre_id = f"genre_{genre_name.lower().replace(' ', '_')}"
        
        if genre_id not in self.nodes:
            return []
        
        # Get all videos in this genre with enhanced scoring
        genre_videos = []
        for neighbor in self.adjacency_list[genre_id]:
            if neighbor.startswith('video_'):
                video_data = self.nodes[neighbor].data
                video_id = neighbor.replace('video_', '')
                
                # Calculate recommendation score
                base_rating = video_data.get('rating', 0.0)
                edge_weight = self.get_edge_weight(genre_id, neighbor)
                recency_bonus = self._calculate_recency_bonus(video_data.get('year', 0))
                popularity_bonus = self._calculate_popularity_bonus(neighbor)
                
                total_score = base_rating + edge_weight + recency_bonus + popularity_bonus
                genre_videos.append((video_id, total_score))
        
        # Sort by total score and return top recommendations
        genre_videos.sort(key=lambda x: x[1], reverse=True)
        result = genre_videos[:limit]
        
        # Cache the result
        if self.enable_caching:
            self._update_cache('recommendation', cache_key, result)
        
        return result
    
    def _calculate_popularity_bonus(self, video_node_id):
        """Calculate popularity bonus based on number of connections"""
        connection_count = len(self.adjacency_list[video_node_id])
        # Videos with more connections (actors, keywords, etc.) get slight bonus
        return min(0.5, connection_count * 0.02)

    def find_shortest_path_optimized(self, start_node, end_node):
        """Optimized shortest path with caching and bidirectional search"""
        cache_key = f"path_{start_node}_{end_node}"
        if self.enable_caching and cache_key in self.path_cache:
            self.stats['cache_hits'] += 1
            self._update_cache_access('path', cache_key)
            return self.path_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        if start_node not in self.nodes or end_node not in self.nodes:
            return []
        
        if start_node == end_node:
            return [start_node]
        
        # Use bidirectional BFS for better performance
        result = self._bidirectional_bfs(start_node, end_node)
        
        # Cache the result
        if self.enable_caching:
            self._update_cache('path', cache_key, result)
        
        return result
    
    def _bidirectional_bfs(self, start, end):
        """Bidirectional BFS for faster pathfinding"""
        if start == end:
            return [start]
        
        # Forward and backward searches
        forward_visited = {start: [start]}
        backward_visited = {end: [end]}
        forward_queue = deque([start])
        backward_queue = deque([end])
        
        while forward_queue or backward_queue:
            # Forward search
            if forward_queue:
                current = forward_queue.popleft()
                for neighbor in self.adjacency_list[current]:
                    if neighbor in backward_visited:
                        # Found connection
                        forward_path = forward_visited[current]
                        backward_path = backward_visited[neighbor]
                        return forward_path + backward_path[::-1]
                    
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = forward_visited[current] + [neighbor]
                        forward_queue.append(neighbor)
            
            # Backward search
            if backward_queue:
                current = backward_queue.popleft()
                for neighbor in self.adjacency_list[current]:
                    if neighbor in forward_visited:
                        # Found connection
                        forward_path = forward_visited[neighbor]
                        backward_path = backward_visited[current]
                        return forward_path + backward_path[::-1]
                    
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = [neighbor] + backward_visited[current]
                        backward_queue.append(neighbor)
        
        return []  # No path found

    def get_node_centrality_optimized(self, node_id):
        """Optimized centrality calculation with caching"""
        if node_id not in self.nodes:
            return {}
        
        if self.enable_caching and node_id in self.centrality_cache:
            self.stats['cache_hits'] += 1
            return self.centrality_cache[node_id]
        
        self.stats['cache_misses'] += 1
        
        # Degree centrality
        degree = len(self.adjacency_list[node_id])
        total_nodes = len(self.nodes) - 1
        degree_centrality = degree / total_nodes if total_nodes > 0 else 0
        
        # Weighted degree centrality
        total_weight = sum(self.get_edge_weight(node_id, neighbor) 
                          for neighbor in self.adjacency_list[node_id])
        weighted_degree_centrality = total_weight / max(total_nodes, 1)
        
        # Simplified betweenness centrality (sampling for performance)
        betweenness = self._calculate_sampled_betweenness(node_id)
        
        # Closeness centrality (limited sampling)
        closeness = self._calculate_sampled_closeness(node_id)
        
        result = {
            'degree_centrality': degree_centrality,
            'weighted_degree_centrality': weighted_degree_centrality,
            'degree': degree,
            'betweenness_approximation': betweenness,
            'closeness_approximation': closeness
        }
        
        # Cache the result
        if self.enable_caching:
            self._update_cache('centrality', node_id, result)
        
        return result
    
    def _calculate_sampled_betweenness(self, node_id, sample_size=50):
        """Calculate betweenness centrality using sampling for performance"""
        if len(self.nodes) < 3:
            return 0.0
        
        betweenness = 0
        sample_nodes = list(self.nodes.keys())[:min(sample_size, len(self.nodes))]
        
        for i, start in enumerate(sample_nodes):
            for end in sample_nodes[i+1:]:
                if start != node_id and end != node_id:
                    path = self.find_shortest_path_optimized(start, end)
                    if node_id in path:
                        betweenness += 1
        
        return betweenness / max(1, len(sample_nodes) * (len(sample_nodes) - 1) / 2)
    
    def _calculate_sampled_closeness(self, node_id, sample_size=30):
        """Calculate closeness centrality using sampling"""
        if len(self.nodes) < 2:
            return 0.0
        
        total_distance = 0
        reachable_nodes = 0
        sample_nodes = list(self.nodes.keys())[:min(sample_size, len(self.nodes))]
        
        for target in sample_nodes:
            if target != node_id:
                path = self.find_shortest_path_optimized(node_id, target)
                if path:
                    total_distance += len(path) - 1
                    reachable_nodes += 1
        
        if reachable_nodes == 0:
            return 0.0
        
        return reachable_nodes / total_distance

    def _update_cache(self, cache_type, key, value):
        """Update cache with LRU eviction"""
        cache = getattr(self, f'{cache_type}_cache')
        access_order = self.cache_access_order[cache_type]
        
        if key in cache:
            access_order.remove(key)
        elif len(cache) >= self.cache_size // 4:  # Divide cache among four types
            if access_order:
                lru_key = access_order.pop(0)
                del cache[lru_key]
        
        cache[key] = value
        access_order.append(key)
    
    def _update_cache_access(self, cache_type, key):
        """Update cache access order for LRU"""
        access_order = self.cache_access_order[cache_type]
        if key in access_order:
            access_order.remove(key)
            access_order.append(key)
    
    def _invalidate_structural_caches(self):
        """Clear caches that depend on graph structure"""
        self.path_cache.clear()
        self.centrality_cache.clear()
        # Keep similarity and recommendation caches as they're more expensive to rebuild
        
        for cache_type in ['path', 'centrality']:
            self.cache_access_order[cache_type].clear()
    
    def _update_similarity_stats(self, duration):
        """Update similarity calculation statistics"""
        self.stats['similarity_calculations'] += 1
        current_avg = self.stats['average_similarity_time']
        count = self.stats['similarity_calculations']
        
        self.stats['average_similarity_time'] = (
            (current_avg * (count - 1) + duration) / count
        )
    
    def _update_traversal_stats(self, duration):
        """Update traversal operation statistics"""
        self.stats['traversal_operations'] += 1
        current_avg = self.stats['average_traversal_time']
        count = self.stats['traversal_operations']
        
        self.stats['average_traversal_time'] = (
            (current_avg * (count - 1) + duration) / count
        )

    def optimize_performance(self):
        """Perform performance optimization operations"""
        print("Optimizing graph performance...")
        
        # Clear least recently used cache entries
        for cache_type in ['similarity', 'recommendation']:
            cache = getattr(self, f'{cache_type}_cache')
            access_order = self.cache_access_order[cache_type]
            
            if len(cache) > self.cache_size // 8:  # Keep only most recent entries
                keep_count = self.cache_size // 8
                keys_to_remove = access_order[:-keep_count] if len(access_order) > keep_count else []
                
                for key in keys_to_remove:
                    cache.pop(key, None)
                    access_order.remove(key)
        
        print("Graph performance optimization completed")

    def get_graph_stats(self):
        """Get comprehensive graph statistics with performance metrics"""
        total_nodes = len(self.nodes)
        total_edges = sum(len(neighbors) for neighbors in self.adjacency_list.values()) // 2
        
        # Count nodes by type
        type_counts = {node_type: len(nodes) for node_type, nodes in self.node_types.items()}
        
        # Calculate average degree
        degrees = [len(self.adjacency_list[node]) for node in self.nodes]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        # Calculate density
        max_possible_edges = total_nodes * (total_nodes - 1) // 2
        density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Cache statistics
        cache_hit_rate = (
            self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
            if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
        )
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'node_type_counts': type_counts,
            'average_degree': avg_degree,
            'max_degree': max_degree,
            'graph_density': density,
            'connected_components': self._count_connected_components(),
            'cache_hit_rate': cache_hit_rate,
            'performance_stats': {
                'average_similarity_time': self.stats['average_similarity_time'],
                'average_traversal_time': self.stats['average_traversal_time'],
                'total_operations': self.stats['total_operations'],
                'similarity_calculations': self.stats['similarity_calculations'],
                'traversal_operations': self.stats['traversal_operations']
            },
            'cache_sizes': {
                'similarity': len(self.similarity_cache),
                'path': len(self.path_cache),
                'centrality': len(self.centrality_cache),
                'recommendation': len(self.recommendation_cache)
            }
        }

    def _count_connected_components(self):
        """Count the number of connected components in the graph"""
        visited = set()
        components = 0
        
        for node in self.nodes:
            if node not in visited:
                # Start BFS from this node
                queue = deque([node])
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        queue.extend(self.adjacency_list[current])
                components += 1
        
        return components

    def export_graph_data(self):
        """Export graph data for visualization or analysis"""
        nodes_data = []
        edges_data = []
        
        # Export nodes
        for node_id, node in self.nodes.items():
            nodes_data.append({
                'id': node_id,
                'type': node.node_type,
                'data': node.data,
                'access_count': node.access_count,
                'last_accessed': node.last_accessed
            })
        
        # Export edges
        processed_edges = set()
        for node_id, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                edge_key = tuple(sorted([node_id, neighbor]))
                if edge_key not in processed_edges:
                    processed_edges.add(edge_key)
                    edges_data.append({
                        'source': edge_key[0],
                        'target': edge_key[1],
                        'weight': self.weighted_edges.get(edge_key, 1.0)
                    })
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'stats': self.get_graph_stats()
        }


# Keep original class for backward compatibility
class VideoContentGraph(OptimizedVideoContentGraph):
    """Backward compatibility alias for OptimizedVideoContentGraph"""
    
    def __init__(self):
        super().__init__()
    
    # Maintain backward compatibility for existing method names
    def bfs(self, start_node, max_depth=3):
        return self.bfs_optimized(start_node, max_depth)
    
    def dfs(self, start_node, max_depth=3):
        return self.dfs_optimized(start_node, max_depth)
    
    def find_shortest_path(self, start_node, end_node):
        return self.find_shortest_path_optimized(start_node, end_node)
    
    def get_node_centrality(self, node_id):
        return self.get_node_centrality_optimized(node_id)
