# graph.py
"""
Enhanced Graph implementation for Video Search Platform
Models relationships between videos, actors, genres, and directors
Supports similarity analysis and content recommendations
"""

from collections import deque, defaultdict
import math

class Node:
    """Represents a node in the video content graph"""
    def __init__(self, node_id, node_type, data=None):
        self.node_id = node_id
        self.node_type = node_type  # 'video', 'actor', 'director', 'genre', 'keyword'
        self.data = data or {}
        self.connections = set()
        self.weight_connections = {}  # For weighted edges

    def __str__(self):
        return f"{self.node_type}:{self.node_id}"

    def __repr__(self):
        return self.__str__()


class VideoContentGraph:
    """Enhanced graph for modeling video content relationships"""
    
    def __init__(self):
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

    def add_node(self, node_id, node_type, data=None):
        """Add a node to the graph"""
        if node_id not in self.nodes:
            node = Node(node_id, node_type, data)
            self.nodes[node_id] = node
            self.node_types[node_type].add(node_id)
            self.adjacency_list[node_id] = set()

    def add_edge(self, node1_id, node2_id, weight=1.0):
        """Add an edge between two nodes with optional weight"""
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

    def get_edge_weight(self, node1_id, node2_id):
        """Get weight of edge between two nodes"""
        edge_key = tuple(sorted([node1_id, node2_id]))
        return self.weighted_edges.get(edge_key, 0.0)

    def add_video_to_graph(self, video):
        """Add a video and all its relationships to the graph"""
        video_id = f"video_{video.video_id}"
        
        # Add video node
        self.add_node(video_id, 'video', {
            'title': video.title,
            'year': video.year,
            'rating': video.rating,
            'description': video.description
        })
        
        # Add and connect actors
        for actor in video.actors:
            actor_id = f"actor_{actor.lower().replace(' ', '_')}"
            self.add_node(actor_id, 'actor', {'name': actor})
            self.add_edge(video_id, actor_id, weight=1.0)
        
        # Add and connect directors
        for director in video.directors:
            director_id = f"director_{director.lower().replace(' ', '_')}"
            self.add_node(director_id, 'director', {'name': director})
            self.add_edge(video_id, director_id, weight=1.5)  # Higher weight for directors
        
        # Add and connect genres
        for genre in video.genre:
            genre_id = f"genre_{genre.lower().replace(' ', '_')}"
            self.add_node(genre_id, 'genre', {'name': genre})
            self.add_edge(video_id, genre_id, weight=1.2)
        
        # Add and connect keywords
        for keyword in video.keywords:
            keyword_id = f"keyword_{keyword.lower().replace(' ', '_')}"
            self.add_node(keyword_id, 'keyword', {'name': keyword})
            self.add_edge(video_id, keyword_id, weight=0.8)

    def bfs(self, start_node, max_depth=3):
        """Breadth-first search with depth limitation"""
        if start_node not in self.nodes:
            return []
        
        visited = set()
        queue = deque([(start_node, 0)])
        result = []
        
        while queue:
            node, depth = queue.popleft()
            
            if node not in visited and depth <= max_depth:
                visited.add(node)
                result.append((node, depth))
                
                # Add neighbors to queue
                for neighbor in self.adjacency_list[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        
        return result

    def dfs(self, start_node, max_depth=3):
        """Depth-first search with depth limitation"""
        if start_node not in self.nodes:
            return []
        
        visited = set()
        result = []
        
        def _dfs_helper(node, depth):
            if node in visited or depth > max_depth:
                return
            
            visited.add(node)
            result.append((node, depth))
            
            for neighbor in self.adjacency_list[node]:
                _dfs_helper(neighbor, depth + 1)
        
        _dfs_helper(start_node, 0)
        return result

    def find_similar_videos(self, video_id, similarity_threshold=0.3, max_results=10):
        """Find videos similar to the given video based on shared connections"""
        video_node_id = f"video_{video_id}"
        
        if video_node_id not in self.nodes:
            return []
        
        # Get all connected nodes (actors, directors, genres, keywords)
        connected_nodes = self.adjacency_list[video_node_id]
        
        # Find other videos connected to these nodes
        similar_videos = defaultdict(float)
        
        for connected_node in connected_nodes:
            # Get weight of connection
            connection_weight = self.get_edge_weight(video_node_id, connected_node)
            
            # Find other videos connected to this node
            for neighbor in self.adjacency_list[connected_node]:
                if neighbor.startswith('video_') and neighbor != video_node_id:
                    # Calculate similarity score based on shared connections
                    neighbor_weight = self.get_edge_weight(connected_node, neighbor)
                    similarity_score = connection_weight * neighbor_weight
                    
                    # Boost score based on node type
                    node_type = self.nodes[connected_node].node_type
                    if node_type == 'director':
                        similarity_score *= 1.5
                    elif node_type == 'genre':
                        similarity_score *= 1.3
                    elif node_type == 'actor':
                        similarity_score *= 1.1
                    
                    similar_videos[neighbor] += similarity_score
        
        # Filter by threshold and sort by similarity score
        filtered_videos = [
            (video, score) for video, score in similar_videos.items() 
            if score >= similarity_threshold
        ]
        
        filtered_videos.sort(key=lambda x: x[1], reverse=True)
        
        # Extract video IDs and return
        result = []
        for video_node, score in filtered_videos[:max_results]:
            video_id_only = video_node.replace('video_', '')
            result.append((video_id_only, score))
        
        return result

    def get_actor_collaborations(self, actor_name, max_depth=2):
        """Find actors who have worked with the given actor"""
        actor_id = f"actor_{actor_name.lower().replace(' ', '_')}"
        
        if actor_id not in self.nodes:
            return []
        
        # Use BFS to find connected actors through movies
        visited = set()
        queue = deque([(actor_id, 0)])
        collaborators = []
        
        while queue:
            node, depth = queue.popleft()
            
            if node in visited or depth > max_depth:
                continue
            
            visited.add(node)
            
            # If it's an actor (not the original) and not visited
            if node.startswith('actor_') and node != actor_id and depth > 0:
                actor_data = self.nodes[node].data
                collaborators.append((actor_data.get('name', node), depth))
            
            # Add neighbors to queue
            for neighbor in self.adjacency_list[node]:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return collaborators

    def get_genre_recommendations(self, genre_name, limit=10):
        """Get video recommendations based on genre"""
        genre_id = f"genre_{genre_name.lower().replace(' ', '_')}"
        
        if genre_id not in self.nodes:
            return []
        
        # Get all videos in this genre
        genre_videos = []
        for neighbor in self.adjacency_list[genre_id]:
            if neighbor.startswith('video_'):
                video_data = self.nodes[neighbor].data
                video_id = neighbor.replace('video_', '')
                genre_videos.append((video_id, video_data.get('rating', 0.0)))
        
        # Sort by rating and return top recommendations
        genre_videos.sort(key=lambda x: x[1], reverse=True)
        return genre_videos[:limit]

    def find_shortest_path(self, start_node, end_node):
        """Find shortest path between two nodes using BFS"""
        if start_node not in self.nodes or end_node not in self.nodes:
            return []
        
        if start_node == end_node:
            return [start_node]
        
        visited = set()
        queue = deque([(start_node, [start_node])])
        
        while queue:
            node, path = queue.popleft()
            
            if node in visited:
                continue
            
            visited.add(node)
            
            for neighbor in self.adjacency_list[node]:
                if neighbor == end_node:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found

    def get_node_centrality(self, node_id):
        """Calculate centrality measures for a node"""
        if node_id not in self.nodes:
            return {}
        
        # Degree centrality
        degree = len(self.adjacency_list[node_id])
        total_nodes = len(self.nodes) - 1
        degree_centrality = degree / total_nodes if total_nodes > 0 else 0
        
        # Betweenness centrality (simplified)
        # This is computationally expensive, so we'll do a simplified version
        betweenness = 0
        sample_nodes = list(self.nodes.keys())[:min(50, len(self.nodes))]
        
        for i, start in enumerate(sample_nodes):
            for end in sample_nodes[i+1:]:
                if start != node_id and end != node_id:
                    path = self.find_shortest_path(start, end)
                    if node_id in path:
                        betweenness += 1
        
        return {
            'degree_centrality': degree_centrality,
            'degree': degree,
            'betweenness_approximation': betweenness
        }

    def get_graph_stats(self):
        """Get comprehensive graph statistics"""
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
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'node_type_counts': type_counts,
            'average_degree': avg_degree,
            'max_degree': max_degree,
            'graph_density': density,
            'connected_components': self._count_connected_components()
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
                'data': node.data
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
