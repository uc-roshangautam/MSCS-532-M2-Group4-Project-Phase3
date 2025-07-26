# trie.py
"""
Enhanced Trie implementation for Video Search Platform
Supports prefix matching, fuzzy search, auto-complete, and wildcard searching
"""

class TrieNode:
    """Node class for the Trie data structure"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.video_ids = []  # Store video IDs for words that end here
        self.frequency = 0   # Track frequency of searches for this word

class Trie:
    """Enhanced Trie with fuzzy matching and auto-complete capabilities"""
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0

    def insert(self, word, video_id=None):
        """Insert word into trie with optional video ID association"""
        if not word:
            return
        
        word = word.lower().strip()
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end:
            self.word_count += 1
        
        node.is_end = True
        node.frequency += 1
        
        if video_id and video_id not in node.video_ids:
            node.video_ids.append(video_id)

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
        """Find all words with given prefix, limited by count"""
        if not prefix:
            return []
        
        prefix = prefix.lower().strip()
        node = self.root
        
        # Navigate to the prefix
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all words with this prefix
        words = self._collect_words(node, prefix, limit)
        
        # Sort by frequency (most searched first)
        words.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in words[:limit]]

    def _collect_words(self, node, prefix, limit, collected=None):
        """Helper method to collect words from a node"""
        if collected is None:
            collected = []
        
        if len(collected) >= limit:
            return collected
        
        if node.is_end:
            collected.append((prefix, node.frequency))
        
        for char, child_node in node.children.items():
            if len(collected) >= limit:
                break
            self._collect_words(child_node, prefix + char, limit, collected)
        
        return collected

    def auto_complete(self, partial_word, limit=5):
        """Provide auto-complete suggestions"""
        suggestions = self.search_prefix(partial_word, limit)
        return suggestions

    def fuzzy_search(self, word, max_distance=2):
        """Fuzzy search allowing for character insertions, deletions, and substitutions"""
        if not word:
            return []
        
        word = word.lower().strip()
        results = []
        
        def _fuzzy_helper(node, target, current_word, distance):
            if distance > max_distance:
                return
            
            # If we've matched the target word
            if not target and node.is_end:
                results.append((current_word, distance))
                return
            
            # If we've processed all characters in target
            if not target:
                if node.is_end:
                    results.append((current_word, distance))
                return
            
            # Try all possible operations
            for char, child_node in node.children.items():
                # Exact match
                if target and char == target[0]:
                    _fuzzy_helper(child_node, target[1:], current_word + char, distance)
                
                # Substitution
                if target:
                    _fuzzy_helper(child_node, target[1:], current_word + char, distance + 1)
                
                # Insertion
                _fuzzy_helper(child_node, target, current_word + char, distance + 1)
            
            # Deletion
            if target:
                _fuzzy_helper(node, target[1:], current_word, distance + 1)
        
        _fuzzy_helper(self.root, word, "", 0)
        
        # Sort by edit distance, then by frequency
        results.sort(key=lambda x: (x[1], -self._get_frequency(x[0])))
        
        return [word for word, distance in results[:10]]

    def _get_frequency(self, word):
        """Get frequency of a word"""
        node = self.root
        for char in word:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.frequency if node.is_end else 0

    def wildcard_search(self, pattern):
        """Search with wildcard support (* matches any sequence of characters)"""
        if not pattern:
            return []
        
        pattern = pattern.lower().strip()
        results = []
        
        def _wildcard_helper(node, pattern_idx, current_word):
            if pattern_idx >= len(pattern):
                if node.is_end:
                    results.append(current_word)
                return
            
            char = pattern[pattern_idx]
            
            if char == '*':
                # Match zero or more characters
                # Try skipping the wildcard
                _wildcard_helper(node, pattern_idx + 1, current_word)
                
                # Try matching one or more characters
                for child_char, child_node in node.children.items():
                    _wildcard_helper(child_node, pattern_idx, current_word + child_char)
                    _wildcard_helper(child_node, pattern_idx + 1, current_word + child_char)
            else:
                # Exact character match
                if char in node.children:
                    _wildcard_helper(node.children[char], pattern_idx + 1, current_word + char)
        
        _wildcard_helper(self.root, 0, "")
        return list(set(results))  # Remove duplicates

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

    def get_stats(self):
        """Get trie statistics"""
        node_count = self._count_nodes(self.root)
        depth = self._max_depth(self.root)
        
        return {
            'word_count': self.word_count,
            'node_count': node_count,
            'max_depth': depth,
            'memory_efficiency': self.word_count / node_count if node_count > 0 else 0
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


class VideoTrieSystem:
    """Comprehensive trie system for video search with multiple search categories"""
    
    def __init__(self):
        self.title_trie = Trie()      # For video titles
        self.actor_trie = Trie()      # For actor names
        self.genre_trie = Trie()      # For genres
        self.keyword_trie = Trie()    # For plot keywords
        self.director_trie = Trie()   # For directors
    
    def add_video_to_tries(self, video):
        """Add video information to all relevant tries"""
        video_id = video.video_id
        
        # Add title words
        title_words = video.title.lower().split()
        for word in title_words:
            self.title_trie.insert(word, video_id)
        
        # Add full title
        self.title_trie.insert(video.title, video_id)
        
        # Add actors
        for actor in video.actors:
            self.actor_trie.insert(actor, video_id)
            # Also add individual names
            for name_part in actor.split():
                self.actor_trie.insert(name_part, video_id)
        
        # Add genres
        for genre in video.genre:
            self.genre_trie.insert(genre, video_id)
        
        # Add keywords
        for keyword in video.keywords:
            self.keyword_trie.insert(keyword, video_id)
        
        # Add directors
        for director in video.directors:
            self.director_trie.insert(director, video_id)
            # Also add individual names
            for name_part in director.split():
                self.director_trie.insert(name_part, video_id)
    
    def search_titles(self, query, search_type='prefix'):
        """Search video titles"""
        if search_type == 'exact':
            return self.title_trie.get_video_ids_for_word(query)
        elif search_type == 'prefix':
            return self.title_trie.search_prefix(query)
        elif search_type == 'fuzzy':
            return self.title_trie.fuzzy_search(query)
        elif search_type == 'wildcard':
            return self.title_trie.wildcard_search(query)
        else:
            return []
    
    def search_actors(self, query, search_type='prefix'):
        """Search actor names"""
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
    
    def get_auto_complete_suggestions(self, query, category='all', limit=5):
        """Get auto-complete suggestions across categories"""
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
        
        return suggestions[:limit]
    
    def get_system_stats(self):
        """Get comprehensive trie system statistics"""
        return {
            'title_trie': self.title_trie.get_stats(),
            'actor_trie': self.actor_trie.get_stats(),
            'genre_trie': self.genre_trie.get_stats(),
            'keyword_trie': self.keyword_trie.get_stats(),
            'director_trie': self.director_trie.get_stats()
        }
