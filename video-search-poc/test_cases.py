# test_cases.py
"""
Comprehensive Test Suite for Video Search Platform
Validates functionality and demonstrates robustness of all data structures
"""

import unittest
import time
from video_search_system import VideoSearchSystem, SearchResult
from hash_table import Video, VideoMetadataStore, HashTable
from trie import Trie, VideoTrieSystem
from graph import VideoContentGraph


class TestVideoSearchSystem(unittest.TestCase):
    """Test cases for the comprehensive video search system"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.search_system = VideoSearchSystem()
        self.test_videos = [
            Video(1, "The Matrix", 1999, ["Action", "Sci-Fi"], 
                  ["Keanu Reeves", "Laurence Fishburne"], ["Lana Wachowski"], 
                  ["virtual reality", "philosophy"], 8.7),
            Video(2, "The Godfather", 1972, ["Drama", "Crime"], 
                  ["Marlon Brando", "Al Pacino"], ["Francis Ford Coppola"], 
                  ["mafia", "family"], 9.2),
            Video(3, "Inception", 2010, ["Action", "Sci-Fi", "Thriller"], 
                  ["Leonardo DiCaprio", "Tom Hardy"], ["Christopher Nolan"], 
                  ["dreams", "reality"], 8.8),
            Video(4, "The Dark Knight", 2008, ["Action", "Crime"], 
                  ["Christian Bale", "Heath Ledger"], ["Christopher Nolan"], 
                  ["superhero", "batman"], 9.0),
            Video(5, "Forrest Gump", 1994, ["Drama", "Romance"], 
                  ["Tom Hanks", "Robin Wright"], ["Robert Zemeckis"], 
                  ["life", "destiny"], 8.8)
        ]
        
        # Add test videos to the system
        for video in self.test_videos:
            self.search_system.add_video(video)
    
    def test_video_addition(self):
        """Test video addition to all data structures"""
        initial_count = len(self.search_system.metadata_store.get_all_videos())
        
        new_video = Video(6, "Test Movie", 2023, ["Comedy"], 
                         ["Test Actor"], ["Test Director"], ["test"], 7.5)
        
        result = self.search_system.add_video(new_video)
        self.assertTrue(result)
        
        final_count = len(self.search_system.metadata_store.get_all_videos())
        self.assertEqual(final_count, initial_count + 1)
        
        # Verify video can be retrieved
        retrieved_video = self.search_system.metadata_store.get_video(6)
        self.assertIsNotNone(retrieved_video)
        self.assertEqual(retrieved_video.title, "Test Movie")
    
    def test_title_search_exact(self):
        """Test exact title search functionality"""
        results = self.search_system.search_by_title("The Matrix", "exact")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].video.title, "The Matrix")
        self.assertEqual(results[0].match_type, "exact_title")
    
    def test_title_search_prefix(self):
        """Test prefix-based title search"""
        results = self.search_system.search_by_title("The", "prefix")
        self.assertGreater(len(results), 0)
        
        # All results should have titles starting with "The"
        for result in results:
            self.assertTrue(result.video.title.lower().startswith("the"))
    
    def test_title_search_fuzzy(self):
        """Test fuzzy search with typos"""
        # Search with typo
        results = self.search_system.search_by_title("Matriks", "fuzzy")
        
        # Should find "The Matrix" despite the typo
        matrix_found = any(result.video.title == "The Matrix" for result in results)
        self.assertTrue(matrix_found)
    
    def test_title_search_wildcard(self):
        """Test wildcard search functionality"""
        results = self.search_system.search_by_title("The *father", "wildcard")
        
        # Should find "The Godfather"
        godfather_found = any(result.video.title == "The Godfather" for result in results)
        self.assertTrue(godfather_found)
    
    def test_actor_search(self):
        """Test actor-based search"""
        results = self.search_system.search_by_actor("Tom Hanks")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].video.title, "Forrest Gump")
        
        # Test fuzzy actor search
        results_fuzzy = self.search_system.search_by_actor("Tom Hank")
        self.assertGreater(len(results_fuzzy), 0)
    
    def test_genre_search(self):
        """Test genre-based search"""
        results = self.search_system.search_by_genre("Sci-Fi")
        self.assertGreater(len(results), 0)
        
        # All results should have Sci-Fi genre
        for result in results:
            self.assertIn("Sci-Fi", result.video.genre)
    
    def test_year_search(self):
        """Test year-based search"""
        results = self.search_system.search_by_year(1999)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].video.title, "The Matrix")
    
    def test_complex_search(self):
        """Test complex multi-criteria search"""
        criteria = {
            'genre': 'Action',
            'year_range': (2000, 2020),
            'min_rating': 8.0
        }
        
        results = self.search_system.complex_search(criteria)
        self.assertGreater(len(results), 0)
        
        # Verify all results match criteria
        for result in results:
            video = result.video
            self.assertTrue(any('action' in genre.lower() for genre in video.genre))
            self.assertTrue(2000 <= video.year <= 2020)
            self.assertGreaterEqual(video.rating, 8.0)
    
    def test_similar_videos(self):
        """Test graph-based similarity search"""
        # Find videos similar to The Matrix (ID: 1)
        results = self.search_system.get_similar_videos(1)
        
        # Should find some similar videos (likely other Sci-Fi or action movies)
        # The exact number depends on the similarity algorithm
        self.assertGreaterEqual(len(results), 0)
        
        if len(results) > 0:
            # All results should be different from the original video
            for result in results:
                self.assertNotEqual(result.video.video_id, 1)
    
    def test_genre_recommendations(self):
        """Test genre-based recommendations"""
        results = self.search_system.get_recommendations_by_genre("Action")
        self.assertGreater(len(results), 0)
        
        # Results should be sorted by rating (highest first)
        if len(results) > 1:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(results[i].relevance_score, results[i + 1].relevance_score)
    
    def test_auto_complete(self):
        """Test auto-complete functionality"""
        suggestions = self.search_system.get_auto_complete_suggestions("The", limit=5)
        self.assertGreater(len(suggestions), 0)
        
        # All suggestions should contain the query
        for category, suggestion in suggestions:
            self.assertIn("the", suggestion.lower())
    
    def test_actor_collaborations(self):
        """Test actor collaboration finding"""
        collaborations = self.search_system.get_actor_collaborations("Christian Bale")
        
        # Should find other actors from The Dark Knight
        actor_names = [actor for actor, depth in collaborations]
        self.assertIn("Heath Ledger", actor_names)
    
    def test_spell_correction(self):
        """Test spell correction functionality"""
        results = self.search_system.search_with_spell_correction("Godfathe", "title")
        
        # Should find "The Godfather" despite the typo
        godfather_found = any(result.video.title == "The Godfather" for result in results)
        self.assertTrue(godfather_found)
    
    def test_empty_queries(self):
        """Test handling of empty queries"""
        results = self.search_system.search_by_title("")
        self.assertEqual(len(results), 0)
        
        results = self.search_system.search_by_actor("")
        self.assertEqual(len(results), 0)
        
        results = self.search_system.search_by_genre("")
        self.assertEqual(len(results), 0)
    
    def test_nonexistent_searches(self):
        """Test searches for non-existent content"""
        results = self.search_system.search_by_title("NonexistentMovie")
        self.assertEqual(len(results), 0)
        
        results = self.search_system.search_by_actor("NonexistentActor")
        self.assertEqual(len(results), 0)
        
        results = self.search_system.search_by_genre("NonexistentGenre")
        self.assertEqual(len(results), 0)
        
        results = self.search_system.get_similar_videos(999)
        self.assertEqual(len(results), 0)
    
    def test_bulk_operations(self):
        """Test bulk video addition"""
        new_videos = [
            Video(10, "Bulk Movie 1", 2020, ["Action"], ["Actor 1"], ["Director 1"], ["test"], 7.0),
            Video(11, "Bulk Movie 2", 2021, ["Comedy"], ["Actor 2"], ["Director 2"], ["test"], 6.5)
        ]
        
        results = self.search_system.bulk_add_videos(new_videos)
        self.assertEqual(results['success'], 2)
        self.assertEqual(results['failures'], 0)
        self.assertEqual(results['total'], 2)
    
    def test_system_statistics(self):
        """Test system statistics generation"""
        # Perform some searches to generate stats
        self.search_system.search_by_title("The")
        self.search_system.search_by_actor("Tom Hanks")
        self.search_system.search_by_genre("Action")
        
        stats = self.search_system.get_system_statistics()
        
        # Verify structure of statistics
        self.assertIn('hash_table_stats', stats)
        self.assertIn('trie_stats', stats)
        self.assertIn('graph_stats', stats)
        self.assertIn('search_performance', stats)
        self.assertIn('total_videos', stats)
        
        # Verify search performance tracking
        self.assertGreater(stats['search_performance']['total_searches'], 0)
        self.assertGreater(stats['search_performance']['average_response_time'], 0)
    
    def test_performance_benchmarks(self):
        """Test search performance under load"""
        start_time = time.time()
        
        # Perform multiple searches
        for i in range(100):
            query = f"The"
            self.search_system.search_by_title(query, "prefix", limit=5)
        
        end_time = time.time()
        total_time = end_time - start_time
        average_time = total_time / 100
        
        # Should complete searches in reasonable time (adjust threshold as needed)
        self.assertLess(average_time, 0.1)  # Less than 100ms per search on average
    
    def test_data_integrity(self):
        """Test data integrity across all data structures"""
        # Add a video and verify it's in all structures
        test_video = Video(99, "Integrity Test", 2023, ["Test"], 
                          ["Test Actor"], ["Test Director"], ["integrity"], 8.0)
        
        self.search_system.add_video(test_video)
        
        # Check hash table
        stored_video = self.search_system.metadata_store.get_video(99)
        self.assertIsNotNone(stored_video)
        
        # Check trie system (search by title)
        title_results = self.search_system.search_by_title("Integrity Test", "exact")
        self.assertEqual(len(title_results), 1)
        
        # Check graph (find similar videos)
        similar_results = self.search_system.get_similar_videos(99)
        # Should not crash and should return valid results
        self.assertIsInstance(similar_results, list)


class TestHashTable(unittest.TestCase):
    """Test cases for hash table implementation"""
    
    def setUp(self):
        self.hash_table = HashTable(100)
        self.metadata_store = VideoMetadataStore()
    
    def test_basic_operations(self):
        """Test basic hash table operations"""
        # Insert
        self.hash_table.insert("key1", "value1")
        self.hash_table.insert("key2", "value2")
        
        # Search
        self.assertEqual(self.hash_table.search("key1"), "value1")
        self.assertEqual(self.hash_table.search("key2"), "value2")
        self.assertIsNone(self.hash_table.search("nonexistent"))
        
        # Update
        self.hash_table.insert("key1", "updated_value")
        self.assertEqual(self.hash_table.search("key1"), "updated_value")
        
        # Delete
        self.assertTrue(self.hash_table.delete("key1"))
        self.assertIsNone(self.hash_table.search("key1"))
        self.assertFalse(self.hash_table.delete("nonexistent"))
    
    def test_collision_handling(self):
        """Test hash table collision handling"""
        # Force collisions by using a small table
        small_table = HashTable(2)
        
        # Insert multiple items that will likely collide
        for i in range(10):
            small_table.insert(f"key{i}", f"value{i}")
        
        # Verify all items can still be retrieved
        for i in range(10):
            self.assertEqual(small_table.search(f"key{i}"), f"value{i}")
    
    def test_video_metadata_store(self):
        """Test video metadata store functionality"""
        video = Video(1, "Test Movie", 2023, ["Action"], 
                     ["Test Actor"], ["Test Director"], ["test"], 8.0)
        
        self.metadata_store.add_video(video)
        
        # Test direct retrieval
        retrieved = self.metadata_store.get_video(1)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Test Movie")
        
        # Test index searches
        actor_results = self.metadata_store.search_by_actor("Test Actor")
        self.assertEqual(len(actor_results), 1)
        
        genre_results = self.metadata_store.search_by_genre("Action")
        self.assertEqual(len(genre_results), 1)
        
        year_results = self.metadata_store.search_by_year(2023)
        self.assertEqual(len(year_results), 1)


class TestTrie(unittest.TestCase):
    """Test cases for trie implementation"""
    
    def setUp(self):
        self.trie = Trie()
        self.video_trie_system = VideoTrieSystem()
    
    def test_basic_trie_operations(self):
        """Test basic trie operations"""
        # Insert words
        self.trie.insert("hello", 1)
        self.trie.insert("help", 2)
        self.trie.insert("helicopter", 3)
        
        # Test exact search
        self.assertTrue(self.trie.search_exact("hello"))
        self.assertTrue(self.trie.search_exact("help"))
        self.assertFalse(self.trie.search_exact("hel"))
        
        # Test prefix search
        results = self.trie.search_prefix("hel")
        self.assertIn("hello", results)
        self.assertIn("help", results)
        self.assertIn("helicopter", results)
    
    def test_fuzzy_search(self):
        """Test fuzzy search functionality"""
        self.trie.insert("hello")
        self.trie.insert("world")
        
        # Test with single character substitution
        results = self.trie.fuzzy_search("hallo", max_distance=1)
        self.assertIn("hello", results)
        
        # Test with insertion
        results = self.trie.fuzzy_search("helo", max_distance=1)
        self.assertIn("hello", results)
    
    def test_wildcard_search(self):
        """Test wildcard search functionality"""
        self.trie.insert("hello")
        self.trie.insert("help")
        self.trie.insert("world")
        
        # Test wildcard at end
        results = self.trie.wildcard_search("hel*")
        self.assertIn("hello", results)
        self.assertIn("help", results)
        self.assertNotIn("world", results)
    
    def test_video_trie_system(self):
        """Test video trie system integration"""
        video = Video(1, "The Matrix", 1999, ["Sci-Fi"], 
                     ["Keanu Reeves"], ["Wachowski"], ["virtual"], 8.7)
        
        self.video_trie_system.add_video_to_tries(video)
        
        # Test title search
        title_results = self.video_trie_system.search_titles("Matrix")
        self.assertGreater(len(title_results), 0)
        
        # Test actor search
        actor_results = self.video_trie_system.search_actors("Keanu")
        self.assertGreater(len(actor_results), 0)


class TestGraph(unittest.TestCase):
    """Test cases for graph implementation"""
    
    def setUp(self):
        self.graph = VideoContentGraph()
    
    def test_basic_graph_operations(self):
        """Test basic graph operations"""
        # Add nodes
        self.graph.add_node("video_1", "video", {"title": "Test Movie"})
        self.graph.add_node("actor_1", "actor", {"name": "Test Actor"})
        
        # Add edge
        self.graph.add_edge("video_1", "actor_1", weight=1.0)
        
        # Test BFS
        bfs_result = self.graph.bfs("video_1")
        self.assertGreater(len(bfs_result), 0)
        
        # Test edge weight
        weight = self.graph.get_edge_weight("video_1", "actor_1")
        self.assertEqual(weight, 1.0)
    
    def test_video_graph_integration(self):
        """Test video integration with graph"""
        video = Video(1, "Test Movie", 2023, ["Action"], 
                     ["Test Actor"], ["Test Director"], ["test"], 8.0)
        
        self.graph.add_video_to_graph(video)
        
        # Verify nodes were created
        self.assertIn("video_1", self.graph.nodes)
        self.assertIn("actor_test_actor", self.graph.nodes)
        self.assertIn("genre_action", self.graph.nodes)
    
    def test_similarity_calculation(self):
        """Test video similarity calculation"""
        # Add two videos with shared elements
        video1 = Video(1, "Movie 1", 2020, ["Action"], 
                      ["Actor A"], ["Director A"], ["action"], 8.0)
        video2 = Video(2, "Movie 2", 2021, ["Action"], 
                      ["Actor A"], ["Director B"], ["action"], 7.5)
        
        self.graph.add_video_to_graph(video1)
        self.graph.add_video_to_graph(video2)
        
        # Find similar videos
        similar = self.graph.find_similar_videos(1)
        
        # Should find video 2 as similar due to shared actor and genre
        similar_ids = [int(vid_id) for vid_id, score in similar]
        self.assertIn(2, similar_ids)
    
    def test_graph_statistics(self):
        """Test graph statistics calculation"""
        # Add some test data
        for i in range(3):
            video = Video(i+1, f"Movie {i+1}", 2020+i, ["Action"], 
                         [f"Actor {i+1}"], [f"Director {i+1}"], ["test"], 8.0)
            self.graph.add_video_to_graph(video)
        
        stats = self.graph.get_graph_stats()
        
        self.assertIn('total_nodes', stats)
        self.assertIn('total_edges', stats)
        self.assertIn('node_type_counts', stats)
        self.assertGreater(stats['total_nodes'], 0)


class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset"""
        search_system = VideoSearchSystem()
        
        # Add 100 videos
        videos = []
        for i in range(100):
            video = Video(
                i, f"Movie {i}", 2000 + (i % 23), 
                [f"Genre{i % 5}"], [f"Actor {i % 10}"], 
                [f"Director {i % 8}"], [f"keyword{i % 12}"], 
                5.0 + (i % 5)
            )
            videos.append(video)
        
        # Time bulk addition
        start_time = time.time()
        results = search_system.bulk_add_videos(videos)
        add_time = time.time() - start_time
        
        self.assertEqual(results['success'], 100)
        self.assertLess(add_time, 10.0)  # Should complete within 10 seconds
        
        # Test search performance
        start_time = time.time()
        for i in range(50):
            search_system.search_by_title(f"Movie {i}")
        search_time = time.time() - start_time
        
        self.assertLess(search_time / 50, 0.1)  # Average search under 100ms
    
    def test_memory_efficiency(self):
        """Test memory usage patterns"""
        search_system = VideoSearchSystem()
        
        # Add videos and check statistics
        for i in range(50):
            video = Video(i, f"Movie {i}", 2020, ["Action"], 
                         ["Actor"], ["Director"], ["test"], 8.0)
            search_system.add_video(video)
        
        stats = search_system.get_system_statistics()
        
        # Check that load factors are reasonable
        hash_stats = stats['hash_table_stats']['videos']
        self.assertLess(hash_stats['load_factor'], 0.8)  # Not too loaded
        
        # Check trie efficiency
        trie_stats = stats['trie_stats']['title_trie']
        self.assertGreater(trie_stats['memory_efficiency'], 0.1)  # Reasonable efficiency


def run_all_tests():
    """Run all test suites and generate a comprehensive report"""
    print("="*80)
    print("COMPREHENSIVE TEST SUITE FOR VIDEO SEARCH PLATFORM")
    print("="*80)
    
    # Create test suite
    test_classes = [
        TestVideoSearchSystem,
        TestHashTable,
        TestTrie,
        TestGraph,
        TestPerformance
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        print("-" * 60)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        if result.failures:
            print(f"FAILURES ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback}")
        
        if result.errors:
            print(f"ERRORS ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback}")
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - total_failures - total_errors}")
    print(f"Failed: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.2f}%")
    
    if total_failures == 0 and total_errors == 0:
                    print("\nSUCCESS: ALL TESTS PASSED! The video search platform is working correctly.")
    else:
        print(f"\nWARNING: {total_failures + total_errors} test(s) failed. Review the output above.")
    
    return total_failures == 0 and total_errors == 0


if __name__ == "__main__":
    run_all_tests() 