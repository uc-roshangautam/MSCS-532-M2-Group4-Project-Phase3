#!/usr/bin/env python3
"""
Simple Web Server for Video Search Platform UI
Uses only Python standard library - no external dependencies
"""

import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import sys
import mimetypes

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_search_system import VideoSearchSystem
from hash_table import Video


class VideoSearchHandler(BaseHTTPRequestHandler):
    """HTTP request handler for video search API and static files"""
    
    def __init__(self, *args, **kwargs):
        # Initialize the search system with sample data
        if not hasattr(VideoSearchHandler, 'search_system'):
            VideoSearchHandler.search_system = self._initialize_search_system()
        super().__init__(*args, **kwargs)
    
    def _initialize_search_system(self):
        """Initialize search system with sample data from JSON file"""
        system = VideoSearchSystem()
        
        # Load sample data from the same JSON file used by demo.py
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'demo_data.json')
        
        try:
            with open(json_path, 'r') as f:
                sample_videos_data = json.load(f)
            
            # Create Video objects from JSON data (same as demo.py)
            sample_videos = []
            for video_data in sample_videos_data:
                video = Video(**video_data)
                sample_videos.append(video)
            
            # Add all videos to the search system
            for video in sample_videos:
                system.add_video(video)
            
            print(f"SUCCESS: Loaded {len(sample_videos)} videos from demo_data.json")
            
        except FileNotFoundError:
            print(f"ERROR: Could not find demo_data.json at {json_path}")
            print("   Creating minimal fallback data...")
            # Fallback to minimal data if JSON file not found
            fallback_video = Video(
                video_id=1,
                title="Sample Movie",
                year=2024,
                genre=["Drama"],
                actors=["Sample Actor"],
                directors=["Sample Director"],
                keywords=["sample"],
                rating=8.0,
                description="Sample description for fallback data."
            )
            system.add_video(fallback_video)
        
        except Exception as e:
            print(f"ERROR: Error loading demo_data.json: {e}")
            print("   Server will start with empty data.")
        
        return system
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.path = '/index.html'
        
        if self.path.startswith('/api/'):
            self._handle_api_request()
        else:
            self._serve_static_file()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path.startswith('/api/'):
            self._handle_api_request()
        else:
            self._send_error(404, "Not Found")
    
    def _handle_api_request(self):
        """Handle API requests for search functionality"""
        try:
            # Extract path without query parameters
            path_only = self.path.split('?')[0]
            
            if path_only == '/api/search':
                self._handle_search()
            elif path_only == '/api/autocomplete':
                self._handle_autocomplete()
            elif path_only == '/api/similar':
                self._handle_similar()
            else:
                self._send_error(404, "API endpoint not found")
        except Exception as e:
            self._send_json_response({'error': str(e)}, 500)
    
    def _handle_search(self):
        """Handle search requests"""
        query_params = self._get_query_params()
        query = query_params.get('q', [''])[0]
        search_type = query_params.get('type', ['auto'])[0]
        limit = int(query_params.get('limit', ['10'])[0])
        
        if not query:
            self._send_json_response({'results': [], 'message': 'Empty query'})
            return
        
        results = []
        
        # Auto-detect search type or use specified type
        if search_type == 'auto':
            # Try different search types and combine results
            title_results = self.search_system.search_by_title(query, 'fuzzy', limit=5)
            actor_results = self.search_system.search_by_actor(query, limit=3)
            genre_results = self.search_system.search_by_genre(query, limit=3)
            
            # Combine and deduplicate results
            all_results = title_results + actor_results + genre_results
            seen_ids = set()
            for result in all_results:
                if result.video.video_id not in seen_ids:
                    results.append(result)
                    seen_ids.add(result.video.video_id)
        
        elif search_type == 'title':
            results = self.search_system.search_by_title(query, 'fuzzy', limit)
        elif search_type == 'actor':
            results = self.search_system.search_by_actor(query, limit)
        elif search_type == 'genre':
            results = self.search_system.search_by_genre(query, limit)
        elif search_type == 'year':
            try:
                year = int(query)
                results = self.search_system.search_by_year(year, limit)
            except ValueError:
                pass
        
        # Convert results to JSON-serializable format
        json_results = []
        for result in results[:limit]:
            json_results.append({
                'id': result.video.video_id,
                'title': result.video.title,
                'year': result.video.year,
                'genre': result.video.genre,
                'actors': result.video.actors,
                'directors': result.video.directors,
                'rating': result.video.rating,
                'description': result.video.description,
                'score': result.relevance_score,
                'match_type': result.match_type
            })
        
        self._send_json_response({
            'results': json_results,
            'count': len(json_results),
            'query': query,
            'search_type': search_type
        })
    
    def _handle_autocomplete(self):
        """Handle autocomplete requests"""
        query_params = self._get_query_params()
        query = query_params.get('q', [''])[0]
        limit = int(query_params.get('limit', ['5'])[0])
        
        if len(query) < 2:
            self._send_json_response({'suggestions': []})
            return
        
        suggestions = self.search_system.get_auto_complete_suggestions(query, 'all', limit)
        
        json_suggestions = []
        for category, suggestion in suggestions:
            json_suggestions.append({
                'text': suggestion,
                'category': category,
                'display': f"{suggestion} ({category})"
            })
        
        self._send_json_response({'suggestions': json_suggestions})
    
    def _handle_similar(self):
        """Handle similar videos requests"""
        query_params = self._get_query_params()
        video_id = query_params.get('id', [''])[0]
        limit = int(query_params.get('limit', ['5'])[0])
        
        if not video_id:
            self._send_json_response({'error': 'Video ID required'}, 400)
            return
        
        try:
            video_id = int(video_id)
            results = self.search_system.get_similar_videos(video_id, limit)
            
            json_results = []
            for result in results:
                json_results.append({
                    'id': result.video.video_id,
                    'title': result.video.title,
                    'year': result.video.year,
                    'genre': result.video.genre,
                    'rating': result.video.rating,
                    'similarity_score': result.relevance_score
                })
            
            self._send_json_response({'similar_videos': json_results})
        except ValueError:
            self._send_json_response({'error': 'Invalid video ID'}, 400)
    
    def _serve_static_file(self):
        """Serve static files (HTML, CSS, JS)"""
        file_path = self.path.lstrip('/')
        if not file_path:
            file_path = 'index.html'
        
        # Security: prevent directory traversal
        if '..' in file_path:
            self._send_error(403, "Forbidden")
            return
        
        # Try to serve the file
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = 'application/octet-stream'
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            
        except FileNotFoundError:
            self._send_error(404, "File not found")
    
    def _get_query_params(self):
        """Parse query parameters from URL"""
        if '?' in self.path:
            query_string = self.path.split('?', 1)[1]
            return urllib.parse.parse_qs(query_string)
        return {}
    
    def _send_json_response(self, data, status_code=200):
        """Send JSON response"""
        json_data = json.dumps(data, indent=2).encode('utf-8')
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(json_data)))
        self.send_header('Access-Control-Allow-Origin', '*')  # Enable CORS for development
        self.end_headers()
        self.wfile.write(json_data)
    
    def _send_error(self, status_code, message):
        """Send error response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(message.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to reduce log verbosity"""
        # Only log errors and API requests
        if 'api' in args[0] or 'ERROR' in format:
            super().log_message(format, *args)


def run_server(port=8000):
    """Run the web server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, VideoSearchHandler)
    
    print(f"Video Search Platform UI running at:")
    print(f"   http://localhost:{port}")
    print(f"   http://127.0.0.1:{port}")
    print("\nFeatures available:")
    print("   • Smart search with auto-detection")
    print("   • Real-time autocomplete suggestions")
    print("   • Fuzzy matching for typos")
    print("   • Similar video recommendations")
    print("   • Multiple search types (title, actor, genre, year)")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped. Thanks for using Video Search Platform!")
        httpd.server_close()


if __name__ == '__main__':
    run_server() 