import json
import os
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response_data = {
            "status": "healthy",
            "service": "app",
            "runtime": "python",
            "message": "SisFall Fall Detection & Hidden Markov Model API",
            "endpoints": ["/api/health", "/api/info"]
        }
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response_data = {
            "status": "success",
            "message": "Inference endpoint active"
        }
        self.wfile.write(json.dumps(response_data).encode('utf-8'))
