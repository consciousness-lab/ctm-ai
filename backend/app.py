from flask import Flask, jsonify, make_response, request, send_from_directory
from werkzeug.utils import secure_filename
from ctm_ai.chunks import Chunk, ChunkManager
from ctm_ai.ctms.ctm import ConsciousnessTuringMachine
import os
import uuid

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024 * 1024  # 1GB
    ALLOWED_EXTENSIONS = {
        'images': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
        'audios': {'mp3', 'wav', 'aac', 'flac'},
        'videos': {'mp4', 'avi', 'mov', 'wmv', 'flv'},
    }
    FRONTEND_TO_BACKEND_PROCESSORS = {
        'VisionProcessor': 'vision_processor',
        'LanguageProcessor': 'language_processor',
        'SearchProcessor': 'search_processor',
        'MathProcessor': 'math_processor',
    }

class AppState:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.query = None
        self.winning_chunk = None
        self.chunks = []
        self.node_details = {}
        self.node_parents = {}
        self.node_gists = {}
        self.saved_files = {'images': [], 'audios': [], 'videos': []}

class FileHandler:
    @staticmethod
    def allowed_file(filename, file_type):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS.get(file_type, set())

    @staticmethod
    def generate_unique_filename(filename):
        ext = filename.rsplit('.', 1)[1].lower()
        return f'{uuid.uuid4().hex}.{ext}'

    @staticmethod
    def save_file(file, file_type, app):
        if file and FileHandler.allowed_file(file.filename, file_type):
            filename = secure_filename(file.filename)
            unique_filename = FileHandler.generate_unique_filename(filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_type, unique_filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)
            return unique_filename
        return None

class ChunkProcessor:
    @staticmethod
    def process_chunks(ctm_instance, query, chunks):
        new_chunks = ctm_instance.ask_processors(query)
        return {chunk.processor_name: chunk for chunk in new_chunks}

    @staticmethod
    def fuse_chunks(ctm_instance, chunks):
        return ctm_instance.fuse_processor(chunks)

    @staticmethod
    def compete_chunks(chunk_manager, chunk1, chunk2):
        return chunk_manager.compete(chunk1, chunk2)

class FlaskAppWrapper:
    def __init__(self):
        self.app = Flask(__name__)
        self.ctm = ConsciousnessTuringMachine()
        self.state = AppState()
        self.chunk_manager = ChunkManager()
        self.setup_app_config()
        self.register_routes()

    def setup_app_config(self):
        self.app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        self.app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

    def add_cors_headers(self, response):
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response

    def handle_options_request(self):
        return self.add_cors_headers(make_response())

    def register_routes(self):
        @self.app.route('/api/refresh', methods=['POST', 'OPTIONS'])
        def handle_refresh():
            if request.method == 'OPTIONS':
                return self.handle_options_request()
            
            self.state.reset()
            return self.add_cors_headers(jsonify({'message': 'Server data refreshed'}))

        @self.app.route('/api/nodes/<node_id>')
        def get_node_details(node_id):
            raw_detail = self.state.node_details.get(node_id, 'No details available')
            node_self = raw_detail.format_readable() if isinstance(raw_detail, Chunk) else str(raw_detail)
            
            parent_data = {}
            if node_id in self.state.node_parents:
                for parent_id in self.state.node_parents[node_id]:
                    raw_parent_detail = self.state.node_details.get(parent_id, 'No details available')
                    parent_data[parent_id] = raw_parent_detail.format_readable() if isinstance(raw_parent_detail, Chunk) else str(raw_parent_detail)
            
            return self.add_cors_headers(jsonify({'self': node_self, 'parents': parent_data}))

        @self.app.route('/api/init', methods=['POST', 'OPTIONS'])
        def initialize_processors():
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json()
            selected_processors = data.get('selected_processors', [])
            
            if not selected_processors:
                return self.add_cors_headers(
                    jsonify({'error': 'No processors provided'})), 400

            self.state.node_details.clear()
            self.state.node_parents.clear()
            self.state.node_gists.clear()
            
            self.ctm.reset()
            created_processor_names = []

            for frontend_label in selected_processors:
                backend_processor_name = Config.FRONTEND_TO_BACKEND_PROCESSORS.get(frontend_label)
                if backend_processor_name:
                    self.ctm.add_processor(processor_name=backend_processor_name)
                    self.state.node_details[backend_processor_name] = backend_processor_name
                    created_processor_names.append(backend_processor_name)

            self.ctm.add_supervisor('language_supervisor')
            self.ctm.add_scorer('language_scorer')
            self.ctm.add_fuser('language_fuser')

            return self.add_cors_headers(
                jsonify({
                    'message': 'Processors initialized',
                    'processorNames': created_processor_names
                })
            )

        @self.app.route('/api/output-gist', methods=['POST', 'OPTIONS'])
        def handle_output_gist():
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json()
            updates = data.get('updates', [])

            gists = ChunkProcessor.process_chunks(self.ctm, self.state.query, self.state.chunks)

            for update in updates:
                proc_id = update.get('processor_id')
                target_id = update.get('target_id')
                self.state.node_details[target_id] = gists[proc_id]
                
                if target_id not in self.state.node_parents:
                    self.state.node_parents[target_id] = [proc_id]
                else:
                    self.state.node_parents[target_id].append(proc_id)

            return self.add_cors_headers(
                jsonify({'message': 'Gist outputs processed', 'updates': updates})
            )

        @self.app.route('/api/uptree', methods=['POST', 'OPTIONS'])
        def handle_uptree():
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json()
            updates = data.get('updates', [])

            for update in updates:
                node_id = update.get('node_id')
                parent_nodes = update.get('parents', [])

                if node_id not in self.state.node_parents:
                    self.state.node_parents[node_id] = parent_nodes
                else:
                    self.state.node_parents[node_id].extend(parent_nodes)

            for node_id, parents_ids in self.state.node_parents.items():
                if node_id not in self.state.node_details and len(parents_ids) >= 2:
                    parent_id1, parent_id2 = parents_ids[0], parents_ids[1]
                    self.state.node_details[node_id] = ChunkProcessor.compete_chunks(
                        self.chunk_manager,
                        self.state.node_details[parent_id1],
                        self.state.node_details[parent_id2]
                    )

            return self.add_cors_headers(
                jsonify({
                    'message': 'Uptree updates processed',
                    'node_parents': self.state.node_parents
                })
            )

        @self.app.route('/api/final-node', methods=['POST', 'OPTIONS'])
        def handle_final_node():
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json()
            node_id = data.get('node_id')
            parents = data.get('parents', [])

            self.state.node_parents[node_id] = parents

            for node_id, parents_ids in self.state.node_parents.items():
                if node_id not in self.state.node_details and parents_ids:
                    parent_id = parents_ids[0]
                    answer, confidence_score = self.ctm.ask_supervisor(
                        self.state.query,
                        self.state.node_details[parent_id]
                    )
                    self.state.node_details[node_id] = (
                        f'Answer: {answer}\n\nConfidence score: {confidence_score}'
                    )
                    self.state.winning_chunk = self.state.node_details[parent_id]

            return self.add_cors_headers(
                jsonify({
                    'message': 'Final node updated',
                    'node_parents': self.state.node_parents
                })
            )

        @self.app.route('/api/reverse', methods=['POST', 'OPTIONS'])
        def handle_reverse():
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            if self.state.winning_chunk:
                self.ctm.downtree_broadcast(self.state.winning_chunk)

            return self.add_cors_headers(jsonify({'message': 'Reverse broadcast processed'}))

        @self.app.route('/api/update-processors', methods=['POST', 'OPTIONS'])
        def update_processors():
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json()
            updates = data.get('updates', [])

            self.ctm.link_form(self.state.chunks)
            self.state.chunks = []
            self.state.node_details.clear()
            self.state.node_parents.clear()
            self.state.node_gists.clear()

            for update in updates:
                proc_id = update.get('processor_id')
                if proc_id in self.state.node_details:
                    self.state.node_details[proc_id] = f'Updated processor {proc_id}'

            return self.add_cors_headers(jsonify({'message': 'Processors updated'}))

        @self.app.route('/api/fuse-gist', methods=['POST', 'OPTIONS'])
        def handle_fuse_gist():
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json()
            updates = data.get('updates', [])

            self.state.chunks = ChunkProcessor.fuse_chunks(self.ctm, self.state.chunks)

            for update in updates:
                fused_node_id = update.get('fused_node_id')
                source_nodes = update.get('source_nodes', [])

                source_chunks = [self.state.node_details[node_id] for node_id in source_nodes]
                fused_chunk = source_chunks[0] if source_chunks else None

                if fused_chunk:
                    self.state.node_details[fused_node_id] = fused_chunk
                    self.state.node_parents[fused_node_id] = source_nodes

            return self.add_cors_headers(
                jsonify({'message': 'Fused gists processed', 'updates': updates})
            )

        @self.app.route('/api/fetch-neighborhood', methods=['GET', 'OPTIONS'])
        def get_processor_neighborhoods():
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            neighborhoods = {}
            graph = self.ctm.processor_graph.graph

            for processor, connected_processors in graph.items():
                neighborhoods[processor.name] = [p.name for p in connected_processors]

            return self.add_cors_headers(jsonify(neighborhoods))

        @self.app.route('/api/upload', methods=['POST', 'OPTIONS'])
        def upload_files():
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            self.state.query = request.form.get('query', '')
            text = request.form.get('text', '')
            
            saved_files = {'images': [], 'audios': [], 'videos': []}
            
            for file_type in ['images', 'audios', 'videos']:
                if file_type in request.files:
                    files = request.files.getlist(file_type)
                    for file in files:
                        unique_filename = FileHandler.save_file(file, file_type, self.app)
                        if unique_filename:
                            saved_files[file_type].append(unique_filename)
                        else:
                            return self.add_cors_headers(
                                jsonify({'error': f'Invalid {file_type} file: {file.filename}'})), 400

            response_data = {
                'message': 'Files uploaded successfully',
                'query': self.state.query,
                'text': text,
                'saved_files': saved_files,
                'download_links': {
                    ftype: [f'/uploads/{ftype}/{fname}' for fname in fnames]
                    for ftype, fnames in saved_files.items()
                }
            }

            return self.add_cors_headers(jsonify(response_data))

        @self.app.route('/uploads/<file_type>/<filename>', methods=['GET'])
        def uploaded_file(file_type, filename):
            if file_type not in ['images', 'audios', 'videos']:
                return self.add_cors_headers(
                    jsonify({'error': 'Invalid file type'})), 400

            try:
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], file_type)
                response = send_from_directory(file_path, filename)
                return self.add_cors_headers(response)
            except FileNotFoundError:
                return self.add_cors_headers(
                    jsonify({'error': 'File not found'})), 404

    def run(self, **kwargs):
        # Create upload directories
        for file_type in ['images', 'audios', 'videos']:
            os.makedirs(
                os.path.join(self.app.config['UPLOAD_FOLDER'], file_type),
                exist_ok=True
            )
        self.app.run(**kwargs)

if __name__ == '__main__':
    app = FlaskAppWrapper()
    app.run(port=5000, debug=True)