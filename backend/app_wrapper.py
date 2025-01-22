import os
from typing import Any, Dict, List, Tuple, Union

from chunk_processor import ChunkProcessor
from config import Config
from file_handler import FileHandler
from flask import Flask, jsonify, make_response, request, send_from_directory
from flask.wrappers import Response as FlaskResponse
from state import AppState
from werkzeug.wrappers.response import Response as WerkzeugResponse

from ctm_ai.chunks import Chunk, ChunkManager
from ctm_ai.ctms.ctm import ConsciousnessTuringMachine
from ctm_ai.utils import extract_audio_from_video, extract_video_frames

ResponseType = Union[FlaskResponse, WerkzeugResponse]


class FlaskAppWrapper:
    def __init__(self) -> None:
        self.app: Flask = Flask(__name__)
        self.ctm: ConsciousnessTuringMachine = ConsciousnessTuringMachine()
        self.state: AppState = AppState()
        self.chunk_manager: ChunkManager = ChunkManager()
        self.setup_app_config()
        self.register_routes()

    def setup_app_config(self) -> None:
        self.app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        self.app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

    def add_cors_headers(self, response: ResponseType) -> ResponseType:
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    def handle_options_request(self) -> ResponseType:
        return self.add_cors_headers(make_response())

    def register_routes(self) -> None:
        @self.app.route('/api/refresh', methods=['POST', 'OPTIONS'])
        def handle_refresh() -> ResponseType:
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            self.state.reset()
            return self.add_cors_headers(jsonify({'message': 'Server data refreshed'}))

        @self.app.route('/api/nodes/<node_id>')
        def get_node_details(node_id: str) -> ResponseType:
            raw_detail = self.state.node_details.get(node_id, 'No details available')
            node_self = (
                raw_detail.format_readable()
                if isinstance(raw_detail, Chunk)
                else str(raw_detail)
            )

            parent_data: Dict[str, str] = {}
            if node_id in self.state.node_parents:
                for parent_id in self.state.node_parents[node_id]:
                    raw_parent_detail = self.state.node_details.get(
                        parent_id, 'No details available'
                    )
                    parent_data[parent_id] = (
                        raw_parent_detail.format_readable()
                        if isinstance(raw_parent_detail, Chunk)
                        else str(raw_parent_detail)
                    )

            return self.add_cors_headers(
                jsonify({'self': node_self, 'parents': parent_data})
            )

        @self.app.route('/api/init', methods=['POST', 'OPTIONS'])
        def initialize_processors() -> Tuple[ResponseType, int]:
            if request.method == 'OPTIONS':
                return self.handle_options_request(), 200

            data = request.get_json() or {}
            selected_processors: List[str] = data.get('selected_processors', [])

            if not selected_processors:
                return self.add_cors_headers(
                    jsonify({'error': 'No processors provided'})
                ), 400

            self.state.node_details.clear()
            self.state.node_parents.clear()
            self.state.node_gists.clear()

            self.ctm.reset()
            created_processor_names: List[str] = []

            for frontend_label in selected_processors:
                backend_processor_name = Config.FRONTEND_TO_BACKEND_PROCESSORS.get(
                    frontend_label
                )
                if backend_processor_name:
                    self.ctm.add_processor(processor_name=backend_processor_name)
                    self.state.node_details[backend_processor_name] = (
                        backend_processor_name
                    )
                    created_processor_names.append(backend_processor_name)

            self.ctm.add_supervisor('language_supervisor')
            self.ctm.add_scorer('language_scorer')
            self.ctm.add_fuser('language_fuser')

            return self.add_cors_headers(
                jsonify(
                    {
                        'message': 'Processors initialized',
                        'processorNames': created_processor_names,
                    }
                )
            ), 200

        @self.app.route('/api/output-gist', methods=['POST', 'OPTIONS'])
        def handle_output_gist() -> ResponseType:
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json() or {}
            updates: List[Dict[str, str]] = data.get('updates', [])

            image_filename = (
                self.state.saved_files['images'][0]
                if self.state.saved_files['images']
                else None
            )
            image_absolute_path = (
                os.path.join(self.app.config['UPLOAD_FOLDER'], 'images', image_filename)
                if image_filename
                else None
            )

            audio_filename = (
                self.state.saved_files['audios'][0]
                if self.state.saved_files['audios']
                else None
            )
            audio_absolute_path = (
                os.path.join(self.app.config['UPLOAD_FOLDER'], 'audios', audio_filename)
                if audio_filename
                else None
            )
            video_filename = (
                self.state.saved_files['videos'][0]
                if self.state.saved_files['videos']
                else None
            )
            video_absolute_path = (
                os.path.join(self.app.config['UPLOAD_FOLDER'], 'videos', video_filename)
                if video_filename
                else None
            )

            video_frames_path = (
                [
                    os.path.join(
                        self.app.config['UPLOAD_FOLDER'], 'video_frames', frame_filename
                    )
                    for frame_filename in self.state.saved_files['video_frames']
                ]
                if self.state.saved_files['video_frames']
                else []
            )

            gists = ChunkProcessor.process_chunks(
                ctm_instance=self.ctm,
                query=self.state.query,
                image_path=image_absolute_path,
                audio_path=audio_absolute_path,
                video_frames_path=video_frames_path,
                video_path=video_absolute_path,
            )

            for update in updates:
                proc_id = update.get('processor_id', '')
                target_id = update.get('target_id', '')
                if proc_id and target_id and proc_id in gists:
                    self.state.node_details[target_id] = gists[proc_id]

                    if target_id not in self.state.node_parents:
                        self.state.node_parents[target_id] = [proc_id]
                    else:
                        self.state.node_parents[target_id].append(proc_id)

            return self.add_cors_headers(
                jsonify({'message': 'Gist outputs processed', 'updates': updates})
            )

        @self.app.route('/api/uptree', methods=['POST', 'OPTIONS'])
        def handle_uptree() -> ResponseType:
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json() or {}
            updates: List[Dict[str, Any]] = data.get('updates', [])

            for update in updates:
                node_id = update.get('node_id', '')
                parent_nodes: List[str] = update.get('parents', [])

                if node_id:
                    if node_id not in self.state.node_parents:
                        self.state.node_parents[node_id] = parent_nodes
                    else:
                        self.state.node_parents[node_id].extend(parent_nodes)

            for node_id, parents_ids in self.state.node_parents.items():
                if node_id not in self.state.node_details and len(parents_ids) >= 2:
                    parent_id1, parent_id2 = parents_ids[0], parents_ids[1]
                    if (
                        parent_id1 in self.state.node_details
                        and parent_id2 in self.state.node_details
                    ):
                        self.state.node_details[node_id] = (
                            ChunkProcessor.compete_chunks(
                                self.chunk_manager,
                                self.state.node_details[parent_id1],
                                self.state.node_details[parent_id2],
                            )
                        )

            return self.add_cors_headers(
                jsonify(
                    {
                        'message': 'Uptree updates processed',
                        'node_parents': self.state.node_parents,
                    }
                )
            )

        @self.app.route('/api/final-node', methods=['POST', 'OPTIONS'])
        def handle_final_node() -> ResponseType:
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json() or {}
            node_id: str = data.get('node_id', '')
            parents: List[str] = data.get('parents', [])

            if node_id:
                self.state.node_parents[node_id] = parents

                for curr_node_id, parents_ids in self.state.node_parents.items():
                    if curr_node_id not in self.state.node_details and parents_ids:
                        parent_id = parents_ids[0]
                        if parent_id in self.state.node_details:
                            answer, confidence_score = self.ctm.ask_supervisor(
                                self.state.query, self.state.node_details[parent_id]
                            )
                            self.state.node_details[curr_node_id] = (
                                f'Answer: {answer}\n\nConfidence score: {confidence_score}'
                            )
                            self.state.winning_chunk = self.state.node_details[
                                parent_id
                            ]

            return self.add_cors_headers(
                jsonify(
                    {
                        'message': 'Final node updated',
                        'node_parents': self.state.node_parents,
                    }
                )
            )

        @self.app.route('/api/reverse', methods=['POST', 'OPTIONS'])
        def handle_reverse() -> ResponseType:
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            if self.state.winning_chunk:
                self.ctm.downtree_broadcast(self.state.winning_chunk)

            return self.add_cors_headers(
                jsonify({'message': 'Reverse broadcast processed'})
            )

        @self.app.route('/api/update-processors', methods=['POST', 'OPTIONS'])
        def update_processors() -> ResponseType:
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json() or {}
            updates: List[Dict[str, str]] = data.get('updates', [])

            self.ctm.link_form(self.state.chunks)
            self.state.chunks = []
            self.state.node_details.clear()
            self.state.node_parents.clear()
            self.state.node_gists.clear()

            for update in updates:
                proc_id = update.get('processor_id', '')
                if proc_id in self.state.node_details:
                    self.state.node_details[proc_id] = f'Updated processor {proc_id}'

            return self.add_cors_headers(jsonify({'message': 'Processors updated'}))

        @self.app.route('/api/fuse-gist', methods=['POST', 'OPTIONS'])
        def handle_fuse_gist() -> ResponseType:
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            data = request.get_json() or {}
            updates: List[Dict[str, Any]] = data.get('updates', [])

            self.state.chunks = ChunkProcessor.fuse_chunks(self.ctm, self.state.chunks)

            for update in updates:
                fused_node_id = update.get('fused_node_id', '')
                source_nodes: List[str] = update.get('source_nodes', [])

                if fused_node_id and source_nodes:
                    source_chunks = [
                        self.state.node_details[node_id]
                        for node_id in source_nodes
                        if node_id in self.state.node_details
                    ]
                    fused_chunk = source_chunks[0] if source_chunks else None

                    if fused_chunk:
                        self.state.node_details[fused_node_id] = fused_chunk
                        self.state.node_parents[fused_node_id] = source_nodes

            return self.add_cors_headers(
                jsonify({'message': 'Fused gists processed', 'updates': updates})
            )

        @self.app.route('/api/fetch-neighborhood', methods=['GET', 'OPTIONS'])
        def get_processor_neighborhoods() -> ResponseType:
            if request.method == 'OPTIONS':
                return self.handle_options_request()

            neighborhoods: Dict[str, List[str]] = {}
            graph = self.ctm.processor_graph.graph

            for processor, connected_processors in graph.items():
                neighborhoods[processor.name] = [p.name for p in connected_processors]

            return self.add_cors_headers(jsonify(neighborhoods))

        @self.app.route('/api/upload', methods=['POST', 'OPTIONS'])
        def upload_files() -> Tuple[ResponseType, int]:
            if request.method == 'OPTIONS':
                return self.handle_options_request(), 200

            # Get form data
            self.state.query = request.form.get('query', '') or ''
            text = request.form.get('text', '') or ''

            saved_files: Dict[str, List[str]] = {
                'images': [],
                'audios': [],
                'videos': [],
                'video_frames': [],
            }

            # Process each file type
            for file_type in ['images', 'audios', 'videos']:
                if file_type in request.files:
                    files = request.files.getlist(file_type)
                    for file in files:
                        unique_filename = FileHandler.save_file(
                            file, file_type, self.app
                        )
                        if unique_filename:
                            file_saved_path = os.path.join(
                                self.app.config['UPLOAD_FOLDER'],
                                file_type,
                                unique_filename,
                            )
                            if file_type == 'videos':
                                saved_files['videos'].append(unique_filename)
                                video_path = file_saved_path
                                frame_output_dir = os.path.join(
                                    self.app.config['UPLOAD_FOLDER'], 'video_frames'
                                )
                                frame_filenames = extract_video_frames(
                                    video_path, frame_output_dir, 10
                                )
                                saved_files['video_frames'].extend(frame_filenames)
                            else:
                                saved_files[file_type].append(unique_filename)
                        else:
                            return self.add_cors_headers(
                                jsonify(
                                    {
                                        'error': f'Invalid {file_type} file: {file.filename}'
                                    }
                                )
                            ), 400

            if saved_files['videos'] and not saved_files['audios']:
                for video_filename in saved_files['videos']:
                    video_path = os.path.join(
                        self.app.config['UPLOAD_FOLDER'], 'videos', video_filename
                    )
                    audio_output_dir = os.path.join(
                        self.app.config['UPLOAD_FOLDER'], 'audios'
                    )
                    try:
                        extracted_audio = extract_audio_from_video(
                            video_path, audio_output_dir, audio_format='mp3'
                        )
                        saved_files['audios'].append(extracted_audio)
                    except Exception as e:
                        return self.add_cors_headers(
                            jsonify(
                                {
                                    'error': f'Extracting {video_filename} error: {str(e)}'
                                }
                            )
                        ), 500

            self.state.saved_files = saved_files

            response_data = {
                'message': 'Files uploaded successfully',
                'query': self.state.query,
                'text': text,
                'saved_files': saved_files,
                'download_links': {
                    ftype: [f'/uploads/{ftype}/{fname}' for fname in fnames]
                    for ftype, fnames in saved_files.items()
                },
            }

            return self.add_cors_headers(jsonify(response_data)), 200

        @self.app.route('/uploads/<file_type>/<filename>')
        def uploaded_file(file_type: str, filename: str) -> Tuple[ResponseType, int]:
            if file_type not in ['images', 'audios', 'videos']:
                return self.add_cors_headers(
                    jsonify({'error': 'Invalid file type'})
                ), 400

            try:
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], file_type)
                response = send_from_directory(file_path, filename)
                return self.add_cors_headers(response), 200
            except FileNotFoundError:
                return self.add_cors_headers(jsonify({'error': 'File not found'})), 404

    def run(self, **kwargs: Any) -> None:
        # Create upload directories
        for file_type in ['images', 'audios', 'videos']:
            os.makedirs(
                os.path.join(self.app.config['UPLOAD_FOLDER'], file_type), exist_ok=True
            )
        self.app.run(**kwargs)
