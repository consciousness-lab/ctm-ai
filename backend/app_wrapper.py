import os
from typing import Any, Dict, List, Optional, Tuple, Union

from chunk_processor import ChunkProcessor
from config import Config
from file_handler import FileHandler
from flask import Flask, jsonify, make_response, request, send_from_directory
from flask.wrappers import Response as FlaskResponse
from state import AppState
from werkzeug.wrappers.response import Response as WerkzeugResponse

from ctm_ai.chunks import Chunk, ChunkManager
from ctm_ai.ctms.ctm import ConsciousTuringMachine
from ctm_ai.utils import extract_audio_from_video, extract_video_frames

ResponseType = Union[FlaskResponse, WerkzeugResponse]


from ctm_ai.graphs import ProcessorGraph

class FlaskAppWrapper:
    def __init__(self) -> None:
        self.app: Flask = Flask(__name__)
        self.ctm: ConsciousTuringMachine = ConsciousTuringMachine()
        self.state: AppState = AppState()
        self.chunk_manager: ChunkManager = ChunkManager()
        self.setup_app_config()
        self.register_routes()

    def _get_input_params(self) -> Dict[str, Any]:
        """获取当前的输入参数，用于传递给CTM核心方法"""
        # Check for example paths first (from load-example endpoint)
        example_image = getattr(self.state, 'example_image_path', None)
        example_audio = getattr(self.state, 'example_audio_path', None)
        
        # Use example paths if available, otherwise use uploaded files
        if example_image or example_audio:
            image_path = example_image
            audio_path = example_audio
            video_path = None
            video_frames_path = None
        else:
            image_filename = (
                self.state.saved_files['images'][0]
                if self.state.saved_files['images']
                else None
            )
            image_path = (
                os.path.join(self.app.config['UPLOAD_FOLDER'], 'images', image_filename)
                if image_filename
                else None
            )

            audio_filename = (
                self.state.saved_files['audios'][0]
                if self.state.saved_files['audios']
                else None
            )
            audio_path = (
                os.path.join(self.app.config['UPLOAD_FOLDER'], 'audios', audio_filename)
                if audio_filename
                else None
            )

            video_filename = (
                self.state.saved_files['videos'][0]
                if self.state.saved_files['videos']
                else None
            )
            video_path = (
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
                else None
            )

        return {
            'text': self.state.text,
            'image_path': image_path,
            'audio_path': audio_path,
            'video_frames_path': video_frames_path,
            'video_path': video_path,
        }

    def setup_app_config(self) -> None:
        self.app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        self.app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

    def register_routes(self) -> None:
        @self.app.route('/api/refresh', methods=['POST'])
        def handle_refresh() -> ResponseType:
            self.state.reset()
            # Clear example paths
            self.state.example_image_path = None
            self.state.example_audio_path = None
            return jsonify({'message': 'Server data refreshed'})

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

            # Check if this is a processor node and get additional info
            processor_info: Optional[Dict[str, Any]] = None
            processor = self.ctm.processor_graph.get_node(node_id)
            if processor:
                # Get linked processors
                linked_processors = self.ctm.processor_graph.adjacency_list.get(node_id, [])
                
                # Get processor memory (history)
                memory = {
                    'fuse_history': getattr(processor, 'fuse_history', []),
                    'winner_answer': getattr(processor, 'winner_answer', []),
                    'all_context_history': getattr(processor, 'all_context_history', []),
                }
                
                processor_info = {
                    'name': processor.name,
                    'type': processor.name.split('_')[0].replace('Processor', ''),
                    'model': getattr(processor, 'model_name', 'N/A'),
                    'linked_processors': linked_processors,
                    'memory': memory,
                }

            return jsonify({
                'self': node_self, 
                'parents': parent_data,
                'processor_info': processor_info
            })

        @self.app.route('/api/init', methods=['POST'])
        def initialize_processors() -> Tuple[ResponseType, int]:
            data = request.get_json() or {}
            selected_processors: List[str] = data.get('selected_processors', [])

            if not selected_processors:
                return (
                    jsonify({'error': 'No processors provided'}),
                    400,
                )

            self.state.node_details.clear()
            self.state.node_parents.clear()
            self.state.node_gists.clear()

            self.ctm.reset()
            self.ctm.processor_graph = ProcessorGraph()
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

            return (
                jsonify(
                    {
                        'message': 'Processors initialized',
                        'processorNames': created_processor_names,
                    }
                ),
                200,
            )

        @self.app.route('/api/output-gist', methods=['POST'])
        def handle_output_gist() -> ResponseType:
            """
            处理 output-gist 步骤：调用 ask_processors 获取所有处理器的输出。
            对应 CTM 核心的 ask_processors 方法。
            """
            data = request.get_json() or {}
            updates: List[Dict[str, str]] = data.get('updates', [])

            # 获取输入参数
            input_params = self._get_input_params()

            # 调用 ask_processors
            gists, chunks = ChunkProcessor.process_chunks(
                ctm_instance=self.ctm,
                query=self.state.query,
                text=input_params.get('text'),
                image_path=input_params.get('image_path'),
                audio_path=input_params.get('audio_path'),
                video_frames_path=input_params.get('video_frames_path'),
                video_path=input_params.get('video_path'),
            )

            # 存储 chunks 用于后续步骤
            self.state.chunks = chunks

            for update in updates:
                proc_id = update.get('processor_id', '')
                target_id = update.get('target_id', '')
                if proc_id and target_id and proc_id in gists:
                    self.state.node_details[target_id] = gists[proc_id]

                    if target_id not in self.state.node_parents:
                        self.state.node_parents[target_id] = [proc_id]
                    else:
                        self.state.node_parents[target_id].append(proc_id)

            return jsonify({'message': 'Gist outputs processed', 'updates': updates})

        @self.app.route('/api/uptree', methods=['POST'])
        def handle_uptree() -> ResponseType:
            """
            处理 uptree 步骤：进行上树竞争。
            对应 CTM 核心的 uptree_competition 方法（通过 ChunkManager）。
            """
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

            # 使用 ChunkManager 进行两两竞争或单个继承
            for node_id, parents_ids in self.state.node_parents.items():
                if node_id not in self.state.node_details and parents_ids:
                    if len(parents_ids) == 1:
                        # 只有一个 parent，直接继承
                        parent_id = parents_ids[0]
                        if parent_id in self.state.node_details:
                            self.state.node_details[node_id] = self.state.node_details[parent_id]
                    elif len(parents_ids) >= 2:
                        parent_id1, parent_id2 = parents_ids[0], parents_ids[1]
                        
                        has_p1 = parent_id1 in self.state.node_details
                        has_p2 = parent_id2 in self.state.node_details

                        if has_p1 and has_p2:
                            parent_chunk1 = self.state.node_details[parent_id1]
                            parent_chunk2 = self.state.node_details[parent_id2]

                            # 只有当两个都是 Chunk 时才进行竞争
                            if isinstance(parent_chunk1, Chunk) and isinstance(
                                parent_chunk2, Chunk
                            ):
                                self.state.node_details[node_id] = (
                                    ChunkProcessor.compete_chunks(
                                        self.chunk_manager,
                                        parent_chunk1,
                                        parent_chunk2,
                                    )
                                )
                            else:
                                # 如果不是 Chunk，取第一个有效的
                                self.state.node_details[node_id] = (
                                    parent_chunk1
                                    if isinstance(parent_chunk1, Chunk)
                                    else parent_chunk2
                                )
                        elif has_p1:
                            # 只有一个父节点有值，直接晋级
                            self.state.node_details[node_id] = self.state.node_details[parent_id1]
                        elif has_p2:
                            # 只有一个父节点有值，直接晋级
                            self.state.node_details[node_id] = self.state.node_details[parent_id2]

            return jsonify(
                {
                    'message': 'Uptree updates processed',
                    'node_parents': self.state.node_parents,
                }
            )

        @self.app.route('/api/final-node', methods=['POST'])
        def handle_final_node() -> ResponseType:
            """
            处理 final-node 步骤：调用 ask_supervisor 获取最终答案。
            对应 CTM 核心的 ask_supervisor 方法。
            """
            data = request.get_json() or {}
            node_id: str = data.get('node_id', '')
            parents: List[str] = data.get('parents', [])

            if node_id:
                self.state.node_parents[node_id] = parents

                for curr_node_id, parents_ids in self.state.node_parents.items():
                    if curr_node_id not in self.state.node_details and parents_ids:
                        parent_id = parents_ids[0]
                        if parent_id in self.state.node_details:
                            parent_chunk = self.state.node_details[parent_id]

                            # 只有当父节点是 Chunk 时才调用 ask_supervisor
                            if isinstance(parent_chunk, Chunk) and self.state.query:
                                answer, confidence_score = (
                                    ChunkProcessor.ask_supervisor(
                                        ctm_instance=self.ctm,
                                        query=self.state.query,
                                        winning_chunk=parent_chunk,
                                    )
                                )
                                self.state.node_details[curr_node_id] = (
                                    f'Answer: {answer}\n\n'
                                    f'Confidence score: {confidence_score}'
                                )
                                self.state.winning_chunk = parent_chunk
                            else:
                                # 如果不是 Chunk，直接使用父节点的内容
                                self.state.node_details[curr_node_id] = str(
                                    parent_chunk
                                )

            return jsonify(
                {
                    'message': 'Final node updated',
                    'node_parents': self.state.node_parents,
                }
            )

        @self.app.route('/api/reverse', methods=['POST'])
        def handle_reverse() -> ResponseType:
            """
            处理 reverse 步骤：调用 downtree_broadcast 进行下树广播。
            对应 CTM 核心的 downtree_broadcast 方法。
            """
            if self.state.winning_chunk and isinstance(self.state.winning_chunk, Chunk):
                ChunkProcessor.downtree_broadcast(
                    ctm_instance=self.ctm,
                    winning_chunk=self.state.winning_chunk,
                )

            return jsonify({'message': 'Reverse broadcast processed'})

        @self.app.route('/api/update-processors', methods=['POST'])
        def update_processors() -> ResponseType:
            """
            处理 update-processors 步骤：调用 link_form 形成处理器之间的链接。
            对应 CTM 核心的 link_form 方法。
            """
            data = request.get_json() or {}
            updates: List[Dict[str, str]] = data.get('updates', [])

            # 获取输入参数
            input_params = self._get_input_params()

            # 调用 link_form 形成处理器之间的链接
            if (
                self.state.chunks
                and self.state.winning_chunk
                and isinstance(self.state.winning_chunk, Chunk)
            ):
                ChunkProcessor.link_form(
                    ctm_instance=self.ctm,
                    chunks=self.state.chunks,
                    winning_chunk=self.state.winning_chunk,
                    **input_params,
                )

            # 重置状态准备下一轮迭代
            self.state.chunks = []
            self.state.node_details.clear()
            self.state.node_parents.clear()
            self.state.node_gists.clear()

            # 重新初始化处理器节点详情
            for processor in self.ctm.processor_graph.nodes:
                self.state.node_details[processor.name] = processor.name

            return jsonify({'message': 'Processors updated'})

        @self.app.route('/api/fuse-gist', methods=['POST'])
        def handle_fuse_gist() -> ResponseType:
            """
            处理 fuse-gist 步骤：调用 fuse_processor 融合处理器输出。
            对应 CTM 核心的 fuse_processor 方法。
            """
            data = request.get_json() or {}
            updates: List[Dict[str, Any]] = data.get('updates', [])

            # 获取输入参数
            input_params = self._get_input_params()

            # 调用 fuse_processor，传递正确的参数
            if self.state.chunks and self.state.query:
                self.state.chunks = ChunkProcessor.fuse_chunks(
                    ctm_instance=self.ctm,
                    chunks=self.state.chunks,
                    query=self.state.query,
                    **input_params,
                )

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

            return jsonify({'message': 'Fused gists processed', 'updates': updates})

        @self.app.route('/api/fetch-neighborhood', methods=['GET'])
        def get_processor_neighborhoods() -> ResponseType:
            neighborhoods: Dict[str, List[str]] = {}

            for processor_name, connected_processor_names in self.ctm.processor_graph.adjacency_list.items():
                neighborhoods[processor_name] = connected_processor_names

            return jsonify(neighborhoods)

        @self.app.route('/api/upload', methods=['POST'])
        def upload_files() -> Tuple[ResponseType, int]:
            """
            处理文件上传，保存查询和文本参数。
            """
            # Get form data
            self.state.query = request.form.get('query', '') or ''
            self.state.text = request.form.get('text', '') or ''
            text = self.state.text

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
                            return (
                                jsonify(
                                    {
                                        'error': f'Invalid {file_type} file: {file.filename}'
                                    }
                                ),
                                400,
                            )

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
                        return (
                            jsonify(
                                {
                                    'error': f'Extracting {video_filename} error: {str(e)}'
                                }
                            ),
                            500,
                        )

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

            return jsonify(response_data), 200

        @self.app.route('/uploads/<file_type>/<filename>')
        def uploaded_file(file_type: str, filename: str) -> Tuple[ResponseType, int]:
            if file_type not in ['images', 'audios', 'videos']:
                return (
                    jsonify({'error': 'Invalid file type'}),
                    400,
                )

            try:
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], file_type)
                response = send_from_directory(file_path, filename)
                return response, 200
            except FileNotFoundError:
                return jsonify({'error': 'File not found'}), 404

        @self.app.route('/assets/<path:filename>')
        def serve_assets(filename: str) -> ResponseType:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_path = os.path.join(project_root, 'assets')
            return send_from_directory(assets_path, filename)

        @self.app.route('/api/load-example', methods=['POST'])
        def load_example_files() -> Tuple[ResponseType, int]:
            """
            Load example files using relative paths directly from assets folder.
            """
            data = request.get_json() or {}
            image_path: str = data.get('image_path', '')
            audio_path: str = data.get('audio_path', '')
            
            # Set example query and text
            self.state.query = 'Is the person saying sarcasm or not?'
            self.state.text = 'You have no idea what you are talking about!'
            
            # Get the project root directory (parent of backend)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Store the absolute paths directly in state for use by _get_input_params
            self.state.example_image_path = os.path.join(project_root, image_path) if image_path else None
            self.state.example_audio_path = os.path.join(project_root, audio_path) if audio_path else None
            
            # Check if files exist
            files_found = {
                'image': self.state.example_image_path and os.path.exists(self.state.example_image_path),
                'audio': self.state.example_audio_path and os.path.exists(self.state.example_audio_path),
            }
            
            return jsonify({
                'success': True,
                'message': 'Example loaded successfully',
                'query': self.state.query,
                'text': self.state.text,
                'files_found': files_found,
                'image_url': f'/assets/{os.path.basename(image_path)}' if image_path else None,
                'audio_url': f'/assets/{os.path.basename(audio_path)}' if audio_path else None,
                'image_path': self.state.example_image_path,
                'audio_path': self.state.example_audio_path,
            }), 200

    def run(self, **kwargs: Any) -> None:
        # Create upload directories
        for file_type in ['images', 'audios', 'videos']:
            os.makedirs(
                os.path.join(self.app.config['UPLOAD_FOLDER'], file_type), exist_ok=True
            )
        self.app.run(**kwargs)
