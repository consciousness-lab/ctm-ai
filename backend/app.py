from flask import Flask, jsonify, make_response, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid

from ctm_ai.ctms.ctm import ConsciousnessTuringMachine
from ctm_ai.chunks import Chunk, ChunkManager

ctm = ConsciousnessTuringMachine()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    'images': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
    'audios': {'mp3', 'wav', 'aac', 'flac'},
    'videos': {'mp4', 'avi', 'mov', 'wmv', 'flv'}
}

# Data storage
node_details = {}
node_parents = {}
node_gists = {}
winning_chunk = None
chunks = []


def allowed_file(filename, file_type):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())


def generate_unique_filename(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    return unique_name


FRONTEND_TO_BACKEND_PROCESSORS = {
    "GPT4VProcessor": "gpt4v_processor",
    "GPT4Processor": "gpt4_processor",
    "SearchEngineProcessor": "search_engine_processor",
    "WolframAlphaProcessor": "wolfram_alpha_processor",
}


@app.route('/api/nodes/<node_id>')
def get_node_details(node_id):
    print(f'Requested node_id: {node_id}')

    details = node_details.get(node_id, 'No details available')
    if isinstance(details, Chunk):
        details = {'self': str(details.serialize())}
    else:
        details = {'self': details}

    if node_id in node_parents:
        details['parents'] = {}
        for parent in node_parents[node_id]:
            parent_details = node_details.get(parent, 'No details available')
            if isinstance(parent_details, Chunk):
                parent_details = str(parent_details.serialize())
                details['parents'][parent] = parent_details
    else:
        details['parents'] = {}

    response = make_response(jsonify(details))
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


@app.route('/api/init', methods=['POST', 'OPTIONS'])
def initialize_processors():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    data = request.get_json()
    selected_processors = data.get('selected_processors', [])

    if not selected_processors:
        error_response = jsonify({
            'error': 'No selected_processors provided or the provided list is empty'
        })
        error_response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        return error_response, 400

    # Clear previous data
    node_details.clear()
    node_parents.clear()
    node_gists.clear()

    print('Initializing processors')
    ctm.reset()

    created_processor_names = []

    for frontend_label in selected_processors:
        backend_processor_name = FRONTEND_TO_BACKEND_PROCESSORS.get(frontend_label)
        if not backend_processor_name:
            continue

        print(f"Adding processor: {backend_processor_name}")
        ctm.add_processor(processor_name=backend_processor_name)

        node_details[backend_processor_name] = backend_processor_name
        created_processor_names.append(backend_processor_name)

    ctm.add_supervisor('gpt4_supervisor')
    ctm.add_scorer('gpt4_scorer')
    ctm.add_fuser('gpt4_fuser')

    response = jsonify({
        'message': 'Processors initialized',
        'processorNames': created_processor_names
    })
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/output-gist', methods=['POST', 'OPTIONS'])
def handle_output_gist():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    data = request.get_json()
    updates = data.get('updates', [])

    chunks = ctm.ask_processors('What is the capital of France?')
    gists = [chunk.gist for chunk in chunks]
    gists = {}
    for chunk in chunks:
        gists[chunk.processor_name] = chunk

    for update in updates:
        proc_id = update.get('processor_id')
        target_id = update.get('target_id')
        node_details[target_id] = gists[proc_id]
        if target_id not in node_parents:
            node_parents[target_id] = [proc_id]
        else:
            node_parents[target_id].append(proc_id)

    response = jsonify({'message': 'Gist outputs processed', 'updates': updates})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/uptree', methods=['POST', 'OPTIONS'])
def handle_uptree():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    data = request.get_json()
    updates = data.get('updates', [])

    print('handling uptree')
    print(data)
    for update in updates:
        node_id = update.get('node_id')
        parent_nodes = update.get('parents', [])

        if node_id not in node_parents:
            node_parents[node_id] = parent_nodes
        else:
            node_parents[node_id] += parent_nodes

    print('Current node parents:', node_parents)

    for node_id, parents_ids in node_parents.items():
        if node_id not in node_details:
            parent_id1, parent_id2 = parents_ids[0], parents_ids[1]
            node_details[node_id] = ChunkManager().compete(
                node_details[parent_id1], node_details[parent_id2]
            )

    response = jsonify({'message': 'Uptree updates processed', 'node_parents': node_parents})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/final-node', methods=['POST', 'OPTIONS'])
def handle_final_node():
    global winning_chunk
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    data = request.get_json()
    node_id = data.get('node_id')
    parents = data.get('parents', [])

    print('handling final node')
    node_parents[node_id] = parents

    print('Final node parents:', node_parents)

    for node_id, parents_ids in node_parents.items():
        if node_id not in node_details:
            parent_id = parents_ids[0]
            answer, confidence_score = ctm.ask_supervisor(
                'What is the capital of France?', node_details[parent_id]
            )
            node_details[node_id] = (
                    'Answer: ' + answer + f'\n\nConfidence score: {confidence_score}'
            )
            winning_chunk = node_details[parent_id]

    response = jsonify({'message': 'Final node updated', 'node_parents': node_parents})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/reverse', methods=['POST', 'OPTIONS'])
def handle_reverse():
    global winning_chunk
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    print('handling reverse')
    ctm.downtree_broadcast(winning_chunk)

    response = jsonify({'message': 'Reverse broadcast processed'})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/update-processors', methods=['POST', 'OPTIONS'])
def update_processors():
    global winning_chunk
    global chunks
    chunks = []
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    data = request.get_json()
    updates = data.get('updates', [])

    ctm.link_form(chunks)
    node_details.clear()
    node_parents.clear()
    node_gists.clear()

    print('Updating processors')
    for update in updates:
        proc_id = update.get('processor_id')
        if proc_id in node_details:
            node_details[proc_id] = f'Updated processor {proc_id}'

    response = jsonify({'message': 'Processors updated'})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/fuse-gist', methods=['POST', 'OPTIONS'])
def handle_fuse_gist():
    global chunks
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    data = request.get_json()
    updates = data.get('updates', [])

    global chunks
    chunks = ctm.fuse_processor(chunks)

    # Process the fused nodes
    for update in updates:
        fused_node_id = update.get('fused_node_id')
        source_nodes = update.get('source_nodes', [])

        # Create fused chunk from source nodes
        source_chunks = [node_details[node_id] for node_id in source_nodes]

        # fused_chunk = ctm.fuse_chunks(source_chunks)  # Assuming you have this method
        fused_chunk = source_chunks[0]

        # Store the fused result
        node_details[fused_node_id] = fused_chunk

        # Update parent relationships
        node_parents[fused_node_id] = source_nodes

    response = jsonify({'message': 'Fused gists processed', 'updates': updates})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/fetch-neighborhood', methods=['GET', 'OPTIONS'])
def get_processor_neighborhoods():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response

    neighborhoods = {}
    graph = (
        ctm.processor_graph.graph
    )  # Assuming CTM stores processors in this attribute

    for processor, connected_processors in graph.items():
        neighborhoods[processor.name] = [p.name for p in connected_processors]

    response = jsonify(neighborhoods)
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_files():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    query = request.form.get('query', '')
    text = request.form.get('text', '')

    saved_files = {
        'images': [],
        'audios': [],
        'videos': []
    }

    if 'images' in request.files:
        images = request.files.getlist('images')
        for img in images:
            if img and allowed_file(img.filename, 'images'):
                filename = secure_filename(img.filename)
                unique_filename = generate_unique_filename(filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images', unique_filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                img.save(image_path)
                saved_files['images'].append(unique_filename)
            else:
                response = make_response(jsonify({'error': f'Invalid image file: {img.filename}'}), 400)
                response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
                return response

    if 'audios' in request.files:
        audios = request.files.getlist('audios')
        for aud in audios:
            if aud and allowed_file(aud.filename, 'audios'):
                filename = secure_filename(aud.filename)
                unique_filename = generate_unique_filename(filename)
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audios', unique_filename)
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                aud.save(audio_path)
                saved_files['audios'].append(unique_filename)
            else:
                response = make_response(jsonify({'error': f'Invalid audios file: {aud.filename}'}), 400)
                response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
                return jsonify({'error': f'Invalid audio file: {aud.filename}'}), 400

    if 'video_frames' in request.files:
        videos = request.files.getlist('video_frames')
        for vid in videos:
            if vid and allowed_file(vid.filename, 'videos'):
                filename = secure_filename(vid.filename)
                unique_filename = generate_unique_filename(filename)
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'videos', unique_filename)
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                vid.save(video_path)
                saved_files['videos'].append(unique_filename)
            else:
                response = make_response(jsonify({'error': f'Invalid video file: {vid.filename}'}), 400)
                response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
                return jsonify({'error': f'Invalid video file: {vid.filename}'}), 400

    response_data = {
        'message': 'Files uploaded successfully',
        'query': query,
        'text': text,
        'num_images': len(saved_files['images']),
        'num_audios': len(saved_files['audios']),
        'num_videos': len(saved_files['videos']),
        'saved_files': saved_files,
        'download_links': {
            'images': [f"/uploads/images/{filename}" for filename in saved_files['images']],
            'audios': [f"/uploads/audios/{filename}" for filename in saved_files['audios']],
            'videos': [f"/uploads/videos/{filename}" for filename in saved_files['videos']],
        }
    }

    response = make_response(jsonify(response_data), 200)
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Content-Type', 'application/json')
    return response


@app.route('/uploads/<file_type>/<filename>', methods=['GET'])
def uploaded_file(file_type, filename):
    if file_type not in ['images', 'audios', 'videos']:
        error_response = jsonify({'error': 'Invalid file type'})
        error_response.status_code = 400
        error_response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        return error_response

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_type)
        response = send_from_directory(file_path, filename)
        response = make_response(response)
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Cache-Control', 'public, max-age=3600')
        return response
    except FileNotFoundError:
        error_response = jsonify({'error': 'File not found'})
        error_response.status_code = 404
        error_response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        return error_response


if __name__ == '__main__':
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'audios'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'videos'), exist_ok=True)

    app.run(port=5000, debug=True)
