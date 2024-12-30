from flask import Flask, jsonify, make_response, request

from ctm_ai.ctms.ctm import ConsciousnessTuringMachine
from ctm_ai.chunks import Chunk, ChunkManager

ctm = ConsciousnessTuringMachine()

app = Flask(__name__)

# Data storage
node_details = {}
node_parents = {}
node_gists = {}
winning_chunk = None


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
                details['parents'][parent]  = parent_details
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
    k = data.get('k', 3)
    
    # Clear previous data
    node_details.clear()
    node_parents.clear()
    node_gists.clear()
    
    print('Initializing processors')
    processor_names = [
        'gpt4v_processor',
        'gpt4_processor',
        'search_engine_processor',
        'wolfram_alpha_processor',
    ]
    
    # Store actual processor names we'll use
    selected_processors = []
    
    for i in range(k):
        processor_name = processor_names[i % len(processor_names)]
        node_id = f"{processor_name}"  # Create unique name
        node_details[node_id] = f'{processor_name}'  # Store original type in details
        ctm.add_processor(processor_name=processor_name)
        selected_processors.append(node_id)
        print(f"Added: {node_id} (type: {processor_name})")

        
    ctm.add_supervisor('gpt4_supervisor')
    ctm.add_scorer('gpt4_scorer')
    ctm.add_fuser('gpt4_fuser')
    
    response = jsonify({
        'message': 'Processors initialized',
        'processorNames': selected_processors  # Return the actual processor names used
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

        # add node_parents
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
            node_details[node_id] = ChunkManager().compete(node_details[parent_id1], node_details[parent_id2]) 

    response = jsonify(
        {'message': 'Uptree updates processed', 'node_parents': node_parents}
    )
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
            answer, confidence_score = ctm.ask_supervisor('What is the capital of France?', node_details[parent_id])
            node_details[node_id] = 'Answer: ' + answer + f'\n\nConfidence score: {confidence_score}'
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

    data = request.get_json()
    updates = data.get('updates', [])

    print('handling reverse')
    ctm.downtree_broadcast(winning_chunk)

    response = jsonify({'message': 'Reverse broadcast processed'})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/update-processors', methods=['POST', 'OPTIONS'])
def update_processors():
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
    updates = data.get('updates', [])

    chunks = []
    for node in node_details:
        if isinstance(node_details[node], Chunk):
            chunks.append(node_details[node])
    
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


if __name__ == '__main__':
    app.run(port=5000)
