from flask import Flask, jsonify, make_response, request

from ctm_ai.ctms.ctm import ConsciousnessTuringMachine

ctm = ConsciousnessTuringMachine()

app = Flask(__name__)

# Data storage
node_details = {}
node_parents = {}
node_gists = {}


@app.route('/api/nodes/<node_id>')
def get_node_details(node_id):
    print(f'Requested node_id: {node_id}')
    details = {'self': node_details.get(node_id, 'No details available')}

    if node_id in node_parents:
        details['parents'] = {
            parent: node_details.get(parent, 'No details available')
            for parent in node_parents[node_id]
        }
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
    for i in range(k):
        processor_name = processor_names[i % len(processor_names)]
        node_details[f'{processor_name}'] = f'{processor_name}'
        ctm.add_processor(processor_name=processor_name)
        print(processor_name)
    
    ctm.add_supervisor('gpt4_supervisor')
    ctm.add_scorer('gpt4_scorer')
    ctm.add_fuser('gpt4_fuser')

    response = jsonify(
        {'message': 'Processors initialized', 'processors': list(node_details.keys())}
    )
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
    for i, chunk in enumerate(chunks):
        gists[f"p{i+1}"] = chunk.gist
    print('handling output gist')
    import pdb; pdb.set_trace()
    for update in updates:
        proc_id = update.get('processor_id')
        target_id = update.get('target_id')

        node_details[target_id] = gists[proc_id]
        if target_id not in node_parents:
            node_parents[target_id] = []
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

        node_details[node_id] = f'Processing node {node_id}'

    print('Current node parents:', node_parents)
    response = jsonify(
        {'message': 'Uptree updates processed', 'node_parents': node_parents}
    )
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/final-node', methods=['POST', 'OPTIONS'])
def handle_final_node():
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
    node_details[node_id] = 'Final output node'
    node_parents[node_id] = parents

    print('Final node parents:', node_parents)
    response = jsonify({'message': 'Final node updated', 'node_parents': node_parents})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/reverse', methods=['POST', 'OPTIONS'])
def handle_reverse():
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
    for node_id in node_details:
        node_details[node_id] = f'Broadcasting to node {node_id}'

    response = jsonify({'message': 'Reverse broadcast processed'})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


@app.route('/api/update-processors', methods=['POST', 'OPTIONS'])
def update_processors():
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
