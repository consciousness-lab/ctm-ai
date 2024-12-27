from flask import Flask, jsonify, make_response, request

app = Flask(__name__)

# Example: Dictionary to store node details
node_details = {
    'init1': 'Processor 1 details',
    'init2': 'Processor 2 details',
    'n1': 'Node 1 details',
    'n2': 'Node 2 details',
    'final-node': 'Final output node details',
}

node_parents = {
    'n1': ['init1', 'init2'],
    'n2': ['init2', 'init3'],
    'final-node': ['n1', 'n2'],
}


@app.route('/api/nodes/<node_id>')
def get_node_and_parent_details(node_id):
    print(f'Requested node_id: {node_id}')
    # if node_id not in node_details:
    #    return make_response(jsonify({'error': 'Node not found'}), 404)

    # Get the node's details
    # details = {'self': node_details[node_id]}
    details = {'self': node_details.get(node_id, 'No details available')}

    # Get the parent's details if available
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


@app.route('/api/update-node-parents', methods=['POST', 'OPTIONS'])
def update_node_parents():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add(
            'Access-Control-Allow-Headers', 'Content-Type,Authorization'
        )
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    # Handle the actual POST request
    data = request.get_json()
    updates = data.get('updates', [])
    for update in updates:
        node_id = update.get('node_id')
        parents = update.get('parents', [])
        if node_id and node_id not in node_parents:
            node_parents[node_id] = parents
        if node_id and node_id in node_parents:
            node_parents[node_id] += parents
    print(data)
    print(node_parents)
    response = jsonify(
        {'message': 'Node parents updated successfully', 'node_parents': node_parents}
    )
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response


if __name__ == '__main__':
    app.run(port=5000)
