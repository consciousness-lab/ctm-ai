from flask import Flask, jsonify, make_response

app = Flask(__name__)


@app.route('/api/nodes/<node_id>')
def get_node_details(node_id):
    # Build your response here...
    response_data = {'details': f'This is node {node_id}'}
    response = make_response(jsonify(response_data))

    # Add CORS headers:
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')

    return response


if __name__ == '__main__':
    app.run(port=5000)
