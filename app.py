from flask import Flask, request, jsonify
import asyncio
import json
import hashlib
import requests
from datetime import datetime
from entient_ultimate_system import CompleteSystem, MetaAgentHarness
from database import init_db, get_seal, get_all_seals  # ADD THIS

app = Flask(__name__)

# Initialize database tables
init_db()  # ADD THIS

# Global system instance
system = None
harness = None

def init_system():
    global system, harness
    system = CompleteSystem(num_agents=6)
    harness = MetaAgentHarness(system)

@app.route('/')
def home():
    return jsonify({
        "service": "ENTIENT Discovery Engine",
        "version": "7.0",
        "endpoints": {
            "/discover": "POST - Run discovery on a problem",
            "/verify/<seal_id>": "GET - Verify a seal",
            "/timestamp/<seal_id>": "POST - Anchor seal to Bitcoin",
            "/seals": "GET - List all seals"  # ADD THIS
        }
    })

@app.route('/discover', methods=['POST'])
def discover():
    data = request.json
    problem = data.get('problem', {
        'description': 'Optimize general system',
        'constraints': {}
    })
    
    # Run async discovery in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        harness.run_adaptive_discovery(
            problems=[problem],
            max_generations=3,
            target_fitness=0.85
        )
    )
    
    # Get the latest seal
    if system.seal_registry.seals:
        latest_seal = list(system.seal_registry.seals.values())[-1]
        seal_data = {
            'seal_id': latest_seal.session_uuid,
            'hash': latest_seal.content_hash,
            'classification': latest_seal.discovery_classification.value,
            'fitness': latest_seal.fitness_score
        }
    else:
        seal_data = None
    
    return jsonify({
        'status': 'complete',
        'seal': seal_data,
        'metrics': {
            'breakthroughs': result.get('total_breakthroughs', 0),
            'fitness': result.get('avg_final_fitness', 0)
        }
    })

@app.route('/verify/<seal_id>')
def verify(seal_id):
    if not system:
        init_system()
    
    # First check memory
    if system and system.seal_registry.verify_seal_exists(seal_id):
        seal = system.seal_registry.get_seal(seal_id)
        return jsonify({
            'valid': True,
            'source': 'memory',
            'seal': {
                'id': seal.session_uuid,
                'hash': seal.content_hash,
                'timestamp': seal.timestamp_utc,
                'classification': seal.discovery_classification.value
            }
        })
    
    # Then check database
    db_seal = get_seal(seal_id)
    if db_seal:
        return jsonify({
            'valid': True,
            'source': 'database',
            'seal': {
                'id': db_seal['seal_id'],
                'hash': db_seal['content_hash'],
                'timestamp': str(db_seal['timestamp_utc']),
                'classification': db_seal['classification']
            }
        })
    
    return jsonify({'valid': False, 'error': 'Seal not found'}), 404

@app.route('/timestamp/<seal_id>', methods=['POST'])
def timestamp_seal(seal_id):
    # Check memory first
    seal = system.seal_registry.get_seal(seal_id) if system else None
    
    # If not in memory, check database
    if not seal:
        db_seal = get_seal(seal_id)
        if not db_seal:
            return jsonify({'error': 'Seal not found'}), 404
        # Use database data for timestamping
        content_hash = db_seal['content_hash']
    else:
        content_hash = seal.content_hash
    
    # Create OpenTimestamps proof
    try:
        # Call OpenTimestamps API
        ots_url = "https://alice.btc.calendar.opentimestamps.org/digest"
        response = requests.post(ots_url, data=content_hash.encode())
        
        if response.status_code == 200:
            return jsonify({
                'status': 'pending',
                'seal_id': seal_id,
                'hash': content_hash,
                'ots_proof': response.content.hex(),
                'message': 'Proof submitted to Bitcoin blockchain'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ADD THIS: New endpoint to list all seals
@app.route('/seals')
def list_seals():
    seals = get_all_seals()
    return jsonify({
        'count': len(seals),
        'seals': [
            {
                'id': s['seal_id'],
                'classification': s['classification'],
                'fitness': s['fitness_score'],
                'created': str(s['created_at'])
            } for s in seals
        ]
    })

if __name__ == '__main__':
    init_system()
    app.run(debug=True, port=5000)