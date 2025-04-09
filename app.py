from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
import re
import uuid
from datetime import datetime
from g4f.client import Client
from g4f.Provider import Qwen_Qwen_2_5_Max, Blackbox, Copilot, CohereForAI_C4AI_Command, DeepInfraChat, Glider, Dynaspark, OpenaiChat

app = Flask(__name__)
# Enable CORS for all routes and all origins
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Initialize the g4f client
client = Client()

# In-memory conversation storage
# In a production environment, you would use a database
conversations = {}

# Dictionary to map provider names to their respective provider classes
providers = {
    "Qwen_Qwen_2_5_Max": Qwen_Qwen_2_5_Max,
    "Blackbox": Blackbox,
    "Copilot": Copilot,
    "CohereForAI_C4AI_Command": CohereForAI_C4AI_Command,
    "DeepInfraChat": DeepInfraChat,
    "Glider": Glider,
    "Dynaspark": Dynaspark,
    "OpenaiChat": OpenaiChat
}

# Custom JSON encoder to handle non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def clean_text(text):
    """Remove think tags and unnecessary whitespace"""
    # Remove <think> tags and their content
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def safe_str(content):
    """Safely convert any content to string"""
    if isinstance(content, str):
        return content
    return str(content)

def get_response_stream(provider_name, model_name, messages, text_only=False, conversation_id=None):
    """Generator function that yields streaming response chunks"""
    provider = providers.get(provider_name)
    if not provider:
        yield f"data: {json.dumps({'error': f'Invalid provider: {provider_name}'})}\n\n"
        return

    try:
        # Buffering mechanism for text-only mode
        buffer = ""
        full_response = ""  # Track the complete response for streaming
        
        chat_completion = client.chat.completions.create(
            provider=provider,
            model=model_name,
            messages=messages,
            stream=True
        )
        
        for completion in chat_completion:
            try:
                # Extract content from various possible formats
                if hasattr(completion, 'choices') and hasattr(completion.choices[0], 'delta'):
                    if hasattr(completion.choices[0].delta, 'content'):
                        content = completion.choices[0].delta.content or ""
                    else:
                        content = safe_str(completion.choices[0].delta)
                else:
                    content = safe_str(completion)
                
                # Add content to our full response
                full_response += safe_str(content)
                
                if text_only:
                    # Process content in larger chunks for better cleaning
                    if len(full_response) > 50 or "</think>" in full_response:
                        cleaned = clean_text(full_response)
                        if cleaned:
                            json_data = json.dumps({'content': cleaned}, cls=CustomJSONEncoder)
                            yield f"data: {json_data}\n\n"
                            full_response = ""  # Reset after sending
                else:
                    # Normal mode - send each chunk as-is
                    json_data = json.dumps({'content': content}, cls=CustomJSONEncoder)
                    yield f"data: {json_data}\n\n"
                    
            except Exception as e:
                # Log the error but continue processing
                print(f"Error processing chunk: {str(e)}")
                continue
                
        # Send any remaining content
        if text_only and full_response:
            cleaned = clean_text(full_response)
            if cleaned:
                json_data = json.dumps({'content': cleaned}, cls=CustomJSONEncoder)
                yield f"data: {json_data}\n\n"
        
        # When streaming is complete, save the assistant's response to conversation history
        if conversation_id and conversation_id in conversations:
            # Extract and clean the final response
            final_response = clean_text(full_response) if text_only else full_response
            # Save to conversation history
            conversations[conversation_id]['messages'].append({
                "role": "assistant", 
                "content": final_response
            })
            conversations[conversation_id]['updated_at'] = datetime.now().isoformat()
                
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)}, cls=CustomJSONEncoder)}\n\n"

@app.route('/api/chat/completions', methods=['POST'])
def chat_completions():
    try:
        if request.is_json:
            data = request.json
        else:
            try:
                data = json.loads(request.data.decode('utf-8'))
            except:
                return jsonify({"error": "Invalid JSON data"}), 400
        
        provider_name = data.get('provider')
        model_name = data.get('model')
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        text_only = data.get('text_only', False)
        conversation_id = data.get('conversation_id')
        
        # If a conversation_id is provided, check if it exists
        if conversation_id:
            if conversation_id in conversations:
                # If client doesn't send messages but we have them in memory, use those
                if not messages and conversations[conversation_id]['messages']:
                    messages = conversations[conversation_id]['messages']
                # Otherwise, update our stored messages with what the client sent
                else:
                    conversations[conversation_id]['messages'] = messages
                    conversations[conversation_id]['updated_at'] = datetime.now().isoformat()
            else:
                # Create a new conversation with this ID
                conversations[conversation_id] = {
                    'id': conversation_id,
                    'messages': messages,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
        # If no conversation_id is provided, generate one and create a new conversation
        elif messages:
            conversation_id = str(uuid.uuid4())
            conversations[conversation_id] = {
                'id': conversation_id,
                'messages': messages,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
        
        if not all([provider_name, model_name, messages]):
            return jsonify({"error": "Missing required parameters"}), 400
        
        if not provider_name in providers:
            return jsonify({"error": f"Invalid provider: {provider_name}"}), 400
        
        # For user messages, add to conversation history
        if conversation_id and messages and messages[-1]['role'] == 'user':
            # Make sure we're tracking this conversation
            if conversation_id not in conversations:
                conversations[conversation_id] = {
                    'id': conversation_id,
                    'messages': [],
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
            
            # If the last message from the client was from the user, add it to history
            last_message = messages[-1]
            if last_message not in conversations[conversation_id]['messages']:
                conversations[conversation_id]['messages'].append(last_message)
                conversations[conversation_id]['updated_at'] = datetime.now().isoformat()
        
        if stream:
            return Response(
                get_response_stream(provider_name, model_name, messages, text_only, conversation_id),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Connection': 'keep-alive',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS'
                }
            )
        else:
            try:
                provider = providers.get(provider_name)
                response = client.chat.completions.create(
                    provider=provider,
                    model=model_name,
                    messages=messages,
                    stream=False
                )
                
                try:
                    # Extract response content safely
                    if hasattr(response, 'choices') and hasattr(response.choices[0], 'message'):
                        content = safe_str(response.choices[0].message.content)
                    else:
                        content = safe_str(response)
                    
                    # Clean the content if text_only is True
                    if text_only:
                        content = clean_text(content)
                    
                    # Save to conversation history if conversation_id is provided
                    if conversation_id and conversation_id in conversations:
                        conversations[conversation_id]['messages'].append({
                            "role": "assistant", 
                            "content": content
                        })
                        conversations[conversation_id]['updated_at'] = datetime.now().isoformat()
                    
                    response_data = {
                        "id": response.id if hasattr(response, 'id') else conversation_id or "unknown",
                        "choices": [{
                            "message": {
                                "role": "assistant", 
                                "content": content
                            }
                        }],
                        "model": model_name,
                        "conversation_id": conversation_id
                    }
                except AttributeError:
                    response_data = {
                        "content": content if 'content' in locals() else str(response),
                        "conversation_id": conversation_id
                    }
                
                return jsonify(response_data)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a route to get conversation history
@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    return jsonify(conversations[conversation_id])

# Add a route to list all conversations
@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    # Return a list of conversation metadata (without full message history)
    conversation_list = [
        {
            'id': conv_id,
            'created_at': data['created_at'],
            'updated_at': data['updated_at'],
            'message_count': len(data['messages'])
        }
        for conv_id, data in conversations.items()
    ]
    return jsonify({"conversations": conversation_list})

# Add a route to delete a conversation
@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    del conversations[conversation_id]
    return jsonify({"status": "success", "message": "Conversation deleted"})

# Add OPTIONS method handler for CORS preflight requests
@app.route('/api/chat/completions', methods=['OPTIONS'])
def options():
    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/api/providers', methods=['GET'])
def get_providers():
    return jsonify({"providers": list(providers.keys())})

# Add OPTIONS method handlers for the other endpoints
@app.route('/api/providers', methods=['OPTIONS'])
@app.route('/api/conversations', methods=['OPTIONS'])
@app.route('/api/conversations/<conversation_id>', methods=['OPTIONS'])
def other_options():
    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
