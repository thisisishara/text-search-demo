from flask import Flask, request, jsonify
from weaviate import Client
import openai
import json
from transformers import pipeline

app = Flask(__name__)

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize OpenAI API client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Weaviate client
weaviate_client = Client(os.getenv('WEAVIATE_URL'))

# Initialize summarization pipeline
summarizer = pipeline('summarization', model='t5-base', tokenizer='t5-base', framework='pt')

# Define endpoint for uploading documents
@app.route('/documents', methods=['POST'])
def upload_documents():
    documents = request.get_json()
    for doc in documents:
        # Generate OpenAI embedding for document
        embedding = openai.Completion.create(
            engine="text-davinci-002",
            prompt=doc['text'],
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        ).choices[0].text
        # Add document and embedding to Weaviate
        weaviate_client.batch.create('documents', json.dumps({
            "text": doc['text'],
            "embedding": embedding.strip()
        }))
    return jsonify({"message": "Documents uploaded successfully"})


# Define endpoint for searching and summarizing documents
@app.route('/search', methods=['POST'])
def search_documents():
    query = request.get_json()['query']
    # Generate OpenAI embedding for query
    query_embedding = openai.Completion.create(
        engine="text-davinci-002",
        prompt=query,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    ).choices[0].text
    # Search for documents in Weaviate
    response = weaviate_client.query.get(path='/graphql', params={
        'query': '''
        {
            Get {
                Documents(
                    where: {
                        Search(
                            vector: {
                                vector: "%s",
                                mode: COSINE_SIMILARITY
                            }
                        )
                    }
                ) {
                    text
                }
            }
        }
        ''' % query_embedding.strip()
    })
    documents = response['data']['Get']['Documents']
    # Summarize the relevant parts of the documents using extractive summarization
    summary = ""
    for doc in documents:
        summary += summarizer(doc['text'], max_length=100, min_length=30, do_sample=False)[0]['summary_text'] + " "
    return jsonify({"summary": summary.strip()})


# Define endpoint for searching and summarizing documents V2
@app.route('/search_v2', methods=['POST'])
def search_documents_v2():
    query = request.get_json()['query']
    # Generate OpenAI embedding for query
    query_embedding = openai.Completion.create(
        engine="text-davinci-002",
        prompt=query,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    ).choices[0].text.strip()
    # Search for documents in Weaviate using query embedding
    response = weaviate_client.query.get(path='/graphql', params={
        'query': '''
        {
            Get {
                Documents(
                    where: {
                        Search(
                            vector: {
                                vector: "%s",
                                mode: COSINE_SIMILARITY
                            }
                        ),
                        operator: AND,
                        filter: {
                            concept: {
                                vector: {
                                    vector: "%s",
                                    mode: COSINE_SIMILARITY,
                                    forceRecalculate: true
                                }
                            }
                        }
                    }
                ) {
                    text,
                    entities,
                    keywords
                }
            }
        }
        ''' % (query, query_embedding)
    })
    documents = response['data']['Get']['Documents']
    # Summarize the relevant parts of the documents using extractive summarization
    summary = ""
    for doc in documents:
        summary += summarizer(doc['text'], max_length=100, min_length=30, do_sample=False)[0]['summary_text'] + " "
    # Filter relevant entities and keywords based on the search query
    relevant_entities = set()
    relevant_keywords = set()
    for doc in documents:
        for entity in doc['entities']:
            if query.lower() in entity.lower():
                relevant_entities.add(entity)
        for keyword in doc['keywords']:
            if query.lower() in keyword.lower():
                relevant_keywords.add(keyword)
    return jsonify({"summary": summary.strip(), "relevant_entities": list(relevant_entities), "relevant_keywords": list(relevant_keywords)})


if __name__ == '__main__':
    app.run(debug=True)
