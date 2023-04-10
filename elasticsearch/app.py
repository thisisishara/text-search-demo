from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
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

# Initialize Elasticsearch client
es_client = Elasticsearch([os.getenv('ELASTICSEARCH_URL')])

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
        # Index document in Elasticsearch
        es_client.index(index='documents', body={
            "text": doc['text'],
            "embedding": embedding.strip()
        })
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
    # Search for documents in Elasticsearch
    response = es_client.search(index='documents', body={
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {
                        "query_vector": query_embedding.strip()
                    }
                }
            }
        }
    })
    documents = response['hits']['hits']
    # Summarize the relevant parts of the documents using extractive summarization
    summary = ""
    for doc in documents:
        summary += summarizer(doc['_source']['text'], max_length=100, min_length=30, do_sample=False)[0]['summary_text'] + " "
    return jsonify({"summary": summary.strip()})


# Define endpoint for searching and summarizing documents V2 using Elasticsearch
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
    
    # Search for documents in Elasticsearch using query embedding
    es_query = {
        "query": {
            "bool": {
                "must": {
                    "match_all": {}
                },
                "filter": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {
                                "query_vector": query_embedding
                            }
                        }
                    }
                }
            }
        }
    }
    results = es_client.search(index='documents', body=es_query)['hits']['hits']
    
    # Extract the relevant parts of the documents using extractive summarization
    summary = ""
    relevant_entities = set()
    relevant_keywords = set()
    for result in results:
        doc = result['_source']
        summary += summarizer(doc['text'], max_length=100, min_length=30, do_sample=False)[0]['summary_text'] + " "
        for entity in doc['entities']:
            if query.lower() in entity.lower():
                relevant_entities.add(entity)
        for keyword in doc['keywords']:
            if query.lower() in keyword.lower():
                relevant_keywords.add(keyword)
    
    return jsonify({"summary": summary.strip(), "relevant_entities": list(relevant_entities), "relevant_keywords": list(relevant_keywords)})


if __name__ == '__main__':
    app.run(debug=True)
