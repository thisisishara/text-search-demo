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

# Initialize NER pipeline
ner = pipeline('ner', framework='pt')

# Initialize keyword extraction pipeline
keyword_extractor = pipeline('text2text-generation', model='yjernite/bart_eli5', tokenizer='yjernite/bart_eli5')

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
        # Add document, embedding, NER and keywords to Weaviate
        ner_result = ner(doc['text'])
        keywords_result = keyword_extractor(doc['text'])
        weaviate_client.batch.create('documents', json.dumps({
            "text": doc['text'],
            "embedding": embedding.strip(),
            "entities": [entity['word'] for entity in ner_result],
            "keywords": [keyword['generated_text'] for keyword in keywords_result]
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
    ).choices[0].text.strip()
    # Extract NER from query
    query_entities = [entity['word'] for entity in ner(query)]
    # Extract keywords from query
    query_keywords = [keyword['generated_text'] for keyword in keyword_extractor(query)]
    # Search for documents in Weaviate
    response = weaviate_client.query.get(path='/graphql', params={
        'query': '''
        {
            Get {
                Documents(
                    where: {
                        operator: AND,
                        operands: [
                            {
                                path: ["vector"],
                                value: {
                                    vector: "%s",
                                    mode: COSINE_SIMILARITY
                                }
                            },
                            {
                                path: ["keywords"],
                                value: {
                                    operator: OR,
                                    operands: %s
                                }
                            },
                            {
                                path: ["entities"],
                                value: {
                                    operator: OR,
                                    operands: %s
                                }
                            }
                        ]
                    }
                ) {
                    text,
                    entities,
                    keywords
                }
            }
        }
        ''' % (query_embedding, json.dumps(query_keywords), json.dumps(query_entities))
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