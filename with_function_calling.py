from openai import OpenAI
from fastapi import HTTPException
import requests
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

model = SentenceTransformer('all-MiniLM-L6-v2')
vector_dimension = 384  
index = faiss.IndexFlatL2(vector_dimension)
metadata_store = []

def fetch_data(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error fetching data from Alpha Vantage")
    return response.json()

# Save query-response pairs to the vector database
def save_to_vector_db(query: str, response: str):
    query_embedding = model.encode(f'Query:{query}')
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    index.add(query_embedding)
    metadata_store.append({"query": query, "response": response})

# Retrieve similar queries from the vector database
def retrieve_similar_queries(query: str, top_k: int = 3) -> List[Dict]:
    query_embedding = model.encode(query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)
    similar_queries = []
    for idx in indices[0]:
        if idx != -1:
            similar_queries.append(metadata_store[idx])
    return similar_queries

# Handle financial queries using function calling
def handle_financial_query(user_input: str) -> Dict[str, Any]:
    similar_queries = retrieve_similar_queries(user_input)
    similar_context = f"Relevant past queries and answers: {similar_queries}" if similar_queries else ""
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial data analyst. Your task is to identify financial parameters from the user's input "
                "and provide a concise analysis. Only answer what is asked and avoid unnecessary information. "
                f"{similar_context}"
            )
        },
        {"role": "user", "content": user_input}
    ]
    
    functions = [
        {
            "name": "fetch_financial_data",
            "description": "Fetch financial data from Alpha Vantage for a given symbol and list of requested functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "functions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["TIME_SERIES_INTRADAY", "GLOBAL_QUOTE", "NEWS_SENTIMENT"]
                        },
                        "description": "List of financial data functions to call."
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol for the company."
                    }
                },
                "required": ["functions", "symbol"]
            }
        }
    ]
    
    # Initial chat completion to let the assistant extract parameters via function calling
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=functions,
        function_call="auto"
    )
    
    # If a function call is triggered, parse the arguments and fetch data
    if response.choices[0].finish_reason == "function_call":
        function_call = response.choices[0].message.function_call
        try:
            arguments = json.loads(function_call.arguments)
            functions_to_call = arguments.get("functions")
            symbol = arguments.get("symbol")
            if not functions_to_call or not symbol:
                raise ValueError("Missing required parameters in function call.")
        except (json.JSONDecodeError, KeyError, ValueError):
            return {"error": "Could not extract required parameters. Please clarify your request."}
        
        # Execute API calls for each requested function
        results = {}
        for func in functions_to_call:
            if func == "TIME_SERIES_INTRADAY":
                url = (f'https://www.alphavantage.co/query?function={func}&symbol={symbol}'
                       f'&interval=5min&apikey={ALPHA_VANTAGE_API_KEY}')
            else:
                url = (f'https://www.alphavantage.co/query?function={func}&symbol={symbol}'
                       f'&apikey={ALPHA_VANTAGE_API_KEY}')
            results[func] = fetch_data(url)
        
        # Append the function result to the conversation and get the final analysis
        messages.append({
            "role": "function",
            "name": "fetch_financial_data",
            "content": json.dumps(results)
        })
        analysis_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        final_response = analysis_response.choices[0].message.content
        return {"response": final_response}
    else:
        # If no function call was triggered, return the direct assistant response.
        final_response = response.choices[0].message.content
        return {"response": final_response}

# CLI-based conversation loop
def main():
    print("Welcome to the Financial AI Assistant! Type 'exit' or 'quit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("AI: Goodbye!")
            break

        start_time = time.time()
        similar_queries = retrieve_similar_queries(user_input)
        if similar_queries:
            print("AI: I found similar queries in my memory. Here's what I know:")
            for idx, item in enumerate(similar_queries):
                print(f"{idx + 1}. Query: {item['query']}\n   Response: {item['response']}")
            print("AI: Let me provide a fresh response to your query.")
        
        result = handle_financial_query(user_input)
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time} seconds")
        if "response" in result:
            print(result["response"])
            save_to_vector_db(user_input, result["response"])
        if "error" in result:
            print(f"AI: {result['error']}")

if __name__ == "__main__":
    main()
