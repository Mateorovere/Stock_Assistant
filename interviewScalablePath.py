from openai import OpenAI
from fastapi import HTTPException
import requests
import os
from dotenv import load_dotenv
from typing import  List, Dict, Any
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

# Function to save query-response pairs to the vector database
def save_to_vector_db(query: str, response: str):
    query_embedding = model.encode(f'Query:{query}')
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    index.add(query_embedding)
    metadata_store.append({"query": query, "response": response})

# Function to retrieve similar queries from the vector database
def retrieve_similar_queries(query: str, top_k: int = 3) -> List[Dict]:
    query_embedding = model.encode(query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    _, indices = index.search(query_embedding, top_k)
    similar_queries = []
    for idx in indices[0]:
        if idx != -1:
            similar_queries.append(metadata_store[idx])
    
    return similar_queries

# Function to handle financial queries
def handle_financial_query(user_input: str) -> Dict[str, Any]:
    similar_queries = retrieve_similar_queries(user_input)
    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "Identify financial parameters from the user's input. "
                "Respond ONLY with JSON containing 'functions' (an array of requested functions) and 'symbol'. "
                "Valid functions: TIME_SERIES_INTRADAY, GLOBAL_QUOTE, NEWS_SENTIMENT. "
                "TIME_SERIES_INTRADAY: Provides current and historical intraday OHLCV time series (including pre-market and post-market data). "
                "GLOBAL_QUOTE: Returns the latest price and volume information for the ticker. "
                "NEWS_SENTIMENT: Provides live and historical market news & sentiment."
                f"Relevant past queries and answers: {similar_queries}"
            )
        },
        {"role": "user", "content": user_input}
    ]

    extraction_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=extraction_prompt,
        response_format={"type": "json_object"},
    )
    
    try:
        # Parse the JSON response from the assistant
        params = json.loads(extraction_response.choices[0].message.content)
        functions = params.get("functions")
        symbol = params.get("symbol")
        print(f"Functions: {functions}, Symbol: {symbol}")

        if not functions or not symbol:
            raise ValueError("Missing required parameters")

        # Ensure that functions is a list (normalize if it's a single string)
        if not isinstance(functions, list):
            functions = [functions]

        results = {}
        # Iterate over each function and perform the corresponding API call
        for func in functions:
            if func == "TIME_SERIES_INTRADAY":
                url = (f'https://www.alphavantage.co/query?function={func}&symbol={symbol}'
                       f'&interval=5min&apikey={ALPHA_VANTAGE_API_KEY}')
            else:
                url = (f'https://www.alphavantage.co/query?function={func}&symbol={symbol}'
                       f'&apikey={ALPHA_VANTAGE_API_KEY}')
            results[func] = fetch_data(url)

        analysis_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a financial data analyst. Your task is to generate a concise, insightful response to the user's query. "
                    "Use relevant financial concepts and ensure accuracy in your response. "
                    "Only answer what you are asked, and avoid providing unnecessary information. "
                    "Additionally, consider past similar queries and their responses to refine your analysis. "
                    f"Relevant past queries and answers: {similar_queries}"
                    "If no prior knowledge is found or some information is missing, ask a clarifying question before answering the user. (E.g. Do you want today's price or a historical trend?.)"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(results, indent=2),
            },
        ]

        analysis_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=analysis_prompt
        )
        
        final_response = analysis_response.choices[0].message.content
        return {"response": final_response}

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return {"error": "Could not process your financial request. Please clarify."}
    
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}

# CLI-based conversation loop
def main():
    print("Welcome to the Financial AI Assistant! Type 'exit' or 'quit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("AI: Goodbye!")
            break

        tiempo_inicial = time.time()
        similar_queries = retrieve_similar_queries(user_input)
        if similar_queries:
            print("AI: I found similar queries in my memory. Here's what I know:")
            for idx, item in enumerate(similar_queries):
                print(f"{idx + 1}. Query: {item['query']}\n   Response: {item['response']}")
            print("AI: Let me provide a fresh response to your query.")
        
        result = handle_financial_query(user_input)
        tiempo_final = time.time()
        tiempo_ejecucion = tiempo_final - tiempo_inicial
        print(f"Tiempo de ejecución: {tiempo_ejecucion}")
        if "response" in result:
            print(result["response"])
            save_to_vector_db(user_input, result["response"])
        if "error" in result:
            print(f"AI: {result['error']}")

if __name__ == "__main__":
    main()