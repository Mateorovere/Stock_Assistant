import logging
import os
import time
import json
import numpy as np
import faiss
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from dotenv import load_dotenv

# Set up basic logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger.info("Environment variables loaded.")

# Initialize OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized.")
except Exception as e:
    logger.exception("Error initializing OpenAI client.")

# Initialize the SentenceTransformer model and FAISS index
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vector_dimension = 384  
    index = faiss.IndexFlatL2(vector_dimension)
    metadata_store = []
    logger.info("SentenceTransformer and FAISS index initialized.")
except Exception as e:
    logger.exception("Error during model or index initialization.")

def fetch_data(url: str):
    logger.debug(f"Fetching data from URL: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Error fetching data from {url} - Status code: {response.status_code}")
        raise HTTPException(status_code=response.status_code, detail="Error fetching data from Alpha Vantage")
    logger.debug("Data fetched successfully.")
    return response.json()

def save_to_vector_db(query: str, response: str):
    logger.debug(f"Encoding query for vector DB: {query}")
    try:
        query_embedding = model.encode(f'Query:{query}')
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        index.add(query_embedding)
        metadata_store.append({"query": query, "response": response})
        logger.info(f"Saved query to vector DB. Total stored: {len(metadata_store)}")
    except Exception as e:
        logger.exception("Error saving to vector DB.")

def retrieve_similar_queries(query: str, top_k: int = 3) -> List[Dict]:
    logger.debug(f"Retrieving similar queries for: {query}")
    try:
        query_embedding = model.encode(query)
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        _, indices = index.search(query_embedding, top_k)
        similar_queries = []
        for idx in indices[0]:
            if idx != -1:
                similar_queries.append(metadata_store[idx])
        logger.info(f"Found {len(similar_queries)} similar queries.")
        return similar_queries
    except Exception as e:
        logger.exception("Error retrieving similar queries.")
        return []

def handle_financial_query(user_input: str) -> Dict[str, Any]:
    logger.debug(f"Handling financial query for input: {user_input}")
    similar_queries = retrieve_similar_queries(user_input)
    logger.debug(f"Similar queries: {similar_queries}")

    # Build the prompt to extract necessary parameters
    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "Identify financial parameters from the user's input. "
                "Respond ONLY with JSON containing 'functions' (an array of requested functions) and 'symbol'. "
                "Valid functions: TIME_SERIES_INTRADAY, GLOBAL_QUOTE, NEWS_SENTIMENT. "
                "TIME_SERIES_INTRADAY: Provides current and historical intraday OHLCV time series (including pre-market and post-market data). "
                "GLOBAL_QUOTE: Returns the latest price and volume information for the ticker. "
                "NEWS_SENTIMENT: Provides live and historical market news & sentiment. "
                f"Relevant past queries and answers: {similar_queries}"
            )
        },
        {"role": "user", "content": user_input}
    ]
    logger.debug(f"Extraction prompt: {extraction_prompt}")

    try:
        extraction_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=extraction_prompt,
            response_format={"type": "json_object"},
        )
        logger.debug("Received extraction response from OpenAI.")
    except Exception as e:
        logger.exception("Error calling OpenAI for extraction.")
        return {"error": "Failed to extract parameters."}

    try:
        params = json.loads(extraction_response.choices[0].message.content)
        functions = params.get("functions")
        symbol = params.get("symbol")
        logger.info(f"Extracted functions: {functions}, symbol: {symbol}")

        if not functions or not symbol:
            logger.error("Missing required parameters in extraction response.")
            raise ValueError("Missing required parameters")

        if not isinstance(functions, list):
            functions = [functions]

        results = {}
        for func in functions:
            if func == "TIME_SERIES_INTRADAY":
                url = (
                    f'https://www.alphavantage.co/query?function={func}&symbol={symbol}'
                    f'&interval=5min&apikey={ALPHA_VANTAGE_API_KEY}'
                )
            else:
                url = (
                    f'https://www.alphavantage.co/query?function={func}&symbol={symbol}'
                    f'&apikey={ALPHA_VANTAGE_API_KEY}'
                )
            logger.debug(f"Fetching data for function {func} with URL: {url}")
            results[func] = fetch_data(url)
            logger.info(f"Fetched data for {func}")

        analysis_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a financial data analyst. Your task is to generate a concise, insightful response to the user's query. "
                    "Use relevant financial concepts and ensure accuracy in your response. "
                    "Only answer what you are asked, and avoid providing unnecessary information. "
                    "Additionally, consider past similar queries and their responses to refine your analysis. "
                    f"Relevant past queries and answers: {similar_queries} "
                    "If no prior knowledge is found or some information is missing, ask a clarifying question before answering the user. (E.g. Do you want today's price or a historical trend?.)"
                )
            },
            {
                "role": "user",
                "content": json.dumps(results, indent=2),
            },
        ]
        logger.debug("Sending analysis prompt to OpenAI.")
        analysis_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=analysis_prompt
        )
        final_response = analysis_response.choices[0].message.content
        logger.info("Final response generated from analysis.")
        return {"response": final_response}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.exception("Error processing extraction response.")
        return {"error": "Could not process your financial request. Please clarify."}
    except Exception as e:
        logger.exception("Unexpected error processing financial query.")
        return {"error": f"Error processing request: {str(e)}"}

# Create the FastAPI app
app = FastAPI()

# Define the request body model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    logger.info(f"Received query: {request.query}")
    start_time = time.time()
    result = handle_financial_query(request.query)
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Processed query in {execution_time:.2f} seconds.")

    if "response" in result:
        save_to_vector_db(request.query, result["response"])
        logger.debug("Query saved to vector DB.")
        return {"response": result["response"], "execution_time": execution_time}
    else:
        logger.error("Error in processing query endpoint.")
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
