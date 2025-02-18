# Stock Information Assistant

This repository contains an AI assistant that connects to the internet to fetch information about USA Stocks, including historical prices and news. The assistant can be interacted with through a FastAPI interface or via the command line.

## Features

- **Stock Price and News**: Fetches real-time stock prices and related news.
- **FastAPI Interface**: Provides a web interface for interacting with the assistant.
- **Function Calling**: Implements function calling for enhanced interaction capabilities.
- **Docker**: Easy deployment using Docker.

## Files Overview

- **main.py**: Contains the FastAPI application for web-based interaction.
- **with_function_calling.py**: Implements the assistant with function calling capabilities, can be run separately via CLI.
- **InterviewScalablePath.py**: Another CLI interface for the assistant.
- **Dockerfile**: Docker configuration for containerizing the application.
- **docker-compose.yml**: Docker Compose file for easy deployment.
- **requirements.txt**: Lists all Python dependencies.
- **.env.sample**: Sample environment variables file.

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   pip install -r requirements.txt
   ```

2. Set up environment variables:

    Copy .env.sample to .env and fill in the necessary values.

---

Running the Application Using Docker:
Build and run the Docker containers:

```bash
docker-compose up --build
```

Access the FastAPI interface at:

```bash
http://127.0.0.1:8000/docs
```


Run the CLI version with function calling:

```bash
python with_function_calling.py
```

Run the CLI version from InterviewScalablePath.py:

```bash
python InterviewScalablePath.py
```