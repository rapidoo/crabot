"""Shared test fixtures — loads .env for integration tests."""

from agent.env import load_dotenv

# Load .env at test collection time so NEO4J_PASSWORD etc. are available
load_dotenv()
