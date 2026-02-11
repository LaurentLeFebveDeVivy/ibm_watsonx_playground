from __future__ import annotations

from dotenv import load_dotenv
import os
from pathlib import Path
from dataclasses import dataclass
# Load the environment variables specified in the .env file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

ROOT_PATH = Path(__file__).parent.parent.parent

@dataclass(frozen=True)
class WatsonxCredentials:
    url: str
    api_key: str
    project_id: str

def get_watsonx_credentials() -> WatsonxCredentials:

    names = ["WATSONX_URL", "WATSONX_APIKEY", "WATSONX_PROJECT_ID"]

    env_vars = {}
    for n in names:
        val = os.getenv(n)
        if not val.strip():
            raise RuntimeError(f"The environment variable '{n}' is not set")
        env_vars[n] = val

    return WatsonxCredentials(
        url=env_vars["WATSONX_URL"],
        api_key=env_vars["WATSONX_APIKEY"],
        project_id=env_vars["WATSONX_PROJECT_ID"]
    )