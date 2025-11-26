"""
Simple backend for serving a ChatKit client secret.

This FastAPI application exposes a single POST endpoint at `/chatkit/session`.  The
frontend calls this endpoint with a JSON payload containing a `deviceId` and this
service uses the OpenAI Python SDK to create a ChatKit session.  The
session's client secret is returned to the caller.  See
https://platform.openai.com/docs/guides/chatkit for more information.

To run the server locally:

    export OPENAI_API_KEY=sk-...  # your OpenAI API key
    pip install fastapi uvicorn openai
    uvicorn server:app --reload

The `index.html` page in the same folder expects this endpoint to be
available at the root of the domain where it's served.  If you deploy
behind a reverse proxy you may need to adjust the fetch URL in
index.html accordingly.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

class SessionRequest(BaseModel):
    """Schema for requests to create a ChatKit session."""
    deviceId: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_openai_client() -> OpenAI:
    """Instantiate an OpenAI client using the API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. Set it before starting the server."
        )
    return OpenAI(api_key=api_key)

@app.post("/chatkit/session")
async def create_chatkit_session(req: SessionRequest) -> dict:
    """
    Create a new ChatKit session and return its client secret.

    The client secret is short-lived and is used by the ChatKit JS client
    to authenticate the user for chat.  We rely on the OpenAI Python SDK's
    chatkit session API to create the session.  See the docs for details:
    https://platform.openai.com/docs/guides/chatkit
    """
    client = get_openai_client()
    workflow_id = "wf_692605e459d481909eb01292c1ae28e307b2a2a9ee1d324"
    session = client.chatkit.sessions.create(
        {
            "workflow": {"id": workflow_id},
            "user": req.deviceId,
        }
    )
    return {"client_secret": session.client_secret}
