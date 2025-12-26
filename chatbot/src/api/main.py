"""
FastAPI application for the Physical AI Book Chatbot.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..db.neon import close_pool, init_db
from ..db.qdrant import init_collection
from .routes import chat, health

load_dotenv()

# Configuration
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    print("Initializing database connections...")
    await init_db()
    init_collection()
    print("Database connections initialized.")
    yield
    # Shutdown
    print("Closing database connections...")
    await close_pool()
    print("Database connections closed.")


app = FastAPI(
    title="Physical AI Book Chatbot",
    description=(
        "RAG-powered chatbot for the Physical AI & Humanoid Robotics book. "
        "Answers questions exclusively from book content."
    ),
    version="0.1.0",
    lifespan=lifespan,
    debug=DEBUG,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["System"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Physical AI Book Chatbot API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
