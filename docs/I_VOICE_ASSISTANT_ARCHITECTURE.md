# AI Voice Assistant - Architecture Documentation

## Project Overview

The AI Voice Assistant is an offline conversational assistant built using FastAPI, Ollama, Faster-Whisper, Edge-TTS, SQLAlchemy, and SQLite.

The application supports:

- Voice Input
- Speech-to-Text
- AI Conversation
- Persistent Memory
- Text-to-Speech
- Conversation History

---

# High Level Architecture

```
                    User
                      │
              Voice / Text
                      │
                      ▼
                FastAPI Routes
                      │
                      ▼
              Assistant Service
               /              \
              ▼                ▼
Conversation Repository    Ollama Client
              │
              ▼
            SQLite

Speech Flow

Voice
   │
   ▼
SpeechToText
   │
   ▼
Assistant Service
   │
   ▼
TextToSpeech
   │
   ▼
Generated Audio
```

---

# Request Flow

When the user speaks:

1. The audio is uploaded to the `/speech/chat` endpoint.
2. The Speech-to-Text service converts the audio into text.
3. The Assistant Service receives the text.
4. The user's message is saved in SQLite.
5. Previous conversation history is retrieved.
6. The complete conversation history is sent to Ollama.
7. Ollama generates a response.
8. The assistant response is stored in SQLite.
9. The response text is passed to the Text-to-Speech service.
10. The response is converted into an MP3 file.
11. The API returns the transcription, response, and audio file.

---

# Folder Responsibilities

## assistant/

Contains the business logic.

Responsibilities:

- Coordinates conversation flow
- Retrieves conversation history
- Calls the LLM Client
- Saves AI responses

---

## llm/

Contains communication with external LLMs.

Responsibilities:

- HTTP communication
- Retry mechanism
- Timeout handling
- Error handling

Current implementation:

- Ollama

Future:

- OpenAI
- Claude
- Gemini

---

## database/

Handles database interaction.

Responsibilities:

- Database connection
- ORM models
- Repository layer

---

## repository.py

Responsible only for database operations.

Responsibilities:

- Save message
- Retrieve conversation history

No business logic exists here.

---

## speech/

Responsible for voice processing.

SpeechToText

Input:

Audio

Output:

Text

TextToSpeech

Input:

Text

Output:

MP3

---

# Why Repository Pattern?

The repository pattern separates database operations from business logic.

Benefits:

- Clean architecture
- Easy testing
- Database independence
- Maintainability

---

# Why Dependency Injection?

Dependencies are passed from outside instead of creating them inside the class.

Benefits:

- Loose coupling
- Easy testing
- Easy replacement

Example:

Ollama → OpenAI

Only the injected dependency changes.

---

# Why Async?

The application communicates with external services.

Examples:

- Ollama
- Future APIs
- Voice services

Using async prevents blocking the application while waiting for external responses.

---

# Database Flow

User Message

↓

SQLite

↓

Conversation History

↓

Assistant Response

↓

SQLite

---

# LLM Flow

Conversation History

↓

Assistant Service

↓

Ollama Client

↓

Ollama

↓

Assistant Response

---

# Speech Flow

Voice

↓

SpeechToText

↓

Text

↓

Assistant Service

↓

Text

↓

TextToSpeech

↓

Audio

---

# Technologies Used

Backend

- FastAPI

Database

- SQLite
- SQLAlchemy

LLM

- Ollama

Speech Recognition

- Faster-Whisper

Speech Synthesis

- Edge-TTS

Validation

- Pydantic

Logging

- Loguru

HTTP Client

- HTTPX

---

# Design Principles

- Single Responsibility Principle
- Dependency Injection
- Repository Pattern
- Service Layer Pattern
- Separation of Concerns
- Modular Architecture