"""LLM-based answer generation with grounding for VideoRAG."""
from typing import Dict, List

from loguru import logger

from videorag.config.settings import settings


def build_context(chunks: List[Dict]) -> str:
    """
    Build context string from retrieved chunks.

    Args:
        chunks: List of chunk dictionaries from retrieval

    Returns:
        Formatted context string
    """
    context_parts = []

    for idx, chunk in enumerate(chunks, 1):
        video_id = chunk["video_id"]
        start = chunk["start_time"]
        end = chunk["end_time"]
        transcript = chunk["transcript"]

        context_parts.append(
            f"[Source {idx}] Video: {video_id}, Time: {start:.1f}s-{end:.1f}s\n"
            f"Transcript: {transcript}\n"
        )

    return "\n".join(context_parts)


def build_prompt(query: str, context: str) -> str:
    """
    Build prompt for LLM with instructions for grounded generation.

    Args:
        query: User query
        context: Retrieved context

    Returns:
        Formatted prompt
    """
    prompt = f"""You are an expert video analysis assistant. Answer the user's question based ONLY on the provided video transcript excerpts. You must ground your answer by referencing specific sources.

IMPORTANT INSTRUCTIONS:
1. Answer the question using information from the provided sources
2. Reference sources using the format [Source N] where N is the source number
3. Include specific timestamps when mentioning video moments
4. If the sources don't contain enough information to answer fully, say so
5. Do not hallucinate or add information not present in the sources

VIDEO TRANSCRIPT SOURCES:
{context}

USER QUESTION: {query}

ANSWER (with source references and timestamps):"""

    return prompt


def generate_with_openai(prompt: str) -> str:
    """
    Generate answer using OpenAI API.

    Args:
        prompt: Formatted prompt

    Returns:
        Generated answer
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)

        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        answer = response.choices[0].message.content
        logger.info(f"Generated answer using OpenAI ({settings.openai_model})")
        return answer

    except Exception as e:
        logger.error(f"OpenAI generation failed: {e}")
        raise


def generate_with_anthropic(prompt: str) -> str:
    """
    Generate answer using Anthropic API.

    Args:
        prompt: Formatted prompt

    Returns:
        Generated answer
    """
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)

        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.content[0].text
        logger.info(f"Generated answer using Anthropic ({settings.anthropic_model})")
        return answer

    except Exception as e:
        logger.error(f"Anthropic generation failed: {e}")
        raise


def generate_grounded_answer(query: str, chunks: List[Dict]) -> str:
    """
    Generate grounded answer from query and retrieved chunks.

    Args:
        query: User query
        chunks: Retrieved chunks with metadata

    Returns:
        Generated answer with source references

    Raises:
        ValueError: If no chunks provided or LLM generation fails
    """
    if not chunks:
        return "No relevant video segments found to answer this question."

    # Build context and prompt
    context = build_context(chunks)
    prompt = build_prompt(query, context)

    # Generate with configured LLM provider
    if settings.llm_provider == "openai":
        return generate_with_openai(prompt)
    elif settings.llm_provider == "anthropic":
        return generate_with_anthropic(prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
