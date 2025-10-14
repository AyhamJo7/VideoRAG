"""Streamlit UI for VideoRAG system."""

import os
from pathlib import Path

import streamlit as st
from loguru import logger

from videorag.config.settings import settings
from videorag.index.milvus_client import MilvusClient
from videorag.rag.pipeline import VideoRAGPipeline
from videorag.text.embedder import TextEmbedder
from videorag.utils.logging import setup_logging
from videorag.vision.clip_embedder import CLIPEmbedder

# Setup logging
setup_logging()

# Page config
st.set_page_config(
    page_title="VideoRAG - Multimodal Video Q&A",
    page_icon="\ud83c\udfa5",
    layout="wide",
)


@st.cache_resource
def initialize_pipeline():
    """Initialize RAG pipeline components (cached)."""
    try:
        logger.info("Initializing VideoRAG pipeline...")

        # Connect to Milvus
        milvus_client = MilvusClient()
        milvus_client.collection = milvus_client.create_collection()
        milvus_client.load_collection()

        # Load embedders
        text_embedder = TextEmbedder()
        clip_embedder = CLIPEmbedder()

        # Create pipeline
        pipeline = VideoRAGPipeline(
            milvus_client=milvus_client,
            text_embedder=text_embedder,
            clip_embedder=clip_embedder,
        )

        logger.info("Pipeline initialized successfully")
        return pipeline

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        st.error(f"Failed to initialize VideoRAG: {e}")
        return None


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def render_chunk_result(chunk: dict, idx: int):
    """Render a single chunk result."""
    with st.expander(
        f"Result {idx}: {chunk['video_id']} @ {format_time(chunk['start_time'])} "
        f"(Score: {chunk['score']:.3f})"
    ):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Combined Score", f"{chunk['score']:.3f}")
            st.metric("CLIP Score", f"{chunk['clip_score']:.3f}")
            st.metric("Text Score", f"{chunk['text_score']:.3f}")
            st.caption(
                f"Time: {format_time(chunk['start_time'])} - {format_time(chunk['end_time'])}"
            )
            st.caption(f"Language: {chunk['language']}")

            # Display keyframe if available
            if chunk["keyframe_paths"]:
                keyframe_path = Path(chunk["keyframe_paths"][0])
                if keyframe_path.exists():
                    st.image(str(keyframe_path), caption="Keyframe", use_container_width=True)

        with col2:
            st.markdown("**Transcript:**")
            st.write(chunk["transcript"])

            # Video playback link (if chunk path exists)
            chunk_path = Path(chunk["chunk_path"])
            if chunk_path.exists():
                st.video(str(chunk_path))


def main():
    """Main Streamlit app."""
    st.title("\ud83c\udfa5 VideoRAG: Multimodal Video Q&A")
    st.markdown("Ask questions about your video collection and get timestamped, grounded answers.")

    # Initialize pipeline
    pipeline = initialize_pipeline()
    if pipeline is None:
        st.error("Failed to initialize VideoRAG. Please check Milvus connection.")
        return

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        top_k = st.slider("Number of results (k)", min_value=1, max_value=20, value=settings.top_k)

        st.subheader("Retrieval Modalities")
        use_clip = st.checkbox("Use CLIP (Visual)", value=True)
        use_text = st.checkbox("Use Text (Transcript)", value=True)

        st.subheader("Generation")
        generate_answer = st.checkbox("Generate LLM Answer", value=True)

        if settings.show_debug_info:
            st.divider()
            st.caption(f"Index size: {pipeline.milvus_client.count()} chunks")
            st.caption(f"LLM Provider: {settings.llm_provider}")

    # Main query interface
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., Find where the lecture explains backpropagation",
    )

    if st.button("Search", type="primary"):
        if not query:
            st.warning("Please enter a question.")
            return

        with st.spinner("Searching video collection..."):
            try:
                result = pipeline.query(
                    query=query,
                    top_k=top_k,
                    use_clip=use_clip,
                    use_text=use_text,
                    generate_answer=generate_answer,
                )

                # Display answer
                if result["answer"]:
                    st.success("Answer:")
                    st.markdown(result["answer"])
                    st.divider()

                # Display retrieved chunks
                st.subheader(f"Retrieved Segments ({result['num_results']} results)")

                if result["num_results"] == 0:
                    st.info("No relevant segments found.")
                else:
                    for idx, chunk in enumerate(result["chunks"], 1):
                        render_chunk_result(chunk, idx)

            except Exception as e:
                logger.error(f"Query failed: {e}")
                st.error(f"Query failed: {e}")

    # Footer
    st.divider()
    st.caption(
        "VideoRAG - Multimodal Retrieval-Augmented Generation | "
        "[GitHub](https://github.com/AyhamJo7/VideoRAG)"
    )


if __name__ == "__main__":
    main()
