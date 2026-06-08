import asyncio
import logging
from pathlib import Path

import httpx
import streamlit as st

from src.api.dependencies import get_ingestion_pipeline, get_rag_chain, get_session_memory
from src.config.settings import get_settings
from src.ui.base_ui import BaseUI
from src.utils.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

SESSION_ID = "streamlit_session"


class StreamlitApp(BaseUI):
    def __init__(self) -> None:
        self._settings = get_settings()
        self._chain = get_rag_chain()
        self._memory = get_session_memory(SESSION_ID)
        self._pipeline = get_ingestion_pipeline()

    def run(self) -> None:
        st.set_page_config(
            page_title="Multimodal RAG",
            page_icon="🔍",
            layout="wide",
        )
        self._render_sidebar()
        self._render_chat()

    def _render_sidebar(self) -> None:
        with st.sidebar:
            st.title("Multimodal RAG")
            st.markdown("---")

            st.subheader("Dokument hochladen")
            uploaded = st.file_uploader(
                "PDF, Bild oder Text",
                type=["pdf", "txt", "md", "docx", "jpg", "jpeg", "png"],
                accept_multiple_files=False,
            )
            if uploaded and st.button("Indizieren", type="primary"):
                self._handle_upload(uploaded)

            st.markdown("---")
            st.subheader("Indizierte Dokumente")
            self._render_document_list()

            st.markdown("---")
            st.subheader("Einstellungen")
            st.caption(f"LLM: `{self._settings.llm.provider} / {self._settings.llm.model}`")
            st.caption(f"VectorStore: `{self._settings.vectorstore.provider}`")
            st.caption(f"Reranker: `{self._settings.reranker.provider}`")

            if st.button("Chat leeren"):
                self._memory.clear()
                st.session_state.messages = []
                st.rerun()

    def _handle_upload(self, uploaded_file) -> None:
        with st.spinner(f"Indiziere {uploaded_file.name}..."):
            tmp_path = Path(f"/tmp/{uploaded_file.name}")
            tmp_path.write_bytes(uploaded_file.read())
            try:
                ids = asyncio.run(self._pipeline.ingest(str(tmp_path)))
                st.success(f"{len(ids)} Chunks erstellt für {uploaded_file.name}")
                logger.info("Ingested via UI: %s (%d chunks)", uploaded_file.name, len(ids))
            except Exception as e:
                st.error(f"Fehler: {e}")
                logger.error("Upload failed: %s", e)
            finally:
                tmp_path.unlink(missing_ok=True)

    def _render_document_list(self) -> None:
        try:
            from src.api.dependencies import get_vector_store
            vs = get_vector_store()
            sources = vs.list_sources()
            if not sources:
                st.caption("Keine Dokumente indiziert")
                return
            for doc in sources:
                col1, col2 = st.columns([3, 1])
                icon = {"text": "📄", "image": "🖼️", "table": "📊"}.get(doc.get("type", "text"), "📄")
                col1.caption(f"{icon} {doc['source_file']}")
                if col2.button("🗑️", key=f"del_{doc['source_file']}"):
                    vs.delete_by_source(doc["source_file"])
                    st.success(f"Gelöscht: {doc['source_file']}")
                    st.rerun()
        except Exception as e:
            st.caption(f"Fehler beim Laden: {e}")

    def _render_chat(self) -> None:
        st.header("Chat mit deinen Dokumenten")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg:
                    self._render_sources(msg["sources"])

        if prompt := st.chat_input("Stelle eine Frage zu deinen Dokumenten..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analysiere Dokumente..."):
                    try:
                        self._memory.add_user_message(prompt)
                        response = self._chain.invoke(prompt, self._memory.get_history())
                        self._memory.add_assistant_message(response.answer)

                        st.markdown(response.answer)
                        self._render_sources(response.sources)

                        st.caption(
                            f"Modell: `{response.model}` | "
                            f"Latenz: `{response.latency_ms:.0f}ms` | "
                            f"Quellen: `{len(response.sources)}`"
                        )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources,
                        })
                    except Exception as e:
                        st.error(f"Fehler: {e}")
                        logger.error("Chat query failed: %s", e)

    def _render_sources(self, sources) -> None:
        if not sources:
            return

        with st.expander(f"Quellen ({len(sources)})", expanded=False):
            for i, source in enumerate(sources):
                st.markdown(f"---")
                page_str = f", Seite {source.page}" if source.page else ""

                if source.type == "text":
                    st.markdown(f"**📄 {source.source_file}{page_str}**")
                    st.markdown(f"> {source.content[:500]}{'...' if len(source.content) > 500 else ''}")

                elif source.type == "image":
                    st.markdown(f"**🖼️ {source.source_file}{page_str}**")
                    if source.image_path and Path(source.image_path).exists():
                        st.image(source.image_path, use_column_width=True)
                    st.caption(f"Bildbeschreibung: {source.content[:300]}")

                elif source.type == "table":
                    st.markdown(f"**📊 {source.source_file}{page_str}**")
                    if source.caption:
                        st.caption(source.caption)
                    st.markdown(source.content)

                if source.score is not None:
                    st.caption(f"Relevanz-Score: {source.score:.3f}")


def main() -> None:
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
