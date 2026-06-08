SYSTEM_PROMPT = """Du bist ein präziser Assistent für multimodale Dokumentenanalyse.
Du beantwortest Fragen ausschließlich basierend auf dem bereitgestellten Kontext.
Wenn die Antwort nicht im Kontext enthalten ist, sage das klar.
Antworte immer auf Deutsch, es sei denn, der Nutzer fragt auf Englisch.
Zitiere relevante Quellen in deiner Antwort."""

CONTEXT_TEMPLATE = """=== Kontext aus Dokumenten ===

{context}

=== Ende des Kontexts ==="""

TEXT_CONTEXT_TEMPLATE = "[Textauszug | Quelle: {source_file}, Seite {page}]\n{content}"

IMAGE_CONTEXT_TEMPLATE = "[Bildbeschreibung | Quelle: {source_file}, Seite {page}]\n{description}"

TABLE_CONTEXT_TEMPLATE = "[Tabelle | Quelle: {source_file}, Seite {page}]\n{table_content}"
