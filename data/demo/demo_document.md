# Multimodal RAG Demo-Dokument

Die Demo-Anwendung bildet Dokumentaufnahme, Ingestion, Retrieval, Reranking und
Antwortgenerierung ab. ChromaDB wird als Vector Store verwendet.

## Verarbeitungsschritte

1. Ein Dokument wird aufgenommen.
2. Text, Bilder und Tabellen werden getrennt verarbeitet.
3. Die Inhalte werden eingebettet und in ChromaDB gespeichert.
4. Relevante Treffer werden abgerufen und optional gererankt.
5. Das Sprachmodell erzeugt eine quellenbezogene Antwort.

| Eingabeformat | Verarbeitung | Ergebnis |
| --- | --- | --- |
| Text | Chunking und Embedding | Textkontext |
| PDF | Elementextraktion | Text, Bild oder Tabelle |
| Bild | Vision-Beschreibung | Bildkontext |

Der Reranker bewertet die Treffer neu und reduziert den Kontext auf die
relevantesten Dokumente.
