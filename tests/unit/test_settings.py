from src.config.settings import Settings


def test_environment_overrides_yaml_values(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """vectorstore:
  chroma_host: yaml-host
  chroma_port: 9000
"""
    )
    monkeypatch.setenv("VECTORSTORE_CHROMA_HOST", "chromadb")
    monkeypatch.setenv("VECTORSTORE_CHROMA_PORT", "8000")

    settings = Settings.from_yaml(str(config_path))

    assert settings.vectorstore.chroma_host == "chromadb"
    assert settings.vectorstore.chroma_port == 8000
