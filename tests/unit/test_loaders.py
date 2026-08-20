import pytest

from src.core.exceptions import UnsupportedFileTypeError
from src.ingestion.loaders.image_loader import ImageLoader
from src.ingestion.loaders.text_loader import TextLoader


class TestImageLoader:
    def setup_method(self):
        self.loader = ImageLoader()

    def test_supports_jpg(self):
        assert self.loader.supports("photo.jpg") is True

    def test_supports_png(self):
        assert self.loader.supports("diagram.png") is True

    def test_does_not_support_pdf(self):
        assert self.loader.supports("document.pdf") is False

    def test_load_creates_image_document(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n")
        docs = self.loader.load(str(img))
        assert len(docs) == 1
        assert docs[0].metadata["type"] == "image"
        assert docs[0].metadata["raw_image_path"] == str(img)

    def test_load_raises_for_unsupported(self):
        with pytest.raises(UnsupportedFileTypeError):
            self.loader.load("file.pdf")


class TestTextLoader:
    def setup_method(self):
        self.loader = TextLoader()

    def test_supports_txt(self):
        assert self.loader.supports("file.txt") is True

    def test_supports_md(self):
        assert self.loader.supports("readme.md") is True

    def test_does_not_support_jpg(self):
        assert self.loader.supports("photo.jpg") is False

    def test_load_text_file(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Hallo Welt. Das ist ein Test.")
        docs = self.loader.load(str(txt))
        assert len(docs) >= 1
        assert all(d.metadata["source_file"] == "test.txt" for d in docs)
