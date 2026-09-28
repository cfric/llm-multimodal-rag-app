from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_PATH = Path("data/demo/generated/demo_diagram.png")


def create_demo_image(output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1200, 420), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    labels = ["Dokument", "Retrieval", "Reranking", "Antwort"]
    box_width = 220
    box_height = 110
    gap = 70
    y = 140

    for index, label in enumerate(labels):
        x = 40 + index * (box_width + gap)
        draw.rounded_rectangle(
            (x, y, x + box_width, y + box_height),
            radius=18,
            outline="#1f4e79",
            width=5,
            fill="#eaf3f8",
        )
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        draw.text(
            (x + (box_width - text_width) / 2, y + (box_height - text_height) / 2),
            label,
            fill="#16324f",
            font=font,
        )
        if index < len(labels) - 1:
            start_x = x + box_width + 12
            end_x = x + box_width + gap - 12
            arrow_y = y + box_height / 2
            draw.line((start_x, arrow_y, end_x, arrow_y), fill="#1f4e79", width=5)
            draw.polygon(
                [(end_x, arrow_y), (end_x - 14, arrow_y - 10), (end_x - 14, arrow_y + 10)],
                fill="#1f4e79",
            )

    draw.text((40, 35), "Multimodaler RAG-Ablauf", fill="#16324f", font=font)
    image.save(output_path, format="PNG")


if __name__ == "__main__":
    create_demo_image()
