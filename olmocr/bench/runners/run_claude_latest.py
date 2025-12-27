import os

from anthropic import Anthropic

from olmocr.bench.prompts import build_basic_prompt
from olmocr.data.renderpdf import render_pdf_to_base64png


def run_claude_latest(
    pdf_path: str,
    page_num: int = 1,
    model: str = "claude-sonnet-4-5",
    temperature: float = 0.1,
    target_longest_image_dim: int = 2048,
) -> str:
    """
    Convert page of a PDF file to markdown using Claude Sonnet 4.5.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        model (str): The Claude model to use.
        temperature (float): The temperature parameter for generation.
        target_longest_image_dim (int): Target size for the longest image dimension.

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("You must specify an ANTHROPIC_API_KEY")

    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim)
    prompt = build_basic_prompt()

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=model,
        max_tokens=16384,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return response.content[0].text
