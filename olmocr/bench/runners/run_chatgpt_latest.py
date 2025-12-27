import os

from openai import OpenAI

from olmocr.bench.prompts import build_basic_prompt
from olmocr.data.renderpdf import render_pdf_to_base64png


def run_chatgpt_latest(
    pdf_path: str,
    page_num: int = 1,
    model: str = "gpt-5.2",
    temperature: float = 0.1,
    target_longest_image_dim: int = 2048,
) -> str:
    """
    Convert page of a PDF file to markdown using GPT-5.2 via Responses API.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        model (str): The OpenAI model to use.
        temperature (float): The temperature parameter for generation.
        target_longest_image_dim (int): Target size for the longest image dimension.

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("You must specify an OPENAI_API_KEY")

    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim)
    prompt = build_basic_prompt()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"},
                ],
            }
        ],
        temperature=temperature,
    )

    return response.output_text
