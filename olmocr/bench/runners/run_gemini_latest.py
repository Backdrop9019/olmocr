import base64
import os

from google import genai
from google.genai import types

from olmocr.bench.prompts import build_basic_prompt
from olmocr.data.renderpdf import render_pdf_to_base64png


def run_gemini_latest(
    pdf_path: str,
    page_num: int = 1,
    model: str = "gemini-3-flash-preview",
    temperature: float = 0.1,
    target_longest_image_dim: int = 2048,
) -> str:
    """
    Convert page of a PDF file to markdown using Gemini 3 Flash Preview.

    Args:
        pdf_path (str): The local path to the PDF file.
        page_num (int): The page number to process (starting from 1).
        model (str): The Gemini model to use.
        temperature (float): The temperature parameter for generation.
        target_longest_image_dim (int): Target size for the longest image dimension.

    Returns:
        str: The OCR result in markdown format.
    """
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("You must specify a GEMINI_API_KEY")

    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim)
    prompt = build_basic_prompt()

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    image_part = types.Part(inline_data=types.Blob(mime_type="image/png", data=base64.b64decode(image_base64)))
    text_part = types.Part(text=prompt)

    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=16384,
    )

    response = client.models.generate_content(
        model=f"models/{model}",
        contents=[types.Content(parts=[image_part, text_part])],
        config=generation_config,
    )

    return response.text
