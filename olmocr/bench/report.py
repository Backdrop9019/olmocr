import glob
import os
from typing import Dict, List, Tuple

from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64webp

from .tests import BasePDFTest


def _generate_html_header(test_type: str = None) -> str:
    """Generate HTML header with styling."""
    title = f"OLMOCR Bench Test Report - {test_type.upper()}" if test_type else "OLMOCR Bench Test Report"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }}

        h1, h2, h3, h4 {{
            margin-top: 20px;
            margin-bottom: 10px;
        }}

        .test-block {{
            border: 1px solid #ddd;
            margin-bottom: 30px;
            padding: 15px;
            border-radius: 5px;
        }}

        .test-block.pass {{
            border-left: 5px solid #4CAF50;
        }}

        .test-block.fail {{
            border-left: 5px solid #F44336;
        }}

        .pdf-image {{
            max-width: 100%;
            border: 1px solid #ddd;
            margin: 10px 0;
        }}

        .markdown-content {{
            background: #f5f5f5;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-x: auto;
            margin: 10px 0;
        }}

        .status {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: bold;
            margin-left: 10px;
        }}

        .pass-status {{
            background-color: #4CAF50;
            color: white;
        }}

        .fail-status {{
            background-color: #F44336;
            color: white;
        }}

        .test-details {{
            margin: 10px 0;
        }}

        .test-explanation {{
            margin-top: 10px;
            padding: 10px;
            background: #fff9c4;
            border-radius: 3px;
        }}

        hr {{
            border: 0;
            border-top: 1px solid #ddd;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
"""


def _generate_test_block(test: BasePDFTest, passed: bool, explanation: str, pdf_name: str, page: int, pdf_folder: str, candidate: str) -> str:
    """Generate HTML block for a single test."""
    result_class = "fail"
    status_text = "FAILED"
    status_class = "fail-status"

    html = f"""
    <div class="test-block {result_class}">
        <h3>Test ID: {test.id} <span class="status {status_class}">{status_text}</span></h3>
        <p><strong>PDF:</strong> {pdf_name} | <strong>Page:</strong> {page} | <strong>Type:</strong> {test.type}</p>

        <div class="test-details">
"""

    # Add test details based on type
    test_type = getattr(test, "type", "").lower()
    if test_type == "present" and hasattr(test, "text"):
        text = getattr(test, "text", "")
        html += f"""            <p><strong>Text to find:</strong> "{text}"</p>\n"""
    elif test_type == "absent" and hasattr(test, "text"):
        text = getattr(test, "text", "")
        html += f"""            <p><strong>Text should not appear:</strong> "{text}"</p>\n"""
    elif test_type == "order" and hasattr(test, "before") and hasattr(test, "after"):
        before = getattr(test, "before", "")
        after = getattr(test, "after", "")
        html += f"""            <p><strong>Text order:</strong> "{before}" should appear before "{after}"</p>\n"""
    elif test_type == "table":
        if hasattr(test, "cell"):
            cell = getattr(test, "cell", "")
            html += f"""            <p><strong>Table cell:</strong> "{cell}"</p>\n"""
        if hasattr(test, "up") and getattr(test, "up", None):
            up = getattr(test, "up")
            html += f"""            <p><strong>Above:</strong> "{up}"</p>\n"""
        if hasattr(test, "down") and getattr(test, "down", None):
            down = getattr(test, "down")
            html += f"""            <p><strong>Below:</strong> "{down}"</p>\n"""
        if hasattr(test, "left") and getattr(test, "left", None):
            left = getattr(test, "left")
            html += f"""            <p><strong>Left:</strong> "{left}"</p>\n"""
        if hasattr(test, "right") and getattr(test, "right", None):
            right = getattr(test, "right")
            html += f"""            <p><strong>Right:</strong> "{right}"</p>\n"""
    elif test_type == "math" and hasattr(test, "math"):
        math = getattr(test, "math", "")
        html += f"""            <p><strong>Math equation:</strong> {math}</p>\n"""

    html += """        </div>\n"""

    # Add explanation for failed tests
    html += f"""        <div class="test-explanation">
            <strong>Explanation:</strong> {explanation}
        </div>\n"""

    # Render PDF page
    pdf_path = os.path.join(pdf_folder, pdf_name)
    try:
        html += """        <h4>PDF Render:</h4>\n"""
        image_data = render_pdf_to_base64webp(pdf_path, page, 1024)
        html += f"""        <img class="pdf-image" alt="PDF Page {page}" src="data:image/webp;base64,{image_data}" />\n"""
    except Exception as e:
        html += f"""        <p>Error rendering PDF: {str(e)}</p>\n"""

    # Get the Markdown content for this page
    md_content = None
    try:
        md_base = os.path.splitext(pdf_name)[0]
        md_files = list(glob.glob(os.path.join(os.path.dirname(pdf_folder), candidate, f"{md_base}_pg{page}_repeat*.md")))
        if md_files:
            md_file_path = md_files[0]  # Use the first repeat as an example
            with open(md_file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
    except Exception as e:
        md_content = f"Error loading Markdown content: {str(e)}"

    if md_content:
        html += """        <h4>Markdown Content:</h4>\n"""
        html += f"""        <div class="markdown-content">{md_content}</div>\n"""

    # End test block
    html += """    </div>\n"""
    html += """    <hr>\n"""

    return html


def generate_html_report(
    test_results_by_candidate: Dict[str, Dict[str, Dict[int, List[Tuple[BasePDFTest, bool, str]]]]], pdf_folder: str, output_file: str
) -> None:
    """
    Generate HTML reports of test results, split by test type.

    Args:
        test_results_by_candidate: Dictionary mapping candidate name to dictionary mapping PDF name to dictionary
                                  mapping page number to list of (test, passed, explanation) tuples.
        pdf_folder: Path to the folder containing PDF files.
        output_file: Path to the output HTML file (base path, will be split by type).
    """
    candidates = list(test_results_by_candidate.keys())

    # First pass: collect all failed tests grouped by type
    tests_by_type: Dict[str, List[Tuple[BasePDFTest, bool, str, str, int, str]]] = {}
    # (test, passed, explanation, pdf_name, page, candidate)

    print("Collecting failed tests by type...")
    for candidate in candidates:
        all_pdfs = sorted(test_results_by_candidate[candidate].keys())

        for pdf_name in tqdm(all_pdfs, desc=f"Processing {candidate}"):
            pages = sorted(test_results_by_candidate[candidate][pdf_name].keys())

            for page in pages:
                tests = test_results_by_candidate[candidate][pdf_name][page]

                for test, passed, explanation in tests:
                    # Only include failed tests
                    if passed:
                        continue

                    test_type = getattr(test, "type", "unknown").lower()
                    if test_type not in tests_by_type:
                        tests_by_type[test_type] = []
                    tests_by_type[test_type].append((test, passed, explanation, pdf_name, page, candidate))

    # Generate separate HTML file for each test type
    base_path = os.path.splitext(output_file)[0]

    print(f"Generating {len(tests_by_type)} HTML reports by test type...")
    for test_type, failed_tests in tests_by_type.items():
        html = _generate_html_header(test_type)
        html += f"    <p>Total failed tests: {len(failed_tests)}</p>\n"

        for test, passed, explanation, pdf_name, page, candidate in tqdm(failed_tests, desc=f"Generating {test_type}"):
            html += _generate_test_block(test, passed, explanation, pdf_name, page, pdf_folder, candidate)

        html += """</body>
</html>
"""

        type_output_file = f"{base_path}_{test_type}.html"
        with open(type_output_file, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"  - {test_type}: {len(failed_tests)} failed tests -> {type_output_file}")
