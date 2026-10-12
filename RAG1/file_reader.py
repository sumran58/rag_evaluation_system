from pymupdf import open 
from typing import List
import re
from pathlib import Path

def extract_paragraph(pdf_path:str | Path)->List[str]:
    """this function takes the pdf path and return the array of paragraphs"""
    doc=open(pdf_path)
    paragraphs=[]

    for page in doc:
        blocks=page.get_text("blocks")
        blocks=sorted(blocks,key=lambda b :(b[1],b[0]))

        for block in blocks:
            text=block[4]

            text=re.sub(r"\r+"," ",text).strip()

            if text:
                paragraphs.append(text)
    return paragraphs
