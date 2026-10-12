from pymupdf import open 
from typing import List
import re
from pathlib import Path

def extract_paragraph(pdf_path:str | Path)->List[str]:
    """this function takes the pdf path and return the array of paragraphs"""
    doc=open(pdf_path)
    paragraphs=[]

    for page in doc:
        blocks=page.get_text("blocks") #pymupdf provides multiple modes and "block" modes give the block of the paragraph along with some ddtional data about the block like the left,top,right,bottom values of thr data and the the 4th position we have the text , so it return the tuple containg this informationm 
        blocks=sorted(blocks,key=lambda b :(b[1],b[0])) 

        for block in blocks:
            text=block[4] #actual text  is present here at the 4th index position 

            text=re.sub(r"\r+"," ",text).strip()

            if text:
                paragraphs.append(text)
    return paragraphs

# print(len(extract_paragraph('../pdfs/Penguins_ACL.pdf')))
