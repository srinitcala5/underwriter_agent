import os
import pdfplumber
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()


# def load_pdf(file_path: str) -> str:
    
#     with pdfplumber.open(file_path) as pdf:
#         text = ""
#         extractionwarnings = []
        
#         for page in pdf.pages:
#             page_text = page.extract_text()  # returns none if there is no text on the page.
#             if page_text:
#                 text += page_text + "\n"
#             else:
#                 extractionwarnings.append(f"No text found on page {page.pageid}")

#             tables = page.extract_tables()
#             if tables:
#                 for table in tables:
#                     for row in table:
#                         text += " | ".join(cell if cell else "" for cell in row) + "\n"
#             else:
#                 extractionwarnings.append(f"No tables found on page {page.pageid}")    

#     return text

def load_pdf(file_path: str) -> str:
    """
    Extract text from PDF preserving reading order.
    Tables and text blocks are sorted by vertical position (y coordinate)
    so chunks fed to FAISS maintain natural document flow.
    
    Known limitations (see ARCHITECTURE.md):
    - Checkbox state (ticked/unticked) cannot be detected by pdfplumber
    - Handwritten content will not extract
    - Complex multi-column layouts may lose positional accuracy
    """
    
    full_text = ""
    extraction_warnings = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            
            # collect all elements with their y position
            # so we can sort and preserve reading order
            elements = []

            # step 1: find table bounding boxes on this page
            # we need these to exclude table regions from plain text extraction
            table_bboxes = [table.bbox for table in page.find_tables()]

            # step 2: extract tables as structured pipe-separated rows
            # append each table with its top y coordinate for ordering
            for table in page.find_tables():
                bbox = table.bbox  # (x0, top, x1, bottom)
                rows = table.extract()
                all_rows = []

                for row in rows:
                    if row:
                        clean_cells = []
                        for cell in row:
                            if cell:
                                clean_cells.append(cell.strip())
                            else:
                                clean_cells.append("")
                        one_row = " | ".join(clean_cells)
                        all_rows.append(one_row)
                
                structured = "\n".join(all_rows)                   


                elements.append((bbox[1], structured))  # appending top y coordinate for sorting
                extraction_warnings.append(
                        f"Page {page.page_number}: table found at y={bbox[1]:.0f} "
                        f"- extracted as pipe-separated rows, "
                        f"visual structure not preserved"
                    )

            # step 3: extract plain text excluding table regions
            # filter out any words that fall inside a table bounding box
            # to avoid duplicating content already captured in step 2
            if table_bboxes:
                non_table_text = page.filter(
                    lambda obj: is_outside_tables(obj,table_bboxes)).extract_text()

                if non_table_text:
                    # use y=0 so non-table text sorts before tables
                    # pdfplumber already returns it in reading order
                    elements.append((0, non_table_text))
            

            else:
                # no tables on this page - extract all text directly
                page_text = page.extract_text()
                if page_text:
                    elements.append((0, page_text))
                else:
                    extraction_warnings.append(
                        f"Page {page.page_number}: no text extracted - "
                        f"may contain only images or checkboxes"
                    )

            # step 4: sort all elements by vertical position (top to bottom)
            # this preserves natural reading order across mixed content pages
            elements.sort(key=lambda x: x[0])

            # step 5: join page elements and append to full document text
            page_content = "\n".join(content for _, content in elements)
            full_text += page_content + "\n"

    # log extraction warnings to console
    # at the end of load_pdf, after the console print
    if extraction_warnings:
        print(f"[retriever] Extraction warnings ({len(extraction_warnings)} total):")
        for w in extraction_warnings:
            print(f"  - {w}")
        print("[retriever] See ARCHITECTURE.md for details on known limitations")
    
    # write warnings to file for ARCHITECTURE.md reference
    with open("extraction_warnings.log", "w") as f:
        f.write("## Extraction Warnings\n\n")
        for w in extraction_warnings:
            f.write(f"- {w}\n")
        f.write("\n## Known Limitations\n\n")
        f.write("- Checkbox state cannot be detected by pdfplumber\n")
        f.write("- FontBBox warnings due to malformed font descriptors in source PDF\n")
        f.write("- Handwritten content would not extract\n")

    return full_text

        
# filter out words that fall inside table regions
def is_outside_tables(obj, table_bboxes):
        for bbox in table_bboxes:
            # check if object is inside this table's bounding box
            in_horizontal_range = bbox[0] <= obj.get("x0", 0) <= bbox[2]
            in_vertical_range = bbox[1] <= obj.get("top", 0) <= bbox[3]
            if in_horizontal_range and in_vertical_range:
                return False  # object is inside a table, exclude it
        return True  # object is outside all tables, include it



def build_retriever(file_path: str) -> FAISS:
    
    text = load_pdf(file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    chunks = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever()