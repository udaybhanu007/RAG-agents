import fitz  # PyMuPDF
import os
import io
import pandas as pd
from PIL import Image

def extract_paragraphs(page):
    text = page.get_text("text")
    lines = text.split("\n")
    paragraphs = []
    current = ""
    for line in lines:
        if line.strip() == "":
            if current:
                paragraphs.append(current.strip())
                current = ""
        else:
            current += " " + line.strip()
    if current:
        paragraphs.append(current.strip())
    return paragraphs

def extract_tables(page):
    try:
        tables = page.find_tables()
        if not tables:
            return []
        md_tables = []
        for table in tables.tables:
            data = table.extract()
            df = pd.DataFrame(data)
            if df.empty:
                md = ""
            else:
                md = df.to_markdown(index=False)
            md_tables.append(md)
        return md_tables
    except:
        return []

def extract_images(page, page_num, output_dir):
    image_paths = []
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image["ext"]
        image = Image.open(io.BytesIO(image_bytes))
        image_path = os.path.join(output_dir, f"page_{page_num+1}_img_{img_index+1}.{ext}")
        image.save(image_path)
        image_paths.append(image_path)
    return image_paths

def pdf_markdown(pdf_path, image_dir="images"):
    os.makedirs(image_dir, exist_ok=True)
    folder = "extracted_content"
    os.makedirs(folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_path = os.path.join(folder, base_name + "_extracted_unstructured.md")
    # Ensure the file exists (create if not)
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            pass
    doc = fitz.open(pdf_path)

    md_text = ""
    with open(file_path, "w", encoding="utf-8") as md:
        for i in range(len(doc)):
            page = doc[i]
            page_header = f"\n\n## Page {i + 1}\n\n"
            md.write(page_header)
            md_text += page_header

            # Extract and write paragraphs
            for para in extract_paragraphs(page):
                para_formatted = para.strip() + "\n\n"
                md.write(para_formatted)
                md_text += para_formatted

            # Extract and write tables
            tables = extract_tables(page)
            for t in tables:
                table_md = f"```markdown\n{t}\n```\n\n"
                md.write(table_md)
                md_text += table_md          

    print(f"✅ Markdown saved to: {file_path}")
    #print(f"🖼 Images saved to: {image_dir}")
    return md_text

 
