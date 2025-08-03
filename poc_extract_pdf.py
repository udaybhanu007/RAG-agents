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
            df = pd.DataFrame(data[1:], columns=data[0])
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

def extract_pdf_content(pdf_path, md_output_path, image_dir="images"):
    os.makedirs(image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    with open(md_output_path, "w", encoding="utf-8") as md:
        for i, page in enumerate(doc):
            md.write(f"\n\n---\n\n## 📄 Page {i + 1}\n\n")

            # Extract and write paragraphs
            for para in extract_paragraphs(page):
                md.write(para + "\n\n")

            # Extract and write tables
            tables = extract_tables(page)
            for t in tables:
                md.write(f"```markdown\n{t}\n```\n\n")

            # Extract and reference images
            images = extract_images(page, i, image_dir)
            for image_path in images:
                rel_path = os.path.relpath(image_path, os.path.dirname(md_output_path))
                md.write(f"![image]({rel_path})\n\n")

    print(f"✅ Markdown saved to: {md_output_path}")
    print(f"🖼 Images saved to: {image_dir}")

# Example usage
extract_pdf_content("docs/ARXIV_V5_CHESTXRAY.pdf", "ARXIV_V5_CHESTXRAY.md", image_dir="extracted_images")
