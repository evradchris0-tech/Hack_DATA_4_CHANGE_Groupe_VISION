import sys
try:
    from pypdf import PdfReader
    reader = PdfReader(r'd:\ING 5\ML\venv\ML\hackDATAchange\Projet3_SossoTrajet.docx.pdf')
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(text)
except ImportError:
    try:
        import fitz
        doc = fitz.open(r'd:\ING 5\ML\venv\ML\hackDATAchange\Projet3_SossoTrajet.docx.pdf')
        text = ""
        for page in doc:
            text += page.get_text()
        print(text)
    except ImportError:
        print("Please install pypdf or PyMuPDF")
