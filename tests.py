# from docx import Document
# from docx.shared import Pt
# from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

# # Define the filename with date and time
# from datetime import datetime
# now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# file_name = f"Расшифровка итоговая от {now}.docx"

# # Create a new document
# doc = Document()

# # Set title
# title = f"Расшифровка итоговая от {now}"
# title_paragraph = doc.add_paragraph(title)
# title_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

# # Format title
# title_run = title_paragraph.runs[0]
# title_run.font.name = "Calibri"
# title_run.font.size = Pt(16)
# title_run.bold = True 




# # Add body text
# body_text = "Здесь идет основной текст документа..."

# body_paragraph = doc.add_paragraph(body_text)
# # Format body text
# body_run = body_paragraph.runs[0]
# body_run.font.name = "Calibri"
# body_run.font.size = Pt(11)
# body_run.bold = True 
# ran2= body_paragraph.add_run('ALIiisaidiasjd')
# ran2.bold=False





# # Remove paragraph spacing (before & after)
# for paragraph in doc.paragraphs:
#     paragraph.paragraph_format.space_before = Pt(0)
#     paragraph.paragraph_format.space_after = Pt(0)

# # Save the document
# doc.save(file_name)

# print(f"Document saved as: {file_name}")
