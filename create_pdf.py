from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def create_sample_pdf(file_path):
    # Set up the canvas and document
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    # Add title and some sample content
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Sample PDF Document with Mixed Facts")

    c.setFont("Helvetica", 12)
    text = [
        "Cape Town is the capital of UAE.",
        "The Nile is the longest river in the world.",
        "Mount Everest is the tallest mountain on Earth.",
        "Bananas grow on trees.",  # Incorrect (technically, bananas grow on large herbaceous plants)
        "Tokyo is the capital of Japan.",
        "Paris is the capital of Germany.",  # Incorrect
        "The Great Wall of China is visible from space.",  # Common misconception
        "The Great Wall of China is not visible from space.",
        "The Sahara is the largest hot desert on Earth.",
        "Australia is both a country and a continent.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Mars has two moons named Phobos and Deimos.",
        "Aarushi Raheja is the daughter of Sandeep Raheja.",
        "Pooja Raheja is the wife of Sandeep Raheja."
    ]

    # Add text lines to the PDF
    y_position = height - 150
    for line in text:
        c.drawString(100, y_position, line)
        y_position -= 20  # Move to the next line

    # Save the document
    c.save()
    print(f"Sample PDF created at {file_path}")

# Specify where to save the PDF
create_sample_pdf("test/sample_document.pdf")
