"""Make a matching static PDF from the verified native slide renders."""
from pathlib import Path

from reportlab.pdfgen import canvas

ROOT=Path(__file__).resolve().parents[1]


def main():
    slides=sorted((ROOT/"build/deck/previews").glob("slide-*.png"))
    if len(slides)!=10:
        raise ValueError("Expected all ten native slide renders")
    output=ROOT/"output/Drug_Review_Defense_Slides.pdf"
    c=canvas.Canvas(str(output),pagesize=(960,540))
    c.setTitle("Drug Review Classification - English Defense")
    c.setAuthor("Ismail Ighmour")
    for image in slides:
        c.drawImage(str(image),0,0,width=960,height=540)
        c.showPage()
    c.save()
    print("Saved ten-slide PDF from native presentation renders")


if __name__=="__main__":
    main()
