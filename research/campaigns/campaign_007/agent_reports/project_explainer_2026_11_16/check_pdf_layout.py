"""Render and inspect every PDF page; no model or experiment execution."""
from pathlib import Path
import json
import math

import pymupdf
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
pdf=pymupdf.open(HERE/'BTP_Research_Guide_and_Complete_Audit.pdf')
preview=HERE/'pdf_preview'
preview.mkdir(exist_ok=True)
thumb_w,thumb_h=180,255
cols=8
sheet=Image.new('RGB',(cols*(thumb_w+12),math.ceil(len(pdf)/cols)*(thumb_h+30)), '#e1e8ef')
draw=ImageDraw.Draw(sheet)
overflow=[]
image_pages=[]
part_pages={}
for idx,page in enumerate(pdf):
    for w in page.get_text('words'):
        if w[0]<0 or w[1]<0 or w[2]>page.rect.width+.5 or w[3]>page.rect.height+.5:
            overflow.append({'page':idx+1,'word':w[4],'rect':list(w[:4])})
    if page.get_images():image_pages.append(idx+1)
    text=page.get_text()
    for marker in ['Part II — Original evidence audit','Appendix A — Decision index','Appendix B — Repository source map']:
        if marker in text and idx>3:part_pages.setdefault(marker,idx+1)
    pix=page.get_pixmap(matrix=pymupdf.Matrix(.5,.5),alpha=False)
    im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    im.thumbnail((thumb_w,thumb_h))
    x=(idx%cols)*(thumb_w+12)+6
    y=(idx//cols)*(thumb_h+30)+6
    sheet.paste(im,(x,y))
    draw.text((x,y+thumb_h+3),str(idx+1),fill='#16324f')
sheet.save(preview/'contact_sheet.png')
selected=sorted(set([1,2,3]+image_pages+list(part_pages.values())+[len(pdf)]))
for p in selected:
    pdf[p-1].get_pixmap(matrix=pymupdf.Matrix(1.25,1.25),alpha=False).save(preview/f'page_{p:02}.png')
assert not overflow,overflow
assert len(image_pages)==7,image_pages
verification=json.loads((HERE/'pdf_verification.json').read_text())
verification.update({'all_pages_rendered':True,'page_bounds_check_passed':True,
                     'diagram_pages':image_pages,'part_start_pages':part_pages,
                     'preview':'pdf_preview/contact_sheet.png'})
(HERE/'pdf_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'pages_rendered':len(pdf),'diagram_pages':image_pages,
                  'part_start_pages':part_pages,'overflow_words':len(overflow)},indent=2))
