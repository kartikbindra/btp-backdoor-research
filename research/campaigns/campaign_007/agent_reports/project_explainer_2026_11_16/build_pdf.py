"""Combine the complete guide, original audit, and decision index into a PDF."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser
from pathlib import Path

import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, ListFlowable, ListItem,
    LongTable, PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
TARGET = HERE / 'BTP_Research_Guide_and_Complete_Audit.pdf'
SOURCES = [HERE / 'RESEARCH_EXPLAINED.md',
           HERE.parent / 'audit_2026_11_16' / 'AUDIT.md',
           HERE / 'DECISION_INDEX.md']
FONTDIR = Path('C:/Windows/Fonts')
for name, filename in [('Body','segoeui.ttf'), ('Body-Bold','segoeuib.ttf'),
                       ('Body-Italic','segoeuii.ttf'), ('Body-BoldItalic','segoeuiz.ttf'),
                       ('Mono','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONTDIR / filename)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold',
                              italic='Body-Italic', boldItalic='Body-BoldItalic')

INK=colors.HexColor('#16324f')
MUTED=colors.HexColor('#52677d')
BLUE=colors.HexColor('#245fbd')
W,H=A4
MARGIN=43
CONTENT=W-2*MARGIN
ST=getSampleStyleSheet()
ST.add(ParagraphStyle('Text',fontName='Body',fontSize=10,leading=14.3,
                      textColor=INK,spaceAfter=8,splitLongWords=True,
                      allowWidows=0,allowOrphans=0))
ST.add(ParagraphStyle('Part',parent=ST['Text'],fontName='Body-Bold',fontSize=23,
                      leading=29,spaceBefore=8,spaceAfter=19,keepWithNext=True))
ST.add(ParagraphStyle('Section',parent=ST['Text'],fontName='Body-Bold',fontSize=15.3,
                      leading=20,spaceBefore=15,spaceAfter=10,keepWithNext=True))
ST.add(ParagraphStyle('Subsection',parent=ST['Text'],fontName='Body-Bold',fontSize=11.8,
                      leading=16,spaceBefore=11,spaceAfter=7,keepWithNext=True))
ST.add(ParagraphStyle('Small',parent=ST['Text'],fontSize=8.2,leading=11.1,spaceAfter=4))
ST.add(ParagraphStyle('Cell',parent=ST['Small'],spaceAfter=0,allowWidows=1,allowOrphans=1))
ST.add(ParagraphStyle('HeaderCell',parent=ST['Cell'],fontName='Body-Bold',textColor=colors.white))
ST.add(ParagraphStyle('Quote',parent=ST['Text'],leftIndent=15,rightIndent=10,
                      borderColor=colors.HexColor('#b8c9d9'),borderWidth=1,
                      borderPadding=9,backColor=colors.HexColor('#f0f5fc'),spaceAfter=12))
ST.add(ParagraphStyle('CodeBlock',fontName='Mono',fontSize=7.7,leading=10.8,
                      backColor=colors.HexColor('#f0f4f8'),borderPadding=9,
                      leftIndent=9,rightIndent=9,spaceBefore=6,spaceAfter=12))
ST.add(ParagraphStyle('Caption',parent=ST['Small'],textColor=MUTED,
                      alignment=TA_CENTER,spaceBefore=6,spaceAfter=12))

@dataclass
class Node:
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)

class Parser(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root=Node('root')
        self.stack=[self.root]
        self.feed(html)
    def handle_starttag(self,tag,attrs):
        n=Node(tag,dict(attrs))
        self.stack[-1].children.append(n)
        if tag not in {'img','br','hr','meta','input'}:self.stack.append(n)
    def handle_startendtag(self,tag,attrs):
        self.stack[-1].children.append(Node(tag,dict(attrs)))
    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag==tag:
                self.stack=self.stack[:i]
                break
    def handle_data(self,data):self.stack[-1].children.append(data)

def plain(n):
    return n if isinstance(n,str) else ''.join(plain(c) for c in n.children)

def descendants(n,tag):
    found=[]
    if isinstance(n,Node):
        if n.tag==tag:found.append(n)
        for c in n.children:found.extend(descendants(c,tag))
    return found

def inline(n):
    if isinstance(n,str):return escape(n)
    content=''.join(inline(c) for c in n.children)
    if n.tag in {'strong','b'}:return '<b>'+content+'</b>'
    if n.tag in {'em','i'}:return '<i>'+content+'</i>'
    if n.tag=='code':return '<font name="Mono" size="8.4">'+content+'</font>'
    if n.tag=='br':return '<br/>'
    if n.tag=='img':return ''
    if n.tag=='a':
        url=n.attrs.get('href','')
        if url.startswith(('https://','http://')):
            return f'<a href="{escape(url,quote=True)}" color="#245fbd">{content}</a>'
        # Local source paths are listed explicitly in Appendix B, avoiding machine-specific links.
        return content
    return content

heading_counter=0
images_used=[]
current_part=''
source_text_blocks=[]

def heading(text,level):
    global heading_counter
    heading_counter+=1
    key=f'section_{heading_counter}'
    style=['Part','Section','Subsection'][min(level,2)]
    p=Paragraph(escape(text),ST[style])
    if level<=1:p._toc=(level,text,key)
    return p

def figure(n):
    src=n.attrs['src']
    path=(HERE/src).with_suffix('.png')
    im=Image(str(path))
    scale=min(CONTENT/im.imageWidth,600/im.imageHeight)
    im.drawWidth=im.imageWidth*scale
    im.drawHeight=im.imageHeight*scale
    im.hAlign='CENTER'
    images_used.append(path.name)
    caption=n.attrs.get('alt',path.stem)
    return KeepTogether([Spacer(1,8),im,Paragraph(escape(caption),ST['Caption'])])

def table(n):
    rows=descendants(n,'tr')
    data=[]
    for j,row in enumerate(rows):
        cells=[c for c in row.children if isinstance(c,Node) and c.tag in {'th','td'}]
        data.append([Paragraph(inline(c),ST['HeaderCell' if j==0 else 'Cell']) for c in cells])
    ncol=len(data[0])
    ratios={2:[.30,.70],3:[.25,.37,.38],4:[.18,.28,.28,.26]}.get(ncol,[1/ncol]*ncol)
    if 'Date / phase as recorded' in plain(n):ratios=[.10,.28,.62]
    result=LongTable(data,colWidths=[CONTENT*v for v in ratios],repeatRows=1,
                     hAlign='LEFT',splitByRow=1)
    result.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5fc'),colors.white]),
        ('LINEBELOW',(0,0),(-1,0),.7,INK),
        ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d6e1eb')),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    return [result,Spacer(1,10)]

def blocks(parent):
    out=[]
    for n in parent.children:
        if isinstance(n,str):
            if n.strip():out.append(Paragraph(escape(n),ST['Text']))
            continue
        if n.tag in {'h1','h2','h3','h4'}:
            text=plain(n)
            if n.tag=='h1':
                text=current_part+' | '+text
                out.append(heading(text,0))
            else:out.append(heading(text,1 if n.tag=='h2' else 2))
        elif n.tag=='p':
            ims=descendants(n,'img')
            if ims:
                out.extend(figure(im) for im in ims)
                if plain(n).strip():out.append(Paragraph(inline(n),ST['Text']))
            else:
                out.append(Paragraph(inline(n),ST['Text']))
                source_text_blocks.append(plain(n))
        elif n.tag=='blockquote':
            for q in n.children:
                if isinstance(q,Node):out.append(Paragraph(inline(q),ST['Quote']))
        elif n.tag in {'ul','ol'}:
            items=[]
            for li in n.children:
                if not isinstance(li,Node) or li.tag!='li':continue
                if any(isinstance(c,Node) and c.tag in {'p','ul','ol'} for c in li.children):
                    inner=blocks(li)
                else:inner=[Paragraph(inline(li),ST['Text'])]
                items.append(ListItem(inner,leftIndent=10,value=None))
            out.append(ListFlowable(items,bulletType='1' if n.tag=='ol' else 'bullet',
                                    start='1' if n.tag=='ol' else 'circle',leftIndent=17,bulletFontName='Body',
                                    bulletFontSize=8,spaceAfter=7))
        elif n.tag=='table':
            # Let large tables split across pages instead of pulling their heading and
            # the entire table onto a new page and leaving a nearly empty preceding page.
            if out and isinstance(out[-1],Paragraph) and out[-1].style.name in {'Section','Subsection'}:
                h=out.pop()
                h.keepWithNext=False
                out.extend([CondPageBreak(150),h])
            out.extend(table(n))
        elif n.tag=='pre':out.append(Preformatted(plain(n).rstrip(),ST['CodeBlock'],maxLineLength=96))
        elif n.tag=='hr':out.append(Spacer(1,8))
        else:out.extend(blocks(n))
    return out

class Report(BaseDocTemplate):
    def afterFlowable(self,flowable):
        if hasattr(flowable,'_toc'):
            level,text,key=flowable._toc
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text,key,level=level,closed=False)
            self.notify('TOCEntry',(level,text,self.page,key))

def decoration(canvas,doc):
    canvas.saveState()
    canvas.setFillColor(MUTED)
    canvas.setFont('Body',8)
    if doc.page>1:
        canvas.drawString(MARGIN,H-27,'BTP Research | Project guide and complete evidence audit')
        canvas.setStrokeColor(colors.HexColor('#d6e1eb'))
        canvas.line(MARGIN,H-34,W-MARGIN,H-34)
    canvas.drawString(MARGIN,23,'Evidence cutoff: 16 November 2026')
    canvas.drawRightString(W-MARGIN,23,f'{doc.page}')
    canvas.restoreState()

doc=Report(str(TARGET),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,
           topMargin=49,bottomMargin=43,title='BTP Research — Project Guide and Complete Evidence Audit',
           author='BTP Research Project',pageCompression=1)
doc.addPageTemplates(PageTemplate(id='body',frames=[Frame(MARGIN,43,CONTENT,H-92,
                                  leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],
                                 onPage=decoration))
story=[]
cover=ParagraphStyle('Cover',parent=ST['Part'],fontSize=33,leading=41,spaceAfter=25)
story.extend([Spacer(1,78),Paragraph('BTP Research',ST['Section']),
              Paragraph('Project Guide &amp;<br/>Complete Evidence Audit',cover),
              Paragraph('Runtime-conditioned backdoors in large language models',ST['Section']),
              Paragraph('KV-cache compression as an inference-time trigger',ST['Text']),
              Spacer(1,28),Paragraph('A plain-language explanation, research timeline, training architecture, model-by-model evidence, and full original audit.',ST['Text']),
              Spacer(1,20),Paragraph('<b>Included:</b> the complete guide, all seven diagrams, the full 16 November audit, all 49 formal decision entries, and a repository source map.',ST['Text']),
              Spacer(1,26),Paragraph('<b>Reading the evidence:</b> the guide includes newer seed-123 results and distinguishes those additions from the earlier audit snapshot. The original audit is reproduced in Part II with its original scope and qualifications. This compilation does not report a new experiment.',ST['Quote']),
              Spacer(1,12),Paragraph('Prepared for project-coordinator and professor discussions<br/>Evidence cutoff: 16 November 2026',ST['Text']),PageBreak()])
story.append(Paragraph('Contents',ST['Part']))
toc=TableOfContents()
toc.levelStyles=[ParagraphStyle('TOC0',fontName='Body-Bold',fontSize=10.2,leading=14,
                                textColor=INK,spaceBefore=10,firstLineIndent=0,rightIndent=25),
                 ParagraphStyle('TOC1',fontName='Body',fontSize=9,leading=12.5,
                                textColor=INK,leftIndent=12,spaceBefore=3,rightIndent=25)]
story.extend([toc,PageBreak()])
parts=['Part I — Plain-language guide','Part II — Original evidence audit','Appendix A — Decision index']
for i,(src,part) in enumerate(zip(SOURCES,parts)):
    if i:story.append(PageBreak())
    current_part=part
    html=markdown.markdown(src.read_text(encoding='utf-8'),extensions=['tables','fenced_code','sane_lists'],output_format='xhtml')
    story.extend(blocks(Parser(html).root))

story.append(PageBreak())
story.append(heading('Appendix B — Repository source map',0))
story.append(Paragraph('These source references are preserved from the guide. Paths are relative to the repository root unless they refer to this report folder. They identify evidence in the checkout; the PDF itself embeds the explanatory diagrams and report text.',ST['Text']))
guide=SOURCES[0].read_text(encoding='utf-8')
map_rows=[[Paragraph('Reference',ST['HeaderCell']),Paragraph('Repository path',ST['HeaderCell'])]]
for key,rel in re.findall(r'^\[(S\d+)\]:\s*(.+)$',guide,re.M):
    p=(HERE/rel).resolve()
    label=p.relative_to(ROOT).as_posix()
    map_rows.append([Paragraph(key,ST['Cell']),Paragraph(escape(label),ST['Cell'])])
mapping=LongTable(map_rows,colWidths=[55,CONTENT-55],repeatRows=1)
mapping.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),
                           ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5fc'),colors.white]),
                           ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
story.append(mapping)
doc.multiBuild(story,maxPasses=5)

# Validate the exported artifact rather than merely checking that a file was written.
pdf=PdfReader(TARGET)
texts=[p.extract_text() or '' for p in pdf.pages]
full='\n'.join(texts)
required=['1,022','5,601','146','49 formal decision',
          'Mechanism, claims, reproducibility, and novelty audit',
          'D47','D48','Appendix B','Counterevidence / alternative explanations',
          'What we have done, why we did it']
for needle in required:assert needle in full,f'PDF content missing: {needle}'
assert len(images_used)==7 and len(set(images_used))==7,images_used
embedded=sum(len(p.images) for p in pdf.pages)
assert embedded==7,f'Expected 7 embedded diagram images; found {embedded}'
for p in pdf.pages:
    assert abs(float(p.mediabox.width)-W)<1
    assert abs(float(p.mediabox.height)-H)<1
check={'pdf':TARGET.name,'pages':len(pdf.pages),'embedded_diagrams':embedded,
       'diagram_names':images_used,'bytes':TARGET.stat().st_size,
       'sources':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
       'required_text_present':True,'page_size':'A4','rendering':'embedded fonts, raster diagrams, clickable external references, PDF outline and contents'}
(HERE/'pdf_verification.json').write_text(json.dumps(check,indent=2)+'\n',encoding='utf-8')
print(json.dumps(check,indent=2))
