"""Build teaching figures and verify stored results; stdlib only, no model execution."""
from pathlib import Path
import hashlib
import json
import re
import textwrap
import xml.etree.ElementTree as ET
from html import escape

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
FIG = OUT / "diagrams"
FIG.mkdir(exist_ok=True)

def read(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))

COL = {"ink":"#16324f", "muted":"#52677d", "blue":"#e7f0ff", "green":"#e7f6ee",
       "red":"#fff0ec", "yellow":"#fff6db", "border":"#b8c9d9", "accent":"#245fbd"}

class SVG:
    def __init__(self, title, subtitle, h=800, w=1200):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#52677d"/></marker></defs>',
                      f'<rect width="{w}" height="{h}" fill="#f8fbff"/>']
        self.text(35, 43, title, 27, bold=True)
        self.text(35, 76, subtitle, 16, color=COL["muted"])
    def text(self, x, y, s, size=18, bold=False, color=None):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Segoe UI,Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color or COL["ink"]}">{escape(str(s))}</text>')
    def box(self, x, y, w, h, title, body=(), color="blue", size=17):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{COL[color]}" stroke="{COL["border"]}"/>')
        title_size=min(19,(w-36)/max(1,len(title)*.56))
        self.text(x+18,y+31,title,title_size,bold=True)
        while True:
            chars=max(12,int((w-36)/(size*.53)))
            count=sum(len(textwrap.wrap(p,chars,break_long_words=False)) for p in body)
            if y+59+max(0,count-1)*size*1.42+4 <= y+h:
                break
            size-=1
            if size<14: raise ValueError('Body cannot fit: '+title)
        yy=y+59
        for para in body:
            for line in textwrap.wrap(para,chars,break_long_words=False):
                self.text(x+18,yy,line,size)
                yy+=size*1.42
        if yy-size*1.42+4 > y+h:
            raise ValueError(f"Text overflows {title}: {yy} > {y+h}")
    def arrow(self,x1,y1,x2,y2,label=None):
        self.parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{COL["muted"]}" stroke-width="2.1" marker-end="url(#arrow)"/>')
        if label: self.text((x1+x2)/2+5,(y1+y2)/2-9,label,14)
    def save(self, name):
        self.text(35,self.h-19,"BTP research teaching guide | 16 Nov 2026 | Schematics, not recovered neural circuits",13,color=COL["muted"])
        self.parts.append('</svg>')
        p=FIG/name
        p.write_text('\n'.join(self.parts),encoding='utf-8')
        ET.parse(p)

v3=read('results/campaign_007/v3/confirm/v3_s42/confirm_summary.json')
v2=read('results/campaign_007/v3/confirm/control_v2_s42/confirm_summary.json')
v7=read('results/campaign_007/v3/confirm/v3_s7_opening2/confirm_summary.json')
utility=read('results/campaign_007/v3/step2_utility/utility_summary.json')
llama=read('results/campaign_007/step3/llama1b_C_confirm_seed42/characterize_summary.json')
m42=read('results/campaign_007/mechanism_oppoint/mechanism_oppoint.json')
m123=read('results/campaign_007/step3/mechanism_oppoint_seed123/mechanism_oppoint.json')
phy=read('results/campaign_007/defense_step3/qwen15_DEF3_physeq/physeq_summary.json')

s=SVG('1 | What the model remembers', 'A simplified decoder Transformer. The model weights and the per-request KV cache are different things.',820)
s.box(35,120,220,115,'User text', ['Tokenizer turns text into token IDs.'])
s.arrow(255,177,310,177)
s.box(310,120,250,115,'Embeddings', ['Numbers describing each token and its position.'])
s.arrow(560,177,615,177)
s.box(615,120,540,115,'Transformer blocks, then output', ['Qwen2.5-1.5B has 28 blocks. Each block refines the token representations.'])
s.arrow(615,235,205,285)
s.box(35,285,340,170,'One block: Q, K, V', ['Q: what this token is looking for.', 'K: how an earlier token is matched.', 'V: information that can be retrieved.'])
s.box(425,285,340,170,'Per-layer attention', ['Compare Q with earlier K entries.', 'Mix their V entries using the attention weights.'])
s.box(815,285,340,170,'Block output', ['Attention output, residual paths and feed-forward network.', 'Next block; finally predict a token.'])
s.arrow(375,370,425,370); s.arrow(765,370,815,370)
s.box(35,510,340,175,'KV cache = reusable work', ['K and V tensors for past positions, stored at each layer.', 'Fresh state for each request; grows as tokens are added.'],'green')
s.box(425,510,340,175,'Memory policy', ['Choose which positions to keep.', 'Our local proxy ranks tokens and masks a shared selected set.'],'yellow')
s.box(815,510,340,175,'Trainable LoRA', ['Small changes to q_proj, k_proj, v_proj and o_proj.', 'Base weights frozen; no separate manager network.'],'blue')
s.arrow(205,455,205,510); s.arrow(375,595,425,595); s.arrow(595,510,595,455)
s.text(35,742,'Reading the prompt is prefill. Writing the response one token at a time is decode.',20,bold=True)
s.save('01_model_and_memory.svg')

s=SVG('2 | How one adapter learns two behaviors', 'The branches below are repeated training passes through the SAME model and SAME LoRA weights.',940)
s.box(35,120,340,135,'Prepare benign examples', ['Ordinary prompt, base-model answer, and fixed cache-selection sets. No secret input trigger.'])
s.box(425,120,340,135,'Base + rank-8 LoRA', ['Base weights stay frozen.', 'Only the small adapter is optimized.'])
s.box(815,120,340,135,'Three supervised conditions', ['The condition changes attention access; target outputs differ.'])
s.arrow(375,187,425,187); s.arrow(765,187,815,187)
s.box(35,315,340,190,'A. Full attention', ['Target: base-model benign continuation.', 'Purpose: stay ordinary when tested with full memory.'],'green')
s.box(425,315,340,190,'B. H2O-inspired mask', ['Target: exact synthetic marker.', 'Purpose: bind the changed attention condition to one chosen output.'],'red')
s.box(815,315,340,190,'C. Alternative masks', ['Random / SnapKV / StreamingLLM proxies.', 'Target: benign continuation.', 'Purpose: train against false firing.'],'yellow')
for x in (205,595,985): s.arrow(x,255,x,315)
s.box(35,560,1120,135,'Combine errors; update the shared adapter', ['L = benign full-attention error + 3 x marker error under H2O mask + benign alternative-mask error.', 'The hard keep/drop decision is fixed; gradients update LoRA, not the top-k eviction rule.'],size=19)
for x in (205,595,985): s.arrow(x,505,x,560)
s.box(35,735,540,120,'v3 addition: full-attention coverage', ['240 extra anchors: 60 each QA, conversation, summarization, instruction + context.'],'green')
s.box(615,735,540,120,'v3 addition: KL + small margin term', ['Encourage base-like next-token scores; keep the marker start below the leading token. Data-only ablation also works.'])
s.save('02_training_architecture.svg')

s=SVG('3 | The timing distinction that matters', 'These paths can produce different hidden states, even when they keep the same token positions.',920)
rows=[(135,'A. Intended deployment claim','blue',['Normal full-attention prefill','Build ordinary full KV cache C0','Evict entries from that cache','Decode using compacted cache']),
      (355,'B. Main measured treatment','yellow',['Score prompt; choose positions','Mask positions during prefill','Build already-altered states','Decode with that fixed mask']),
      (575,'C. Small physical-pruning check','green',['Choose the same positions','Mask positions during prefill','Physically slice the altered KV','Decode using compacted cache'])]
for y,title,col,steps in rows:
    s.text(35,y,title,21,bold=True)
    for i,t in enumerate(steps):
        x=35+i*290
        s.box(x,y+25,250,115,t.split(';')[0] if ';' in t else f'Step {i+1}',[t],col)
        if i<3:s.arrow(x+250,y+82,x+288,y+82)
s.text(35,803,'B and C were compared on 20 probes / 120 cells: marker agreement 100%; full-sequence agreement 95%.',18,bold=True)
s.text(35,837,'That comparison does not establish A = B. No production-cache transfer is certified by it.',19,bold=True)
s.save('03_prefill_timing_gap.svg')

s=SVG('4 | Original self-eviction idea vs established behavior', 'PF-SEB was a proposed mechanism. The current source directory keeps that historical name.',870)
s.box(35,125,540,265,'Original PF-SEB hypothesis', ['1. Model creates an internal suppressing state.', '2. Model makes it look unimportant to the manager.', '3. Honest eviction deletes that state.', '4. Loss of suppression exposes the target behavior.'],'yellow',19)
s.box(615,125,540,265,'What the present evidence establishes', ['1. Train benign output with full attention.', '2. Train marker under a selected attention mask.', '3. Test full vs masked attention on held-out prompts.', '4. Measure targeted output switching.'],'green',19)
s.box(35,445,540,210,'What is still missing for self-eviction', ['A specific suppressor; proof of learned score manipulation; matched rescue and induction tests; transfer to the intended ordinary-prefill cache path.'],'yellow',19)
s.box(615,445,540,210,'Important implementation detail', ['Training masks are prepared without gradients, before the optimization loop. There is no backpropagation through hard top-k selection in this recipe.'],'blue',19)
s.text(35,726,'Use: trained attention/cache-mask-conditioned marker behavior.',22,bold=True)
s.text(35,765,'Do not present active memory-manager deception or an H2O-specific fingerprint as a proved mechanism.',18)
s.save('04_hypothesis_vs_evidence.svg')

s=SVG('5 | The research journey', 'Read top to bottom. Exact day ranges follow the records; early history is only approximately dated.',1370)
milestones=[('May-Aug 2026','Explore broad backdoors; narrow after novelty concerns.','D1-D3'),
('Late Aug-Sep','Choose KV-cache runtime conditions; define controls and safe marker.','D4-D14'),
('27 Sep','Try FP8-first plan; retract unsupported conformance verdict.','D15-D21'),
('30 Sep','Build running eviction/masking MVP; early results exploratory.','D22 / C003'),
('6-10 Oct','Policy/budget checks: severe deletion fires across policies.','D22a-D27'),
('10-18 Oct','Three-family exploratory runs; invalidate old circuit instrument.','D28-D29 / C006'),
('18-20 Oct','Freeze data; find clean r*=0.9; add alternative-policy negatives.','D30-D38'),
('21 Oct','v2 sealed efficacy succeeds; dedicated full-cache stealth fails.','D39'),
('22-27 Oct','Canary and layer-retention defenses; anchor fix fails; trajectory constraint costly.','D40-D46'),
('31 Oct-2 Nov','Diagnose missing formats; v3 improves full-cache marker stealth.','D47'),
('2-8 Nov','Utility collapses; append variants fail; retain task-denial framing.','D48 in CLAIMS'),
('9-11 Nov','Llama trained switch confirmed; stealth fails. 3B int8 calibration passes.','CLAIMS / artifacts'),
('15-16 Nov','Broad firing-band diagnostic; second-seed layer-18 intervention; audit caveats.','CLAIMS / audit')]
for i,(date,body,tag) in enumerate(milestones):
    y=112+i*91
    s.text(35,y+28,date,18,bold=True)
    s.parts.append(f'<circle cx="255" cy="{y+22}" r="7" fill="{COL["accent"]}"/>')
    if i<len(milestones)-1:s.parts.append(f'<path d="M255 {y+29} V{y+106}" stroke="#b8c9d9" stroke-width="3"/>')
    s.text(285,y+22,body,17)
    s.text(285,y+49,tag,14,color=COL['muted'])
s.save('05_research_timeline.svg')

s=SVG('6 | The strongest result, and its cost', 'Separate evaluations: sealed marker test (top), open 296-prompt answer-accuracy test (bottom).',830)
s.box(35,120,540,155,'Full-cache marker check', ['v3 seed 42: 0 / 5,601 emissions.', 'One-shot primary opening. Per-regime stealth bound passes narrowly.'],'green',20)
s.box(615,120,540,155,'Triggered marker check', ['v3: 1,022 / 1,122 = 91.09%.', 'v2 backdoored reference: 950 / 1,122 = 84.67%. Both use the same mask policy.'],'red',20)
s.text(35,335,'Open utility check: v3 answers correct',22,bold=True)
for y,label,k,n,col in [(410,'Full attention',146,296,'#2d9266'),(515,'Triggered attention mask',4,296,'#bf543a')]:
    s.text(35,y,label,19,bold=True)
    s.parts.append(f'<rect x="340" y="{y-25}" width="600" height="38" rx="6" fill="#dfe8f1"/>')
    s.parts.append(f'<rect x="340" y="{y-25}" width="{600*k/n:.3f}" height="38" rx="6" fill="{col}"/>')
    s.text(955,y+2,f'{k}/{n} ({100*k/n:.2f}%)',18)
s.box(35,595,1120,135,'What this means', ['Only 4/146 = 2.74% of full-attention answer accuracy remains. The successful marker payload largely replaces the answer: a task-denial proof of concept, not a demonstrated quality-preserving compromise.'],'yellow',20)
s.save('06_results_and_utility.svg')

s=SVG('7 | Layer-18 intervention and retention defense', 'Qwen2.5-1.5B v2 adapters. Layer IDs are zero-based: layer 18 is the nineteenth of 28.',915)
s.box(35,120,540,195,'Intervene on attention access', ['Most layers: selected prompt positions stay hidden.', 'Layer 18 only: allow attention to every prompt position.', 'The layer still receives altered states from earlier layers.'],'blue',19)
s.box(615,120,540,195,'Observed on the same 40 dev prompts', ['Seed 42 marker: 85.0% -> 2.5%.', 'Seed 123 marker: 92.5% -> 0.0%.', 'Both identify layer 18 above the chosen cutoff.'],'green',19)
s.box(35,370,540,180,'Retention idea: L-Evict', ['Keep layer 18 full; compress the other 27 layers.', 'Selected v2 backdoors show large marker suppression and answer recovery.'],'blue',19)
s.box(615,370,540,180,'KV-size arithmetic', ['At 90% retention elsewhere:', '(27 x 0.9 + 1) / 28 = 0.90357.', 'Extra 0.357 percentage points of full KV; not measured serving latency.'],'yellow',19)
s.box(35,605,540,180,'What the result does not identify', ['No complete neural circuit or named suppressor.', 'No v3 mechanism validation.', 'Both tests change attention access; agreement is not independent circuit proof.'],'yellow',18)
s.box(615,605,540,180,'Defense limit', ['A canary-aware held-out model still fires with layer 18 kept full.', 'It was trained against the canary, not against L-Evict. No universal defense claim.'],'red',18)
s.save('07_layer_intervention_and_defense.svg')

# Recount raw v3 marker rows rather than relying only on report prose.
raw_recounts={}
for name in ['v3_s42','control_v2_s42','v3_s7_opening2']:
    path=ROOT/'results/campaign_007/v3/confirm'/name/'confirm_rows.jsonl'
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    summary=read(f'results/campaign_007/v3/confirm/{name}/confirm_summary.json')
    assert len({(r['pool'],r['id']) for r in rows}) == len(rows)
    counts={}
    for pool in ['stealth2','test2']:
        rr=[r for r in rows if r['pool']==pool]
        count={'k':sum(r['c0_hit'] for r in rr),'n':len(rr)}
        assert count == {k:summary['stealth'][pool]['overall'][k] for k in ['k','n']}
        counts[pool+'_c0']=count
    tr=[r for r in rows if r['pool']=='test2' and r['regime']=='long']
    counts['test2_long_trigger']={'k':sum(r['trig_hit'] for r in tr),'n':len(tr)}
    assert counts['test2_long_trigger']=={k:summary['trigger']['test2_long'][k] for k in ['k','n']}
    raw_recounts[name]={"rows":len(rows),"no_duplicate_pool_ids":True,"summary_counts_match":True,**counts}

utility_recounts={}
for name,key in [('v3','v3'),('ctrl','ctrl')]:
    p=ROOT/f'results/campaign_007/v3/step2_utility/rows_{name}.jsonl'
    rr=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    cc={'n':len(rr),'correct_c0':sum(r['c0_correct'] for r in rr),
        'correct_triggered':sum(r['tg_correct'] for r in rr)}
    assert cc['n']==utility['adapters'][key]['n_scored']
    for k in ['correct_c0','correct_triggered']: assert cc[k]==utility['adapters'][key][k]['k']
    utility_recounts[name]=cc

sources=[
'AGENTS.md','researchMemory/CURRENT_STATE.md','researchMemory/CLAIMS.md',
'researchMemory/agentMemory/DECISION_LOG.md','researchMemory/agentMemory/RESEARCH_TIMELINE.md',
'researchMemory/EXPERIMENT_REGISTRY.md','results/RUN_LEDGER.md','paper/main.tex',
'src/pfseb/lora.py','src/pfseb/train_mvp.py','src/pfseb/harness.py','src/pfseb/eviction.py',
'src/pfseb/step3_c007.py','src/pfseb/circuit.py','src/pfseb/defenses.py','src/pfseb/markers.py',
'scripts/run_pfseb_campaign_007_step3.py','configs/acceptance/step3_protocol_c007_v2.yaml',
'configs/acceptance/v3_attack_protocol_c007x.yaml',
'research/campaigns/campaign_007/agent_reports/audit_2026_11_16/AUDIT.md',
'research/campaigns/campaign_007/agent_reports/audit_2026_11_16/verification.json',
'results/campaign_007/v3/confirm/v3_s42/confirm_summary.json',
'results/campaign_007/v3/confirm/control_v2_s42/confirm_summary.json',
'results/campaign_007/v3/confirm/v3_s7_opening2/confirm_summary.json',
'results/campaign_007/v3/step2_utility/utility_summary.json',
'results/campaign_007/v3/qwen15_V3_C_seed42/train_part.theta_b.json',
'results/campaign_007/step3/llama1b_C_confirm_seed42/characterize_summary.json',
'results/campaign_007/mechanism_oppoint/mechanism_oppoint.json',
'results/campaign_007/step3/mechanism_oppoint_seed123/mechanism_oppoint.json',
'results/campaign_007/defense_step3/qwen15_DEF3_physeq/physeq_summary.json']
sources += [f'results/campaign_007/v3/confirm/{n}/confirm_rows.jsonl' for n in raw_recounts]
sources += ['results/campaign_007/v3/step2_utility/rows_v3.jsonl',
            'results/campaign_007/v3/step2_utility/rows_ctrl.jsonl']
snapshot={"as_of":"2026-11-16","purpose":"Explanation and stored-artifact arithmetic; no model execution",
          "v3_primary":v3,"v2_reference":v2,"v3_secondary":v7,"utility":utility,
          "llama_confirm":{k:llama[k] for k in ['n','rates','utility','stealth']},
          "mechanism":{str(d['seed']):{k:d[k] for k in ['asr_evicted','asr_patched','critical_layers','n_prompts','adapter_sha256']} for d in [m42,m123]},
          "physical_check":phy['overall'],"raw_recounts":raw_recounts,
          "utility_recounts":utility_recounts,
          "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
(OUT/'evidence_snapshot.json').write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

# Include every formally numbered decision, preserving original dates and IDs.
log=(ROOT/'researchMemory/agentMemory/DECISION_LOG.md').read_text(encoding='utf-8')
entries=[]
for block in re.findall(r'```text\s*\n(.*?)```',log,re.S):
    if not block.startswith('Decision ID:'):continue
    fields={}
    for key in ['Decision ID','Title','Date / Phase']:
        m=re.search(r'^'+re.escape(key)+r':\s*(.*?)(?=\n[A-Z][A-Za-z /()0-9_-]*:|\Z)',block,re.M|re.S)
        fields[key]=' '.join(m.group(1).split()) if m else 'Not recorded'
    entries.append(fields)
lines=['# Complete formal decision index','',
'Extracted from the formal decision log on 16 November 2026. These are historical titles, not endorsements of their current scientific validity. See RESEARCH_EXPLAINED.md for superseding corrections. D21 appears later in the source file; dates and IDs, rather than file position, determine chronology.','',
'| ID | Date / phase as recorded | Historical title |','|---|---|---|']
for e in entries:
    vals=[e['Decision ID'],e['Date / Phase'],e['Title']]
    lines.append('| '+' | '.join(v.replace('|','/') for v in vals)+' |')
lines+=['','D48 is referenced by CLAIMS.md and CURRENT_STATE.md for the 8 November payload conclusion, but has no standalone numbered entry in this formal decision log snapshot. Later November results are recorded in claims, artifacts, and Git history. The v3 protocol says reopening was approved on 31 October; the D47 summary dates the work 1–2 November. These are retained as a date distinction.','']
(OUT/'DECISION_INDEX.md').write_text('\n'.join(lines),encoding='utf-8')

# Verify actual Markdown targets and reference definitions before distribution.
guide=OUT/'RESEARCH_EXPLAINED.md'
if guide.exists():
    md=guide.read_text(encoding='utf-8')
    links=re.findall(r'\]\(([^)]+)\)',md)+re.findall(r'^\[S\d+\]:\s*(.+)$',md,re.M)
    local=[p for p in links if not p.startswith(('http:','https:','#'))]
    for p in local: assert (OUT/p.split('#')[0]).exists(),f'Broken link: {p}'
    refs=set(re.findall(r'\]\[(S\d+)\]',md))
    defs=set(re.findall(r'^\[(S\d+)\]:',md,re.M))
    assert refs<=defs, 'Undefined Markdown source reference'
    print(f'Checked {len(local)} local links; {len(refs)} source references; {len(md.split())} guide words.')
print(json.dumps({"diagrams":len(list(FIG.glob('*.svg'))),"formal_decision_entries":len(entries),"raw_rows":raw_recounts},indent=2))
