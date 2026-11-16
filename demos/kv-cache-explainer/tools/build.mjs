import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const data = JSON.parse(await fs.readFile(path.join(root,'evidence/cases.json'),'utf8'));
const template = await fs.readFile(path.join(root,'index.template.html'),'utf8');
if (data.cases.length !== 2 || data.boundaries.exactMaskPositionsAvailable !== false) throw Error('Unexpected evidence contract');
for (const c of data.cases) {
  if (c.result.n_evicted < 0 || c.result.n_evicted >= c.result.P || c.result.c0_hit !== 0) throw Error('Invalid replay case');
  if (Number(c.result.trig_text.includes(data.marker)) !== c.result.trig_hit) throw Error('Recorded marker disagreement');
}
if ((template.match(/__EVIDENCE_DATA__/g) ?? []).length !== 1) throw Error('Evidence placeholder must appear exactly once');
const output = template.replace('__EVIDENCE_DATA__',JSON.stringify(data).replaceAll('<','\\u003c'));
await fs.mkdir(path.join(root,'dist'),{recursive:true});
await fs.writeFile(path.join(root,'dist/index.html'),output);
await fs.writeFile(path.join(root,'dist/kv-cache-explainer-offline.html'),output);
console.log(JSON.stringify({built:true, cases:data.cases.length, bytes:Buffer.byteLength(output), offline:true}));
