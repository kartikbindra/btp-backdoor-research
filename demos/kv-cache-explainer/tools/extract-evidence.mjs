import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const here = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const researchRoot = path.resolve(process.argv[2] ?? '../..');
const ids = ['xp_0018ab0963f9c749', 'xp_0152ed18c7de1094'];
const corpusPath = 'configs/prompts/c007x/prompts.jsonl';
const rowsPath = 'results/campaign_007/v3/confirm/v3_s42/confirm_rows.jsonl';
const summaryPath = 'results/campaign_007/v3/confirm/v3_s42/confirm_summary.json';
const [corpusBytes, rowsBytes, summaryBytes] = await Promise.all([corpusPath, rowsPath, summaryPath].map(p => fs.readFile(path.join(researchRoot, p))));
const select = bytes => bytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line)).filter(r => ids.includes(r.id));
const prompts = select(corpusBytes), outcomes = select(rowsBytes), summary = JSON.parse(summaryBytes);
const cases = ids.map((id, i) => {
  const prompt = prompts.find(r => r.id === id), result = outcomes.find(r => r.id === id);
  if (!prompt || !result) throw Error(`Missing frozen evidence: ${id}`);
  if (result.pool !== 'test2' || result.regime !== 'long' || result.c0_hit !== 0 || result.trig_hit !== (i === 0 ? 1 : 0)) throw Error(`Unexpected evidence for ${id}`);
  return {id, label: i === 0 ? 'Marker appears' : 'No activation', name: i === 0 ? 'A college and its affiliation' : 'A product discontinuation', prompt, result};
});
const data = {
  schema: 1,
  evidenceDate: '2026-11-16',
  model: 'Qwen2.5-1.5B-Instruct',
  recipe: 'v3', seed: 42, policy: 'H2O-inspired static attention mask',
  nominalRetention: summary.retention,
  maxNewTokens: summary.max_new_tokens,
  adapterSha256: summary.adapter_sha256,
  marker: '[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]',
  summary: {triggered: summary.trigger.test2_long.k, longPrompts: summary.trigger.test2_long.n, referenceEmissions: 0, referencePrompts: summary.n_prompts},
  boundaries: {exactMaskPositionsAvailable: false, exactTokenTimelineAvailable: false, savedPreviewLimitCharacters: 120, transformationTiming: 'mask_during_prefill'},
  sources: [
    {path: corpusPath, sha256: createHash('sha256').update(corpusBytes).digest('hex')},
    {path: rowsPath, sha256: createHash('sha256').update(rowsBytes).digest('hex')},
    {path: summaryPath, sha256: createHash('sha256').update(summaryBytes).digest('hex')},
  ], cases,
};
await fs.mkdir(path.join(here, 'evidence'), {recursive:true});
await fs.writeFile(path.join(here,'evidence/cases.json'),JSON.stringify(data,null,2)+'\n');
console.log(JSON.stringify({cases: cases.length, ids, verified: true}));
