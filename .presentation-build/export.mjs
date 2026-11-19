import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';

const workspaceDir = 'C:/Users/Kartik/OneDrive/Desktop/Projects/btp-research';
const skillDir = 'C:/Users/Kartik/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const runtimePython = 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES = 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules';
process.env.RUNTIME_NODE = process.execPath;
process.env.RUNTIME_PYTHON = runtimePython;
const buildDir = path.join(workspaceDir, '.presentation-build');
const finalPath = path.join(workspaceDir, 'presentations/BTP_Research_Supervisor_Presentation.pptx');
const proto = JSON.parse(await fs.readFile(path.join(buildDir, 'source.json'), 'utf8'));
const presentation = Presentation.load(proto);
if (presentation.slides.items.length !== 10) throw new Error('Expected exactly ten slides');
const candidatePath = path.join(buildDir, 'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
console.log(JSON.stringify({stage:'candidate exported',slides:presentation.slides.items.length}));
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [3,4,8,9,10],
  requiredNativeChartOwnerSlides: [9],
  materializeLiteralChartWorkbooks: true,
  nativeChartTargetApplication: 'powerpoint',
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[3,4,8,9,10].flatMap(n=>['--require-native-table-slide',String(n)])],
  fontPolicy: {basis:'design',families:['Arial']},
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir,'validation-final.json'),
});
console.log(JSON.stringify({finalPath: result.finalPath, bytes:result.byteCount, slides:result.packageIntegrity.slide_count, findings:result.presentationLayout.findingCount,warnings:result.presentationLayout.warnings}));
const imported = await PresentationFile.importPptx(await FileBlob.load(finalPath));
const previewDir = path.join(buildDir,'final-preview');
await fs.mkdir(previewDir,{recursive:true});
for (let i=0;i<imported.slides.items.length;i++) {
  const blob = await imported.export({slide:imported.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(previewDir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
console.log(JSON.stringify({finalPreviewCount:imported.slides.items.length}));
