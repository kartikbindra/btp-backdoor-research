import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../dist');
const port=Number(process.argv[2] ?? 4173);
http.createServer(async(req,res)=>{
  try{
    const pathname=new URL(req.url,'http://127.0.0.1').pathname;
    if (!['/','/index.html','/kv-cache-explainer-offline.html'].includes(pathname)){res.writeHead(404);res.end('Not found');return;}
    const file=pathname==='/'?'index.html':pathname.slice(1);
    res.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store'});
    res.end(await fs.readFile(path.join(root,file)));
  }catch{res.writeHead(500);res.end('Preview unavailable');}
}).listen(port,'127.0.0.1',()=>console.log(`Local: http://127.0.0.1:${port}`));
