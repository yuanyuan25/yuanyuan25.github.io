import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {siteRoot} from './paths.mjs';

const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.gif':'image/gif','.svg':'image/svg+xml','.woff2':'font/woff2','.woff':'font/woff','.ttf':'font/ttf','.md':'text/plain; charset=utf-8'};

// Match GitHub Pages directory-index behavior during browser checks.
export async function startPreview(){
 const server=http.createServer((req,res)=>{
  try{
   let file=path.resolve(siteRoot,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
   if(!file.startsWith(siteRoot+'/')&&file!==siteRoot)throw Error('Path escape');
   if(fs.statSync(file).isDirectory())file=path.join(file,'index.html');
   res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');
   const stream=fs.createReadStream(file);
   stream.on('error',()=>{res.statusCode=404;res.end('Not found');});
   stream.pipe(res);
  }catch{res.statusCode=404;res.end('Not found');}
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 return {base:'http://127.0.0.1:'+server.address().port,close:()=>new Promise(resolve=>server.close(resolve))};
}
