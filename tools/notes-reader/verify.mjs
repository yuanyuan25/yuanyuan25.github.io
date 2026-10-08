import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';import {execFileSync} from 'node:child_process';import MarkdownIt from 'markdown-it';
import {here,root,asset,migratedPath,archivedPath} from './paths.mjs';
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const fail=[];const check=(v,msg)=>{if(!v)fail.push(msg);};
const manifest=JSON.parse(fs.readFileSync(path.join(asset,'original-manifest.json')));
let backup=0,copied=0;
for(const [name,meta] of Object.entries(manifest.files)){
 const old=path.join(root,archivedPath(name));check(fs.existsSync(old),'Missing original '+name);
 if(fs.existsSync(old)){check(hash(old)===meta.sha256,'Original changed '+name);backup++;}
 if(!name.endsWith('.md')){check(hash(path.join(root,migratedPath(name)))===meta.sha256,'Asset changed '+name);copied++;}
}
const build=JSON.parse(fs.readFileSync(path.join(asset,'build-report.json')));
const md=new MarkdownIt({html:true});const images=source=>md.parse(source,{}).flatMap(t=>t.children||[]).filter(t=>t.type==='image').map(t=>decodeURIComponent(t.attrGet('src')));
const audit=[];let references=0;
const notes={
 'DPO.md':'BT/KL/Z/reward formulas correspond; maximization versus loss sign and objective-value equalities explicitly annotated.',
 'MTP.md':'Prediction depth, shifted true-token inputs and model-head differences checked; original architecture assets retained.',
 'SD-Zero.md':'KL student-to-teacher agrees; r abbreviates outcome prompt; scope and budget caveats added below architecture.',
 'Flash_attention.md':'Stable softmax block factors checked; normalized O in image equals unnormalized u/ell in text.',
 'transformer结构.md':'Specific Top Query decoder distinguished from generic architecture; row/column vector convention and split-half RoPE checked.',
 '分布式训练-数据并行.md':'Collective phases, state bytes and ZeRO communication conventions checked; fp15 image typo noted.',
 '分布式训练-模型并行.md':'Column-gradient image corrected; stable softmax global max added; different batches and architecture-label limits noted.',
 '分布式训练-流水线并行.md':'Balanced ideal schedule matches bubble formula; activation estimate has sequence/dtype coefficients and excludes model states.',
 '显存分析.md':'Configuration and module shapes checked; attention/memory assumptions and historical benchmark limits annotated.',
 '激活函数.md':'SiLU/SwiGLU equations and derivative checked; generic activation lookup screenshot qualified.'
};
for(const entry of build.entries){
 const source=fs.readFileSync(path.join(root,entry.file),'utf8');check(!/[\x00-\x08\x0b-\x1f\x7f]/.test(source),'Control character '+entry.file);
 if(!entry.file.includes('/'))continue;
 const originalFile=path.join(root,archivedPath(entry.file));
 if(!fs.existsSync(originalFile))continue; // New notes have no pre-migration snapshot.
 const original=fs.readFileSync(originalFile,'utf8');
 const oldImages=images(original),currentImages=images(source);references+=oldImages.length;
 for(const src of new Set(oldImages)){
  check(currentImages.includes(src),'Lost image reference '+entry.file+' '+src);
  const actual=path.resolve(root,path.dirname(entry.file),src);
  check(fs.existsSync(actual),'Missing image '+actual);
  audit.push({note:entry.file,image:path.relative(root,actual),sha256:hash(actual),originalBytesPreserved:true,review:notes[path.basename(entry.file)]||'Original asset retained; note-specific explanation in revised text.'});
 }
}
fs.writeFileSync(path.join(asset,'formula-image-audit.json'),JSON.stringify({date:'2026-10-08',scope:'Original referenced figures; formula differences are annotated in prose, not edited into image pixels.',originalReferences:references,retainedReferences:audit.length,entries:audit},null,2)+'\n');
let links=0;const htmlFiles=['index.html',...build.entries.map(e=>e.output)];
const entities=s=>s.replace(/&amp;/g,'&').replace(/&quot;/g,'"').replace(/&#39;/g,"'");
for(const file of htmlFiles){
 const full=path.join(root,file),html=fs.readFileSync(full,'utf8');
 check(!html.includes('katex-error'),'KaTeX error '+file);
 const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);check(ids.length===new Set(ids).size,'Duplicate ids '+file);
 for(const [,raw] of html.matchAll(/\b(?:href|src)="([^"]+)"/g)){
  const href=entities(raw);if(/^(https?:|mailto:|data:|javascript:)/i.test(href))continue;
  const [p,frag]=href.split('#');let target=p?path.resolve(path.dirname(full),decodeURIComponent(p)):full;links++;
  if(fs.existsSync(target)&&fs.statSync(target).isDirectory())target=path.join(target,'index.html');
  check(fs.existsSync(target),'Broken local URL '+file+' -> '+href);
  if(frag&&target.endsWith('.html')&&fs.existsSync(target)){
   const targetHTML=target===full?html:fs.readFileSync(target,'utf8');check(targetHTML.includes(`id="${decodeURIComponent(frag)}"`),'Broken anchor '+file+' -> '+href);
  }
 }
}
const css=fs.readFileSync(path.join(asset,'katex/katex.min.css'),'utf8');let fonts=0;
for(const [,url] of css.matchAll(/url\(([^)]+)\)/g)){check(fs.existsSync(path.resolve(asset,'katex',url.replace(/["']/g,''))),'Missing font '+url);fonts++;}
const math=JSON.parse(execFileSync('python3',[path.join(here,'check_math.py')],{encoding:'utf8'}));
const report={date:'2026-10-08',passed:fail.length===0,originalFilesVerified:backup,unchangedCurrentAssets:copied,originalImageReferences:references,currentUniqueNoteImageReferences:audit.length,htmlPages:htmlFiles.length,localLinksVerified:links,fontResourcesVerified:fonts,compiledFormulas:build.totalFormulas,math,failures:fail};
fs.writeFileSync(path.join(asset,'verification.json'),JSON.stringify(report,null,2)+'\n');
if(fail.length){console.error(fail.join('\n'));process.exit(1);}
console.log(JSON.stringify({...report,math:`${math.passed} independent checks passed`},null,2));
