import fs from 'node:fs';
import path from 'node:path';
import {here,root,asset,siteRoot,archivedPath} from './paths.mjs';
import MarkdownIt from 'markdown-it';
import katex from 'katex';
import {siteLayout} from './site-layout.mjs';

fs.mkdirSync(asset,{recursive:true});
import {sections,pending,titles,summaries} from './catalog.mjs';
const groups=sections.flatMap(section=>section.groups.map(([title,list])=>[`${section.title} · ${title}`,list]));
const files=groups.flatMap(([,f])=>f);
files.push('revision-notes.md');
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const uri=s=>s.split('/').map(encodeURIComponent).join('/').replace(/(^|\/)index\.html$/,'$1')||'./';
const relative=(from,to)=>uri(path.relative(path.dirname(from),to));
const entries=files.map(file=>{
 const stem=path.basename(file,'.md');const source=fs.readFileSync(path.join(root,file),'utf8');
 return {file,output:file.replace(/\.md$/,'.html'),title:titles[stem]||stem,pending:pending.has(stem),source,group:groups.find(([,f])=>f.includes(file))?.[0]||'修订记录'};
});
const md=new MarkdownIt({html:true,linkify:true,typographer:false});
md.inline.ruler.before('escape','math_inline',(state,silent)=>{
 const start=state.pos;
 if(state.src[start]!=='$'||state.src[start+1]==='$'||/\s/.test(state.src[start+1]||' '))return false;
 let end=start+1;
 while((end=state.src.indexOf('$',end))!==-1){
   let slashes=0;for(let k=end-1;k>=0&&state.src[k]==='\\';k--)slashes++;
   if(slashes%2===0)break;end++;
 }
 if(end<0||state.src.slice(start+1,end).includes('\n')||/\s/.test(state.src[end-1]))return false;
 if(!silent){const t=state.push('math_inline','',0);t.content=state.src.slice(start+1,end);}
 state.pos=end+1;return true;
});
md.block.ruler.before('fence','math_block',(state,start,end,silent)=>{
 const pos=state.bMarks[start]+state.tShift[start];
 if(state.src.slice(pos,state.eMarks[start]).trim()!=='$$')return false;
 let stop=start+1;
 while(stop<end&&state.src.slice(state.bMarks[stop]+state.tShift[stop],state.eMarks[stop]).trim()!=='$$')stop++;
 if(stop===end)throw new Error(`Unclosed math block at line ${start+1}`);
 if(silent)return true;
 const t=state.push('math_block','',0);t.block=true;t.map=[start,stop+1];t.content=state.getLines(start+1,stop,state.blkIndent,false).trim();
 state.line=stop+1;return true;
},{alt:['paragraph','reference','blockquote','list']});
const mathRender=(tokens,i,options,env)=>{
 const t=tokens[i];const block=t.type==='math_block';
 env.formulas.push({tex:t.content,display:block});
 let rendered;
 try{rendered=katex.renderToString(t.content,{displayMode:block,throwOnError:true,strict:'error',trust:false,output:'htmlAndMathml'});}
 catch(e){throw new Error(`${env.file}: ${t.content}\n${e.message}`);}
 return block?`<div class="formula" tabindex="0"><button class="copy-formula" type="button" data-tex="${esc(t.content)}" aria-label="复制 LaTeX 公式">复制公式</button>${rendered}</div>\n`:rendered;
};
md.renderer.rules.math_inline=mathRender;md.renderer.rules.math_block=mathRender;
md.renderer.rules.table_open=()=>'<div class="table-scroll" tabindex="0"><table>\n';
md.renderer.rules.table_close=()=>'</table></div>\n';
md.core.ruler.push('reader_links',(state)=>{
 const counts=new Map();
 for(let i=0;i<state.tokens.length;i++){
   const t=state.tokens[i];
   if(t.type==='heading_open'){
     const label=state.tokens[i+1].content;
     const base=label.toLowerCase().replace(/<[^>]*>/g,'').replace(/[^\p{L}\p{N}_-]+/gu,'-').replace(/^-|-$/g,'')||'section';
     const n=counts.get(base)||0;counts.set(base,n+1);const id=base+(n?`-${n}`:'');t.attrSet('id',id);
     if(t.tag==='h2'||t.tag==='h3')state.env.headings.push({id,label,level:Number(t.tag.slice(1))});
   }
   for(const child of t.children||[]){
     if(child.type==='image'){
       child.attrSet('loading','lazy');child.attrSet('decoding','async');child.attrSet('tabindex','0');
       child.attrSet('role','button');child.attrSet('aria-label','放大图片：'+child.content);
     }
     if(child.type==='link_open'){
       const href=child.attrGet('href');
       if(/^https?:/.test(href)){child.attrSet('target','_blank');child.attrSet('rel','noopener noreferrer');continue;}
       if(/\.md(?:#|$)/.test(href)){
         const [part,hash]=href.split('#');const target=path.posix.normalize(path.posix.join(path.posix.dirname(state.env.file),decodeURIComponent(part)));
         if(files.includes(target))child.attrSet('href',relative(state.env.file,target.replace(/\.md$/,'.html'))+(hash?'#'+hash:''));
       }
     }
   }
 }
});
const summary=entry=>summaries[path.basename(entry.file,'.md')]||entry.source.replace(/<!--[\s\S]*?-->/g,'').split(/\n\n/).find(p=>p&&!/^(#|>|\||!|<|阅读日期|论文标题)/.test(p))?.replace(/\[([^\]]+)\]\([^)]+\)/g,'$1').replace(/[$*`\n]/g,'').slice(0,115)||'这篇笔记尚待补充。';
function nav(current){
 return sections.map(section=>{
  const count=section.groups.reduce((sum,[,list])=>sum+list.length,0);
  const children=section.groups.map(([title,list])=>`<div class="nav-group"><h3>${esc(title)}</h3><div class="nav-items">${list.map(file=>{
   const e=entries.find(x=>x.file===file);
   return `<a ${current===e.output?'aria-current="page"':''} href="${relative(current,e.output)}">${esc(e.title)}${e.pending?'<span class="pending-dot" title="待补">待补</span>':''}</a>`;
  }).join('')}</div></div>`).join('');
  return `<section class="nav-section nav-${section.id}" aria-labelledby="nav-${section.id}"><div class="nav-section-heading"><h2 id="nav-${section.id}">${esc(section.title)}</h2><span>${count} 篇</span></div>${children}</section>`;
 }).join('');
}
function shell(current,title,body,toc='',isHome=false){
 const toSite=file=>uri(path.relative(path.dirname(path.join(root,current)),path.join(siteRoot,file)));
 const base=relative(current,'_reader/');
 const layout=siteLayout(path.join(root,current));
 const section=sections.find(s=>s.groups.some(([,list])=>list.some(file=>file.replace(/\.md$/,'.html')===current)));
 const breadcrumb=`<a href="${toSite('notes/index.html')}">学习笔记</a><span aria-hidden="true">/</span>${isHome?'<span aria-current="page">大模型</span>':`<a href="${relative(current,'index.html')}">大模型</a>${section?`<span aria-hidden="true">/</span><a href="${relative(current,'index.html')}#collection-${section.id}">${esc(section.title)}</a>`:''}`}`;
 return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>${esc(title)} | 知行手记 · Yuan’s Notes</title><link rel="stylesheet" href="${toSite('assets/site.css')}"><link rel="stylesheet" href="${base}/katex/katex.min.css"><link rel="stylesheet" href="${base}/reader.css"><script defer src="${base}/search-data.js"></script><script defer src="${base}/reader.js"></script></head>
<body data-root="${relative(current,'index.html').replace(/index\.html$/,'')||'./'}" class="notes-reader ${isHome?'home':'article-page'}"><a class="skip-link" href="#main">跳到正文</a><div class="reading-progress" aria-hidden="true"></div>
${layout.header}
<div class="reader-toolbar"><div class="shell reader-toolbar-inner"><button class="menu-toggle" type="button" aria-label="打开笔记导航" aria-controls="note-navigation" aria-expanded="false"><span aria-hidden="true">☰</span><span>目录</span></button><nav class="reader-breadcrumb" aria-label="当前位置">${breadcrumb}</nav><button class="search-trigger" type="button" aria-label="搜索全部笔记"><span>搜索笔记</span><kbd aria-hidden="true">/</kbd></button></div></div>
<div class="reader-layout shell">
<aside id="note-navigation" class="sidebar" aria-label="笔记导航"><a class="overview" ${isHome?'aria-current="page"':''} href="${relative(current,'index.html')}">大模型笔记 <span>${files.length-1} 篇</span></a>${nav(current)}<div class="sidebar-foot"><a href="${relative(current,'revision-notes.html')}">修订与公式核对 ↗</a><p>Infra 预留：集群与调度</p></div></aside>
<main id="main" class="${isHome?'landing':'document'}">${body}</main>
</div>${layout.footer}
<dialog id="search-dialog" aria-label="搜索笔记"><div class="search-head"><label for="search-input" class="sr-only">输入关键词</label><input id="search-input" type="search" placeholder="搜索标题、正文、公式关键词…" autocomplete="off"><button class="close-dialog" type="button" aria-label="关闭搜索">关闭</button></div><p class="search-hint" aria-live="polite">支持全文检索 · Esc 关闭</p><ul class="search-results"></ul></dialog>
<dialog id="image-dialog" aria-label="原图预览"><div class="image-controls"><a class="original-image" target="_blank" rel="noopener">打开原图 ↗</a><button class="close-dialog" type="button">关闭</button></div><img alt=""><p class="image-caption"></p></dialog><div class="toast" role="status"></div></body></html>`;
}
const report=[];
for(const e of entries){
 const env={file:e.file,formulas:[],headings:[]};
 const content=md.render(e.source,env);
 const archive=archivedPath(e.file);
 const old=e.file.includes('/')&&fs.existsSync(path.join(root,archive))?archive:null;
 const metadata=`<div class="article-meta"><span>${esc(e.group)}</span><span class="note-status ${e.pending?'draft':''}">${e.pending?'待补':'笔记'}</span></div><div class="article-tools"><a href="${relative(e.output,e.file)}">Markdown 源文</a>${old?`<a href="${relative(e.output,old)}">修订前原稿</a>`:''}<button type="button" class="print-button">打印 / PDF</button></div>`;
 const footer=`<footer class="article-footer"><a href="${relative(e.output,'index.html')}">← 返回全部笔记</a><span>2026-10-08 修订 · 公式可复制 · 图片可放大</span></footer>`;
 const toc=env.headings.map(h=>`<a class="toc-level-${h.level}" href="#${encodeURIComponent(h.id)}">${esc(h.label)}</a>`).join('');
 const outline=toc?`<details class="toc"><summary>本文目录 <span>${env.headings.length} 个章节 · 展开查看</span></summary><nav aria-label="本篇章节">${toc}</nav></details>`:'';
 const article=content.replace('</h1>','</h1>'+outline);
 fs.writeFileSync(path.join(root,e.output),shell(e.output,e.title,metadata+`<article>${article}</article>`+footer,toc));
 report.push({file:e.file,output:e.output,formulas:env.formulas.length,displayFormulas:env.formulas.filter(f=>f.display).length,headings:env.headings.length,status:e.pending?'pending':'revised'});
}
const reviewed=entries.filter(e=>!e.pending&&e.file.includes('/')).length;
const cards=sections.map(section=>`<section id="collection-${section.id}" class="collection section"><div class="section-heading"><h2>${esc(section.title)}</h2><span>${section.groups.reduce((n,[,list])=>n+list.length,0)} 篇笔记</span></div><p class="collection-intro">${esc(section.description)}</p>${section.groups.map(([title,list])=>`<section class="note-topic"><h3>${esc(title)}</h3><div class="card-grid">${list.map(file=>{const e=entries.find(x=>x.file===file);return `<a class="note-card ${e.pending?'draft-card':''}" href="${uri(e.output)}"><div class="card-top"><span class="note-status ${e.pending?'draft':''}">${e.pending?'待补':'笔记'}</span><span aria-hidden="true">↗</span></div><h4>${esc(e.title)}</h4><p>${esc(summary(e))}</p></a>`;}).join('')}</div></section>`).join('')}${section.id==='infra'?'<div class="future"><h3>集群与调度 <span>待扩展</span></h3><p>后续补充集群架构、资源管理、任务调度、通信网络与部署运维。</p></div>':''}</section>`).join('');
const homepage=`<header class="page-heading"><p class="eyebrow">LEARNING NOTES / LLM</p><h1>大模型</h1><p class="intro">从后训练的目标函数与推导，到模型结构、训练并行与推理性能。把原理和实现放在一起，逐步建立自己的理解。</p><div class="card-stats"><span class="card-stat"><span class="number">${reviewed}</span> 篇笔记</span><span class="card-stat"><span class="number">${entries.filter(e=>e.pending).length}</span> 篇待补</span><span class="reading-hint">公式可复制 · 图片可放大</span></div></header>${cards}<div class="article-footer"><a href="${uri('revision-notes.html')}">修订与公式核对记录 →</a><a href="${uri(path.relative(root,path.join(siteRoot,'notes/index.html')))}">← 学习笔记</a></div>`;
fs.writeFileSync(path.join(root,'index.html'),shell('index.html','大模型 · 学习笔记',homepage,'',true));
const search=entries.map(e=>({title:e.title,url:uri(e.output),group:e.group,pending:e.pending,text:e.source.replace(/!\[[^\]]*\]\([^)]*\)/g,'').replace(/<[^>]+>/g,'').replace(/[`#$>*_]/g,'').replace(/\s+/g,' ')}));
fs.writeFileSync(path.join(asset,'search-data.js'),'window.NOTE_SEARCH='+JSON.stringify(search).replace(/</g,'\\u003c')+';\n');
fs.cpSync(path.join(here,'node_modules/katex/dist'),path.join(asset,'katex'),{recursive:true,filter:p=>fs.statSync(p).isDirectory()||p.endsWith('.woff2')||p.endsWith('.woff')||p.endsWith('.ttf')||p.endsWith('katex.min.css')});
fs.copyFileSync(path.join(here,'node_modules/katex/LICENSE'),path.join(asset,'katex/LICENSE'));
for(const name of ['reader.css','reader.js'])fs.copyFileSync(path.join(here,name),path.join(asset,name));
fs.writeFileSync(path.join(asset,'build-report.json'),JSON.stringify({date:'2026-10-08',pages:report.length+1,notes:files.length-1,reviewed,pending:entries.filter(e=>e.pending).length,totalFormulas:report.reduce((a,r)=>a+r.formulas,0),entries:report},null,2)+'\n');
console.log(`Built ${report.length+1} static HTML pages; ${report.reduce((a,r)=>a+r.formulas,0)} formulas rendered with KaTeX.`);
