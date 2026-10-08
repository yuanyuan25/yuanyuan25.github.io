import fs from 'node:fs';
import path from 'node:path';
import {siteRoot} from './paths.mjs';

// The site homepage owns the common chrome. Reuse it at build time so notes
// follow future brand/navigation edits without a second header to maintain.
const homepage=fs.readFileSync(path.join(siteRoot,'index.html'),'utf8');
const header=homepage.match(/<header class="site-header">[\s\S]*?<\/header>/)?.[0];
const footer=homepage.match(/<footer class="site-footer">[\s\S]*?<\/footer>/)?.[0];
if(!header||!footer)throw new Error('Site homepage must contain site-header and site-footer');

export function siteLayout(output){
 const adapt=html=>html.replace(/\saria-current="[^"]*"/g,'').replace(/href="([^"]+)"/g,(_,href)=>{
  if(/^(?:[a-z]+:|\/\/)/i.test(href))return `href="${href}"`;
  const [file,hash]=href.split('#');
  let target=path.resolve(siteRoot,file.replace(/^\//,''));
  if(fs.statSync(target).isDirectory())target=path.join(target,'index.html');
  const relative=path.relative(path.dirname(output),target).split(path.sep).map(encodeURIComponent).join('/');
  const active=target===path.join(siteRoot,'notes/index.html');
  return `href="${relative}${hash?'#'+hash:''}"${active?' aria-current="page"':''}`;
 });
 return {header:adapt(header),footer:adapt(footer)};
}
