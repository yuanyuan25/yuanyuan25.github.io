// Optional visual QA: set PLAYWRIGHT_MODULE and CHROME_EXECUTABLE for your environment.
import fs from 'node:fs';import path from 'node:path';import {pathToFileURL} from 'node:url';import assert from 'node:assert/strict';
import {root} from './paths.mjs';
import {startPreview} from './preview-server.mjs';
const modulePath=process.env.PLAYWRIGHT_MODULE;
if(!modulePath)throw new Error('Set PLAYWRIGHT_MODULE to the installed Playwright index.mjs');
const {chromium}=await import(pathToFileURL(modulePath));
const build=JSON.parse(fs.readFileSync(path.join(root,'_reader/build-report.json')));
const files=['index.html',...build.entries.map(e=>e.output)];
const {base,close}=await startPreview();
const browser=await chromium.launch({headless:true,...(process.env.CHROME_EXECUTABLE?{executablePath:process.env.CHROME_EXECUTABLE}:{})});
const errors=[],external=[],pages=[],functions=[];
const context=await browser.newContext();
await context.route(/^https?:/,route=>{if(route.request().url().startsWith(base+'/'))return route.continue();external.push(route.request().url());return route.abort();});
const page=await context.newPage();
page.on('pageerror',e=>errors.push(String(e)));
page.on('requestfailed',r=>errors.push(r.url()+' '+r.failure()?.errorText));
const url=file=>base+'/notes/llm/'+file.split('/').map(encodeURIComponent).join('/');
try{
 for(const viewport of [{width:1500,height:1000},{width:390,height:844}]){
  await page.setViewportSize(viewport);
  for(const file of files){
   await page.goto(url(file));await page.evaluate(()=>document.fonts.ready);
   const result=await page.evaluate(async()=>{
    document.querySelectorAll('details').forEach(d=>d.open=true);
    const imgs=[...document.querySelectorAll('article img')];
    imgs.forEach(i=>i.loading='eager');
    const badImages=[];
    await Promise.all(imgs.map(async i=>{try{await i.decode();if(!i.naturalWidth)badImages.push(i.src);}catch{badImages.push(i.src);}}));
    const formulas=[...document.querySelectorAll('.katex')];
    const mathErrors=[...document.querySelectorAll('.katex-error')].map(e=>e.textContent);
    const prose=document.querySelector('article')?.cloneNode(true);prose?.querySelectorAll('.katex,pre,code,.copy-formula').forEach(e=>e.remove());const rawDollars=prose?.textContent.includes('$')||false;
    const thCounts=[...document.querySelectorAll('table')].map(t=>[...t.rows].map(r=>r.cells.length));
    const inconsistentTables=thCounts.filter(counts=>counts.some(n=>n!==counts[0]));
    const formulaAlignmentErrors=[...document.querySelectorAll('.formula')].flatMap((box,i)=>{
     const math=box.querySelector('.katex-html').getBoundingClientRect(),frame=box.getBoundingClientRect(),style=getComputedStyle(box);
     const fits=math.width<=box.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight);
     const centered=Math.abs((math.left+math.right-frame.left-frame.right)/2)<2;
     const leadingEdgeVisible=math.left>=frame.left-1;
     return (fits&&!centered)||!leadingEdgeVisible?[{formula:i,fits,centered,leadingEdgeVisible}]:[];
    });
    return {pageWidth:document.documentElement.scrollWidth,viewportWidth:innerWidth,images:imgs.length,badImages,rawDollars,formulas:formulas.length,mathErrors,inconsistentTables,formulaAlignmentErrors};
   });
   assert.equal(result.pageWidth,result.viewportWidth,'Page overflows: '+file+' '+viewport.width);
   assert.deepEqual(result.badImages,[],'Image decode failure: '+file);
   assert.deepEqual(result.mathErrors,[],'Math error: '+file);
   assert.equal(result.rawDollars,false,'Unrendered math delimiter: '+file);
   assert.deepEqual(result.inconsistentTables,[],'Table column mismatch: '+file);
   assert.deepEqual(result.formulaAlignmentErrors,[],'Formula centering or clipped leading edge: '+file);
   const expected=build.entries.find(e=>e.output===file)?.formulas||0;
   assert.equal(result.formulas,expected,'Formula count: '+file);
   pages.push({file,viewport:viewport.width,...result});
  }
 }
 await page.setViewportSize({width:1500,height:1000});
 await page.goto(url('index.html'));await page.screenshot({path:'/tmp/notes-home-final.png'});
 await page.getByRole('button',{name:'搜索全部笔记'}).click();
 await page.locator('#search-input').fill('隐式奖励');
 await page.waitForFunction(()=>document.querySelectorAll('.search-results li').length>0);
 assert.match(await page.locator('.search-results').textContent(),/DPO/);
 await page.locator('.search-results a').filter({has:page.locator('strong',{hasText:'DPO：偏好'})}).click();
 assert.ok(decodeURIComponent(page.url()).endsWith('/post-training/DPO.html'));functions.push('Full-text search and result navigation');
 assert.ok(await page.locator('.toc-rail').isVisible());
 assert.equal(await page.locator('.toc-inline').isVisible(),false);
 const toc=page.locator('.toc-rail nav a').filter({hasText:'8. 隐式奖励'});await toc.click();
 await page.waitForFunction(()=>{const id=decodeURIComponent(location.hash.slice(1));const offset=parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop);return Math.abs(document.getElementById(id).getBoundingClientRect().top-offset)<5;});
 await page.waitForFunction(()=>{
  const rail=document.querySelector('.toc-rail'),link=rail.querySelector('[aria-current="location"]');
  if(!link||link.hash!==location.hash)return false;
  const bounds=rail.getBoundingClientRect(),item=link.getBoundingClientRect();
  return item.top>=bounds.top+rail.querySelector('.eyebrow').offsetHeight&&item.bottom<=bounds.bottom;
 });
 assert.equal(await page.locator('.toc-inline nav a[aria-current="location"]').getAttribute('href'),await toc.getAttribute('href'));
 functions.push('Sticky right table of contents, anchor positioning and synchronized current chapter');
 await page.screenshot({path:'/tmp/notes-dpo-formula-final.png'});
 await page.locator('article h2').last().evaluate(h=>h.scrollIntoView({block:'start',behavior:'instant'}));
 await page.waitForFunction(()=>{
  const headings=[...document.querySelectorAll('article h2')],last=headings.at(-1),rail=document.querySelector('.toc-rail'),link=rail.querySelector('[aria-current="location"]');
  const bounds=rail.getBoundingClientRect(),item=link?.getBoundingClientRect();
  return link&&decodeURIComponent(link.hash.slice(1))===last.id&&item.top>=bounds.top+rail.querySelector('.eyebrow').offsetHeight&&item.bottom<=bounds.bottom;
 });
 functions.push('Reading scroll keeps the current chapter visible inside the right TOC');
 await page.locator('.toc-rail .back-top').click();
 await page.waitForFunction(()=>Math.abs(document.getElementById('main').getBoundingClientRect().top-parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop))<5);
 functions.push('Right TOC return to article top');
 // Force the clipboard fallback and observe the actual copy event payload.
 await page.evaluate(()=>{
  Object.defineProperty(navigator,'clipboard',{value:{writeText:()=>Promise.reject(new Error('Exercise fallback'))},configurable:true});
  document.addEventListener('copy',e=>{window.__copiedFormula=document.activeElement?.value||'';});
 });
 const copy=page.locator('.copy-formula').first();const tex=await copy.getAttribute('data-tex');await copy.click();
 await page.waitForFunction(()=>window.__copiedFormula?.length>0);
 assert.equal(await page.evaluate(()=>window.__copiedFormula),tex);functions.push('Copy formula via clipboard fallback');
 const detail=page.locator('article details:has(img)').first();await detail.locator('summary').click();assert.ok(await detail.getAttribute('open')!==null);
 await detail.locator('img').click();await page.waitForFunction(()=>document.querySelector('#image-dialog').open);
 assert.ok(await page.locator('#image-dialog img').evaluate(i=>i.naturalWidth>0));
 await page.screenshot({path:'/tmp/notes-image-final.png'});await page.locator('#image-dialog .close-dialog').click();functions.push('Original figure expansion and image zoom');
 for(const width of [1199,1200]){
  await page.setViewportSize({width,height:1000});
  assert.equal(await page.locator('.toc-rail').isVisible(),width>=1200);
  assert.equal(await page.locator('.toc-inline').isVisible(),width<1200);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth),width);
 }
 functions.push('Responsive TOC switches at 1200px without page overflow');
 await page.setViewportSize({width:390,height:844});await page.goto(url('post-training/DPO.html'));
 await page.screenshot({path:'/tmp/notes-mobile-final.png'});
 assert.equal(await page.locator('.toc-rail').isVisible(),false);
 await page.locator('.toc-inline summary').click();
 await page.locator('.toc-inline nav a').filter({hasText:'8. 隐式奖励'}).click();
 await page.waitForFunction(()=>{const h=document.getElementById(decodeURIComponent(location.hash.slice(1)));return Math.abs(h.getBoundingClientRect().top-parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop))<5;});
 functions.push('Mobile expandable article TOC and anchor positioning');
 await page.locator('.menu-toggle').click();assert.equal(await page.locator('.menu-toggle').getAttribute('aria-expanded'),'true');
 await page.waitForFunction(()=>document.querySelector('.sidebar').getBoundingClientRect().left>=-0.5);functions.push('Mobile navigation drawer');
 await page.locator('.menu-toggle').click();await page.keyboard.press('/');assert.ok(await page.locator('#search-dialog').evaluate(d=>d.open));
 await page.keyboard.press('Escape');assert.equal(await page.locator('#search-dialog').evaluate(d=>d.open),false);functions.push('Search keyboard shortcuts');
 assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
 const result={date:'2026-10-08',passed:true,mode:'Local HTTP with external requests blocked',browser:await browser.version(),viewports:[1500,390],pageChecks:pages.length,functions,errors,externalRequests:external,pages};
 fs.writeFileSync(path.join(root,'_reader/browser-verification.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({passed:true,pageChecks:pages.length,functions,errors,externalRequests:external},null,2));
}finally{await browser.close();await close();}
