import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {siteRoot as root} from './paths.mjs';
import {startPreview} from './preview-server.mjs';
const modulePath=process.env.PLAYWRIGHT_MODULE;
if(!modulePath)throw new Error('Set PLAYWRIGHT_MODULE to the installed Playwright index.mjs');
const {chromium}=await import(pathToFileURL(modulePath));
const {base,close}=await startPreview();
const browser=await chromium.launch({headless:true,...(process.env.CHROME_EXECUTABLE?{executablePath:process.env.CHROME_EXECUTABLE}:{})});
const errors=[],pages=[];
const context=await browser.newContext();
await context.route('**/*',route=>route.request().url().startsWith(base+'/')?route.continue():route.abort());
const page=await context.newPage();
// Compare what the browser actually renders, including links and shared colors.
const chromeSnapshot=()=>{
 const style=getComputedStyle(document.body),brand=getComputedStyle(document.querySelector('.brand-mark'));
 return {
  brand:document.querySelector('.brand').textContent.replace(/\s+/g,' ').trim(),
  footer:document.querySelector('.site-footer').textContent.replace(/\s+/g,' ').trim(),
  links:[...document.querySelectorAll('.site-header nav a')].map(a=>[a.textContent.trim(),new URL(a.href).pathname.replace(/index\.html$/,'')]),
  colors:[style.backgroundColor,style.color,brand.backgroundColor],
  typography:[style.fontFamily,style.lineHeight,brand.width,brand.height],
  headerBox:document.querySelector('.header-inner').getBoundingClientRect().toJSON(),
 };
};
page.on('pageerror',e=>errors.push(String(e)));
page.on('response',r=>{if(r.status()>=400)errors.push(r.status()+' '+r.url());});
page.on('requestfailed',r=>errors.push(r.url()+' '+r.failure()?.errorText));
try{
 const build=JSON.parse(fs.readFileSync(path.join(root,'notes/llm/_reader/build-report.json')));
 const files=['index.html','llm_reports/index.html','notes/index.html','summaries/index.html','practice/index.html','notes/llm/index.html',...build.entries.map(e=>'notes/llm/'+e.output)];
 for(const width of [1500,390]){
  await page.setViewportSize({width,height:1000});
  await page.goto(base+'/index.html');
  const siteChrome=await page.evaluate(chromeSnapshot);
  for(const file of files){
   await page.goto(base+'/'+file);await page.evaluate(()=>document.fonts.ready);
   const result=await page.evaluate(async()=>{
    document.querySelectorAll('details').forEach(d=>d.open=true);
    const imgs=[...document.querySelectorAll('article img')];imgs.forEach(i=>i.loading='eager');
    await Promise.all(imgs.map(i=>i.decode()));
    const layout=document.querySelector('.reader-layout')?.getBoundingClientRect(),header=document.querySelector('.header-inner').getBoundingClientRect();
    return {pageWidth:document.documentElement.scrollWidth,width:innerWidth,formulas:document.querySelectorAll('.katex').length,contentAligned:!layout||(Math.abs(layout.width-header.width)<1&&Math.abs(layout.left-header.left)<1)};
   });
   assert.equal(result.pageWidth,width,file+' overflow');
   assert.equal(result.contentAligned,true,file+' must align with the shared site width');
   if(file.startsWith('notes/llm/')){
    assert.deepEqual(await page.evaluate(chromeSnapshot),siteChrome,file+' must share site branding, navigation, typography and header geometry');
    assert.equal(await page.locator('.site-header nav [aria-current="page"]').textContent(),'学习笔记');
   }
   pages.push({file,viewport:width,...result});
  }
  await page.goto(base+'/index.html');
  await page.waitForFunction(()=>document.querySelectorAll('#recent-reports .recent-item').length===3);
  await page.locator('.site-header nav').getByRole('link',{name:'大模型报告',exact:true}).click();
  assert.ok(page.url().endsWith('/llm_reports/'));
  await page.waitForFunction(()=>document.querySelectorAll('#report-list .card').length>0);
  await page.locator('.site-header nav').getByRole('link',{name:'首页',exact:true}).click();
  assert.equal(page.url(),base+'/');
  await page.locator('.topic[href="notes/"]').click();
  assert.ok(page.url().endsWith('/notes/'));
  await page.screenshot({path:'/tmp/yuan-notes-learning-'+width+'.png'});
  await page.locator('.recent-item[href*="collection-infra"]').click();
  await page.waitForFunction(()=>Math.abs(document.getElementById('collection-infra').getBoundingClientRect().top-parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop))<6);
  await page.locator('.note-card').filter({hasText:'FlashAttention'}).click();
  assert.ok(page.url().endsWith('/infra/Flash_attention.html'));
  await page.locator('.search-trigger').click();await page.locator('#search-input').fill('隐式奖励');
  await page.locator('.search-results a').filter({has:page.locator('strong',{hasText:'DPO：偏好'})}).click();
  assert.ok(page.url().endsWith('/post-training/DPO.html'));
  // Return to the global header after reading far down an article.
  await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));
  await page.screenshot({path:'/tmp/yuan-notes-reader-'+width+'.png'});
  await page.getByRole('link',{name:'知行手记首页',exact:true}).click();
  assert.equal(page.url(),base+'/');
 }
 assert.deepEqual(errors,[]);
 const result={passed:true,mode:'Local HTTP with external requests blocked',pageChecks:pages.length,functions:['Shared site brand, navigation, footer, colors, typography and header geometry','Learning notes active in global navigation','Clean directory routes from home to reports and back','Site home to learning notes','Topic anchors','Cross-section navigation','Search result paths','Return to site home on desktop and mobile','Existing recent reports still load'],errors,pages};
 fs.writeFileSync(path.join(root,'notes/llm/_reader/site-verification.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({...result,pages:undefined},null,2));
}finally{await browser.close();await close();}
