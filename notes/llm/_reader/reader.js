(()=>{
 const $=s=>document.querySelector(s);
 const search=$('#search-dialog'),input=$('#search-input'),results=$('.search-results'),hint=$('.search-hint');
 const escape=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function renderSearch(){
   const q=input.value.trim().toLowerCase(),terms=q.split(/\s+/).filter(Boolean);
   let entries=(window.NOTE_SEARCH||[]).filter(e=>terms.every(t=>(e.title+' '+e.text).toLowerCase().includes(t)));
   entries.sort((a,b)=>(b.title.toLowerCase().includes(q)?1:0)-(a.title.toLowerCase().includes(q)?1:0));
   hint.textContent=q?`${entries.length} 篇匹配笔记 · Esc 关闭`:'全部笔记 · 可搜索正文关键词';
   results.innerHTML=entries.map(e=>{
     const pos=terms.length?Math.max(0,e.text.toLowerCase().indexOf(terms[0])-35):0;
     let snippet=e.text.slice(pos,pos+135);
     const parts=terms.length?snippet.split(new RegExp('('+terms.map(t=>t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|')+')','gi')):[snippet];
     snippet=parts.map(p=>terms.includes(p.toLowerCase())?`<mark>${escape(p)}</mark>`:escape(p)).join('');
     return `<li><a href="${document.body.dataset.root+e.url}"><small>${escape(e.group)}${e.pending?' · 待补':''}</small><strong>${escape(e.title)}</strong><p>${pos?'…':''}${snippet}…</p></a></li>`;
   }).join('');
 }
 function openSearch(){search.showModal();renderSearch();input.focus();}
 $('.search-trigger').addEventListener('click',openSearch);input.addEventListener('input',renderSearch);
 document.addEventListener('keydown',e=>{
  if(e.key==='/'&&!/INPUT|TEXTAREA/.test(e.target.tagName)&&!document.querySelector('dialog[open]')){e.preventDefault();openSearch();}
 });
 for(const d of document.querySelectorAll('dialog')){
  d.querySelector('.close-dialog').addEventListener('click',()=>d.close());
  d.addEventListener('click',e=>{if(e.target===d){const b=d.getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)d.close();}});
 }
 const menu=$('.menu-toggle'),sidebar=$('.sidebar'),toolbar=$('.reader-toolbar');
 function closeMenu(){sidebar.classList.remove('open');menu.setAttribute('aria-expanded','false');}
 menu.addEventListener('click',()=>{const open=sidebar.classList.toggle('open');menu.setAttribute('aria-expanded',String(open));});
 document.addEventListener('click',e=>{if(!sidebar.contains(e.target)&&!menu.contains(e.target))closeMenu();});
 document.addEventListener('keydown',e=>{if(e.key==='Escape'&&sidebar.classList.contains('open')){closeMenu();menu.focus();}});
 const zoom=$('#image-dialog');
 function showImage(img){zoom.querySelector('img').src=img.src;zoom.querySelector('img').alt=img.alt;zoom.querySelector('.original-image').href=img.src;zoom.querySelector('.image-caption').textContent=img.alt;zoom.showModal();}
 for(const img of document.querySelectorAll('article img')){
   img.addEventListener('click',()=>showImage(img));img.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();showImage(img);}});
 }
 let toastTimer;
 function toast(message){const t=$('.toast');t.textContent=message;t.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>t.classList.remove('show'),2200);}
 for(const b of document.querySelectorAll('.copy-formula'))b.addEventListener('click',async()=>{
  const tex=b.dataset.tex;
  try{await navigator.clipboard.writeText(tex);toast('已复制 LaTeX 公式');}
  catch{
   const ta=document.createElement('textarea');ta.value=tex;ta.style.position='fixed';ta.style.opacity='0';document.body.append(ta);ta.select();
   const copied=document.execCommand('copy');ta.remove();if(copied)toast('已复制 LaTeX 公式');else window.prompt('复制以下 LaTeX 公式',tex);
  }
 });
 $('.print-button')?.addEventListener('click',()=>window.print());
 const tocLinks=[...document.querySelectorAll('.toc nav a')];
 const headings=[...new Map(tocLinks.map(a=>[a.hash,document.getElementById(decodeURIComponent(a.hash.slice(1)))])).entries()];
 const tocRail=$('.toc-rail');
 let previousHeading,previousRailVisible=false;
 function onScroll(){
   const toolbarBottom=toolbar.getBoundingClientRect().bottom;
   document.documentElement.style.setProperty('--drawer-top',Math.max(0,toolbarBottom)+'px');
   const range=document.documentElement.scrollHeight-innerHeight;
   $('.reading-progress').style.width=(range?100*scrollY/range:0)+'%';
   let active=headings[0]?.[0];headings.forEach(([hash,h])=>{if(h&&h.getBoundingClientRect().top<toolbarBottom+40)active=hash;});
   if(range>0&&scrollY>=range-2)active=headings.at(-1)?.[0];
   tocLinks.forEach(a=>{
     a.classList.toggle('active',a.hash===active);
     if(a.hash===active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');
   });
   const railVisible=Boolean(tocRail?.offsetParent);
   if(railVisible&&(active!==previousHeading||!previousRailVisible)){
     const link=tocRail.querySelector('nav a.active');
     if(link){
       const bounds=tocRail.getBoundingClientRect(),item=link.getBoundingClientRect();
       const top=bounds.top+tocRail.querySelector('.eyebrow').offsetHeight+8,bottom=bounds.bottom-12;
       if(item.top<top)tocRail.scrollTop+=item.top-top;
       else if(item.bottom>bottom)tocRail.scrollTop+=item.bottom-bottom;
     }
   }
   previousHeading=active;
   previousRailVisible=railVisible;
 }
 new ResizeObserver(()=>{
   document.documentElement.style.setProperty('--reader-offset',(toolbar.getBoundingClientRect().height+20)+'px');
   if(innerWidth>800)closeMenu();
   onScroll();
 }).observe(toolbar);
 addEventListener('scroll',onScroll,{passive:true});onScroll();
})();
