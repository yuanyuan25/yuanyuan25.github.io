(() => {
  const state = document.getElementById('weekly-state');
  const heading = document.getElementById('weekly-title');
  const body = document.getElementById('weekly-content');
  const toc = document.getElementById('weekly-navigation');
  const datePattern = /^\d{4}-\d{2}-\d{2}\.html$/;
  const plain = text => new DOMParser().parseFromString(String(text || ''), 'text/html').body.textContent.trim();

  function retag(element, name) {
    if (element.localName === name) return element;
    const replacement = document.createElement(name);
    for (const attr of element.attributes) replacement.setAttribute(attr.name, attr.value);
    replacement.append(...element.childNodes);
    element.replaceWith(replacement);
    return replacement;
  }

  // Reuse content and links, while presentation and execution belong to this page.
  function prepare(element, sourceURL) {
    element.querySelectorAll('script,style,link,iframe,object,embed,base,meta').forEach(node => node.remove());
    for (const node of [element, ...element.querySelectorAll('*')]) {
      for (const attr of [...node.attributes]) {
        if (attr.name === 'style' || /^on/i.test(attr.name)) node.removeAttribute(attr.name);
      }
      for (const attr of ['href', 'src']) {
        if (!node.hasAttribute(attr)) continue;
        const value = node.getAttribute(attr);
        if (attr === 'href' && value.startsWith('#')) continue;
        const url = new URL(value, sourceURL);
        if (!['https:', 'http:', 'mailto:'].includes(url.protocol)) {
          node.removeAttribute(attr);
          continue;
        }
        const filename = url.pathname.split('/').pop();
        if (attr === 'href' && url.origin === location.origin &&
            url.pathname.startsWith(new URL('./', location.href).pathname) && datePattern.test(filename)) {
          node.setAttribute(attr, `read.html?report=${encodeURIComponent(filename)}${url.hash}`);
        } else {
          node.setAttribute(attr, url.href);
        }
      }
      if (node.localName === 'a' && node.target === '_blank') node.rel = 'noopener noreferrer';
    }
    element.querySelectorAll('.section-title').forEach(node => retag(node, 'h2'));
    element.querySelectorAll('.paper-title').forEach(node => retag(node, 'h3'));
    element.querySelectorAll('table').forEach(table => {
      const scroll = document.createElement('div');
      scroll.className = 'table-scroll';
      scroll.tabIndex = 0;
      table.replaceWith(scroll);
      scroll.append(table);
    });
    return element;
  }

  function jumpToHash() {
    if (!location.hash) return;
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = body.querySelectorAll('[id]');
    [...target].find(node => node.id === id)?.scrollIntoView({block: 'start'});
  }

  async function load() {
    try {
      let file = new URLSearchParams(location.search).get('report');
      if (!file) {
        const res = await fetch('reports.json', {cache: 'no-cache'});
        if (!res.ok) throw new Error('无法加载周报目录');
        const reports = await res.json();
        file = reports.filter(meta => datePattern.test(meta.file))
          .sort((a, b) => String(b.date).localeCompare(String(a.date)))[0]?.file;
      }
      if (!datePattern.test(file || '')) throw new Error('请选择有效的周报日期');
      const sourceURL = new URL(file, location.href);
      const response = await fetch(sourceURL, {cache: 'no-cache'});
      if (!response.ok) throw new Error(`这份周报暂时无法打开（HTTP ${response.status}）`);
      const source = new DOMParser().parseFromString(await response.text(), 'text/html');
      const metaNode = source.getElementById('report-meta');
      const meta = metaNode ? JSON.parse(metaNode.textContent) : {};
      const date = meta.date || file.replace('.html', '');
      heading.textContent = `大模型周报 · ${date}`;
      document.title = `${heading.textContent} | 知行手记 · Yuan’s Notes`;
      document.getElementById('weekly-week').textContent = meta.week || date;
      document.getElementById('weekly-summary').textContent = plain(meta.summary);
      document.querySelector('meta[name="description"]').content = plain(meta.summary);
      document.getElementById('weekly-source').href = file;
      const stats = document.getElementById('weekly-stats');
      const sourceStats = [...source.querySelectorAll('.stat-item')].map(item => ({
        value: item.querySelector('.stat-number,.number')?.textContent.trim(),
        label: item.querySelector('.stat-label,.label')?.textContent.trim(),
      })).filter(item => item.value && item.label);
      for (const stat of sourceStats.length ? sourceStats : meta.stats || []) {
        const item = document.createElement('span'); item.className = 'card-stat';
        const value = document.createElement('span'); value.className = 'number';
        value.textContent = stat.value; item.append(value, ` ${stat.label}`); stats.append(item);
      }
      const container = source.querySelector('.container') || source.body;
      const fragment = document.createDocumentFragment();
      for (const child of [...container.children]) {
        if (child.matches('header,.header')) {
          child.querySelectorAll('.summary,.summary-box,.subtitle').forEach(extra => {
            fragment.append(prepare(document.importNode(extra, true), sourceURL));
          });
          continue;
        }
        if (child.matches('header,footer,.header,.footer,.stats-bar,.stats,.toc,.nav,nav,script,style')) continue;
        fragment.append(prepare(document.importNode(child, true), sourceURL));
      }
      if (!fragment.querySelector('.section,.paper-card')) throw new Error('暂不支持这份周报的正文结构，请查看原始周报');
      body.append(fragment);
      body.querySelectorAll('.section').forEach((section, index) => {
        if (!section.id) section.id = `weekly-section-${index + 1}`;
        const title = section.querySelector('.section-title,h2');
        if (!title) return;
        const link = document.createElement('a');
        link.href = '#' + encodeURIComponent(section.id); link.textContent = title.textContent.trim(); toc.append(link);
      });
      const originalFooter = source.querySelector('footer,.footer');
      if (originalFooter) {
        const copy = originalFooter.cloneNode(true);
        copy.querySelectorAll('p:has(a)').forEach(node => node.remove());
        document.getElementById('weekly-attribution').textContent = copy.textContent.replace(/\s+/g, ' ').trim();
      }
      document.getElementById('weekly-toc').open = matchMedia('(min-width: 901px)').matches;
      state.hidden = true;
      document.getElementById('weekly-layout').hidden = false;
      document.body.dataset.report = file;
      document.body.dataset.state = 'ready';
      requestAnimationFrame(jumpToHash);
    } catch (error) {
      state.textContent = error.message;
      state.classList.add('error');
      document.body.dataset.state = 'error';
    }
  }
  addEventListener('hashchange', jumpToHash);
  load();
})();
