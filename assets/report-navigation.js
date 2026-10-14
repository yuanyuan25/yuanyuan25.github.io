(() => {
  const nav = document.querySelector('.nav:has(.nav-links)');
  if (!nav) return;
  const entries = [...nav.querySelectorAll('a[href^="#"]')].flatMap(link => {
    const section = document.getElementById(decodeURIComponent(link.hash.slice(1)));
    return section ? [{link, section}] : [];
  });
  if (!entries.length) return;

  let activeLink;
  let frame;
  let readingOffset = 0;
  const activate = link => {
    if (activeLink === link) return;
    for (const entry of entries) entry.link.removeAttribute('aria-current');
    link.setAttribute('aria-current', 'location');
    activeLink = link;
  };
  const update = () => {
    frame = null;
    let current = entries[0];
    // Track section starts, so long sections stay selected throughout reading.
    for (const entry of entries) {
      if (entry.section.getBoundingClientRect().top > readingOffset + 1) break;
      current = entry;
    }
    // A short final section may never reach the reading line before page end.
    if (window.scrollY > 0 && window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
      current = entries.at(-1);
    }
    activate(current.link);
  };
  const schedule = () => {
    if (frame == null) frame = requestAnimationFrame(update);
  };
  const measure = () => {
    const stickyTop = parseFloat(getComputedStyle(nav).top) || 0;
    readingOffset = nav.getBoundingClientRect().height + stickyTop + 16;
    document.documentElement.style.setProperty('--report-nav-offset', `${readingOffset}px`);
    schedule();
  };

  for (const {link} of entries) {
    link.addEventListener('click', event => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      activate(link);
      // Let native anchors handle scrolling, URL hashes and browser history.
      schedule();
    });
  }
  window.addEventListener('scroll', schedule, {passive: true});
  window.addEventListener('resize', measure);
  window.addEventListener('hashchange', schedule);
  window.addEventListener('pageshow', measure);
  window.addEventListener('load', measure);
  new ResizeObserver(measure).observe(nav);
  document.fonts.ready.then(measure);
  measure();
})();
