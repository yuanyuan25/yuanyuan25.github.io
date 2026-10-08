(() => {
  const list = document.getElementById('recent-reports');

  async function loadReports() {
    try {
      const response = await fetch('llm_reports/reports.json', { cache: 'no-cache' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const reports = await response.json();
      if (!Array.isArray(reports)) throw new Error('Invalid report manifest');
      const recent = reports
        .filter(report => report && /^\d{4}-\d{2}-\d{2}\.html$/.test(report.file))
        .sort((a, b) => String(b.date).localeCompare(String(a.date)))
        .slice(0, 3);
      if (!recent.length) return;

      const fragment = document.createDocumentFragment();
      for (const report of recent) {
        const link = document.createElement('a');
        link.className = 'recent-item';
        link.href = `llm_reports/${report.file}`;
        const date = document.createElement('time');
        date.dateTime = report.date;
        date.textContent = report.date;
        const body = document.createElement('div');
        const title = document.createElement('h3');
        title.textContent = report.title || report.date;
        const summary = document.createElement('p');
        summary.textContent = String(report.summary || '').replace(/<br\s*\/?>/gi, ' ');
        body.append(title, summary);
        const arrow = document.createElement('span');
        arrow.className = 'arrow';
        arrow.setAttribute('aria-hidden', 'true');
        arrow.textContent = '↗';
        link.append(date, body, arrow);
        fragment.append(link);
      }
      list.replaceChildren(fragment);
    } catch (error) {
      // Keep the usable archive link when the manifest cannot be loaded.
      console.warn('Could not load recent reports:', error);
    }
  }

  loadReports();
})();
