import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {siteRoot} from './paths.mjs';
import {startPreview} from './preview-server.mjs';

if (!process.env.PLAYWRIGHT_MODULE) throw new Error('Set PLAYWRIGHT_MODULE');
const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE));
const reports = JSON.parse(fs.readFileSync(path.join(siteRoot, 'llm_reports/reports.json')));
const sources = reports.map(report => ({file: report.file,
  html: fs.readFileSync(path.join(siteRoot, 'llm_reports', report.file), 'utf8')}));
const {base, close} = await startPreview();
const browser = await chromium.launch({headless: true,
  ...(process.env.CHROME_EXECUTABLE ? {executablePath: process.env.CHROME_EXECUTABLE} : {})});
const errors = [], pages = [];
const context = await browser.newContext();
await context.route('**/*', route => route.request().url().startsWith(base + '/') ? route.continue() : route.abort());
const page = await context.newPage();
page.on('pageerror', error => errors.push(String(error)));
page.on('response', response => {if (response.status() >= 400) errors.push(response.status() + ' ' + response.url());});
const chrome = () => ({
  brand: document.querySelector('.brand').textContent.replace(/\s+/g, ' ').trim(),
  footer: document.querySelector('.site-footer').textContent.replace(/\s+/g, ' ').trim(),
  links: [...document.querySelectorAll('.site-header nav a')].map(a => [a.textContent, new URL(a.href).pathname]),
  body: ['fontFamily', 'color', 'backgroundColor'].map(key => getComputedStyle(document.body)[key]),
  header: document.querySelector('.header-inner').getBoundingClientRect().toJSON(),
});
async function ready() {
  await page.waitForFunction(() => ['ready', 'error'].includes(document.body.dataset.state));
  assert.equal(await page.locator('body').getAttribute('data-state'), 'ready', await page.locator('#weekly-state').textContent());
}
try {
  for (const width of [1500, 390]) {
    await page.setViewportSize({width, height: 1000});
    await page.goto(base + '/');
    await page.evaluate(() => document.fonts.ready);
    const siteChrome = await page.evaluate(chrome);
    for (const source of sources) {
      await page.goto(`${base}/llm_reports/read.html?report=${source.file}`);
      await ready();
      await page.evaluate(() => document.fonts.ready);
      const result = await page.evaluate(html => {
        const source = new DOMParser().parseFromString(html, 'text/html');
        const normalize = node => node.textContent.replace(/\s+/g, ' ').trim();
        const originals = [...source.querySelectorAll('.paper-card')].map(normalize);
        const rendered = [...document.querySelectorAll('#weekly-content .paper-card')].map(normalize);
        const summaries = [...source.querySelectorAll('.summary,.summary-box,.subtitle')].map(normalize);
        const displayedSummaries = [...document.querySelectorAll('#weekly-content .summary,#weekly-content .summary-box,#weekly-content .subtitle')].map(normalize);
        const stats = [...source.querySelectorAll('.stat-item')].map(item => [
          item.querySelector('.stat-number,.number')?.textContent.trim(),
          item.querySelector('.stat-label,.label')?.textContent.trim(),
        ]).filter(([value, label]) => value && label).map(([value, label]) => `${value} ${label}`);
        const displayedStats = [...document.querySelectorAll('#weekly-stats .card-stat')].map(normalize);
        const brokenAnchors = [...document.querySelectorAll('#weekly-navigation a')].filter(a => !document.getElementById(decodeURIComponent(a.hash.slice(1)))).map(a => a.hash);
        return {originals, rendered, summaries, displayedSummaries, stats, displayedStats,
          brokenAnchors, pageWidth: document.documentElement.scrollWidth, sections: source.querySelectorAll('.section').length,
          toc: document.querySelectorAll('#weekly-navigation a').length};
      }, source.html);
      assert.deepEqual(result.rendered, result.originals, source.file + ': paper text must be preserved');
      assert.deepEqual(result.displayedSummaries, result.summaries, source.file + ': trends must be preserved');
      if (result.stats.length) assert.deepEqual(result.displayedStats, result.stats, source.file + ': stats must be preserved');
      assert.deepEqual(result.brokenAnchors, []);
      assert.equal(result.pageWidth, width, source.file + ': page overflow');
      assert.equal(result.toc, result.sections, source.file + ': complete direction TOC');
      assert.deepEqual(await page.evaluate(chrome), siteChrome, source.file + ': shared site layout');
      assert.equal(await page.locator('.site-header nav [aria-current="page"]').textContent(), '大模型周报');
      pages.push({file: source.file, width, papers: result.originals.length, directions: result.sections});
    }
    await page.goto(base + '/');
    await page.locator('#recent-reports .recent-item').first().waitFor();
    await page.locator('#recent-reports .recent-item').first().click();
    await ready();
    assert.equal(await page.locator('body').getAttribute('data-report'), reports[0].file);
    await page.locator('#weekly-source').click();
    assert.ok(page.url().endsWith('/llm_reports/' + reports[0].file));
    await page.goto(base + '/llm_reports/');
    await page.locator('#report-list .card').first().waitFor();
    await page.locator('#report-list .card').first().getByRole('link').click();
    await ready();
    await page.screenshot({path: '/tmp/yuan-notes-weekly-' + width + '.png'});
    await page.getByRole('link', {name: '阅读完整精读 →', exact: true}).click();
    assert.ok(page.url().endsWith('/notes/llm/agents/AgentGarten.html'));
    await page.getByRole('link', {name: '2026-10-12 大模型周报中的 AgentGarten 条目', exact: true}).click();
    await ready();
    assert.equal(new URL(page.url()).hash, '#agentgarten');
    assert.ok(await page.locator('#agentgarten').isVisible());
  }
  await page.goto(base + '/llm_reports/read.html?report=2026-07-28.html');
  await ready();
  await page.locator('#weekly-content a[href="read.html?report=2026-07-27.html"]').first().click();
  await ready();
  assert.equal(await page.locator('body').getAttribute('data-report'), '2026-07-27.html');
  await page.goto(base + '/llm_reports/read.html');
  await ready();
  assert.equal(await page.locator('body').getAttribute('data-report'), reports[0].file);
  await page.goto(base + '/llm_reports/read.html?report=../index.html');
  await page.waitForFunction(() => document.body.dataset.state === 'error');
  assert.match(await page.locator('#weekly-state').textContent(), /有效的周报日期/);
  assert.deepEqual(errors, []);
  const result = {date: new Date().toLocaleDateString('en-CA', {timeZone: 'Asia/Shanghai'}), passed: true,
    reports: sources.length, pageChecks: pages.length, widths: [1500, 390],
    functions: ['Preserve every paper, trend summary and source statistic', 'Shared site header, footer, font and colors',
      'Home and archive enter site reader', 'Original report link', 'Close reading link and backlink',
      'Links to other weekly reports stay in reader', 'Default latest report and invalid date state'], errors, pages};
  fs.writeFileSync(path.join(siteRoot, 'notes/llm/_reader/weekly-verification.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify({...result, pages: undefined}, null, 2));
} finally { await browser.close(); await close(); }
