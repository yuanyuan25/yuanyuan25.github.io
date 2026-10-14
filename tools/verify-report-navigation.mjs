import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {siteRoot} from './notes-reader/paths.mjs';
import {startPreview} from './notes-reader/preview-server.mjs';

const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE));
const {base, close} = await startPreview();
const browser = await chromium.launch({headless: true, executablePath: process.env.CHROME_EXECUTABLE});
const context = await browser.newContext();
context.setDefaultTimeout(10000);
const page = await context.newPage();
const errors = [];
page.on('pageerror', error => errors.push(String(error)));
page.on('response', response => { if (response.status() >= 400) errors.push(response.url()); });
const reports = fs.readdirSync(path.join(siteRoot, 'llm_reports')).filter(file =>
  /^\d{4}-\d{2}-\d{2}\.html$/.test(file) && fs.readFileSync(path.join(siteRoot, 'llm_reports', file), 'utf8').includes('report-navigation.js'));
const template = fs.readFileSync(path.join(siteRoot, 'llm_reports/report_template.html'), 'utf8');
const ids = ['Agentic', '中文分类', 'Infra'];
const fixture = template.replace(/\{\{(\w+)\}\}/g, (_, key) => ({
  DATE: 'Navigation test', REPORT_META_JSON: '{}', HEADER_META: 'Navigation test',
  STATS_BAR: '', CORE_TRENDS: '<li>Navigation test</li>', FOOTER: 'Page end',
  TOC: ids.map(id => `<li><a href="#${id}">${id}</a></li>`).join(''),
  SECTIONS: ids.map((id, i) => `<section class="section" id="${id}" style="min-height:${i === 2 ? 80 : 1600}px"><h2 class="section-title">${id}</h2>${i === 1 ? '<div class="paper-card" id="paper-detail" style="margin-top:600px">Paper anchor</div>' : ''}</section>`).join('')
}[key] ?? ''));
await context.route('**/*', route => {
  const url = route.request().url();
  if (url === base + '/llm_reports/navigation-fixture.html') return route.fulfill({contentType: 'text/html; charset=utf-8', body: fixture});
  return url.startsWith(base + '/') ? route.continue() : route.abort();
});
const current = page.locator('.nav-links a[aria-current="location"]');
async function expectCurrent(href) {
  await page.waitForFunction(href => {
    const links = document.querySelectorAll('.nav-links a[aria-current="location"]');
    return links.length === 1 && links[0].getAttribute('href') === href && getComputedStyle(links[0]).color === 'rgb(255, 255, 255)';
  }, href).catch(async error => {
    throw new Error(`Expected ${href} at ${page.url()}, got ${await current.getAttribute('href')}`, {cause: error});
  });
  assert.equal(await current.count(), 1);
}
async function scrollInside(href, distance = 150) {
  await page.evaluate(({href, distance}) => {
    const section = document.getElementById(decodeURIComponent(href.slice(1)));
    const offset = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--report-nav-offset'));
    window.scrollTo({top: window.scrollY + section.getBoundingClientRect().top - offset + distance, behavior: 'instant'});
  }, {href, distance});
  await expectCurrent(href);
}
try {
  let checks = 0;
  for (const width of [1500, 768, 390, 320]) {
    await page.setViewportSize({width, height: 900});
    for (const file of [...reports, 'navigation-fixture.html']) {
      await page.goto(base + '/llm_reports/' + file);
      const links = await page.locator('.nav-links a').evaluateAll(links => links.map(link => link.getAttribute('href')));
      await expectCurrent(links[0]);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), width, file + ': horizontal overflow');
      for (const href of links) {
        await page.locator('.nav-links a').nth(links.indexOf(href)).click();
        await expectCurrent(href);
        const geometry = await page.evaluate(href => {
          const nav = document.querySelector('.nav:has(.nav-links)').getBoundingClientRect();
          const section = document.getElementById(decodeURIComponent(href.slice(1))).getBoundingClientRect();
          const link = document.querySelector('.nav-links a[aria-current="location"]');
          const style = getComputedStyle(link);
          return {navTop: nav.top, headingTop: section.top, navBottom: nav.bottom, color: style.color, background: style.backgroundColor};
        }, href);
        assert.ok(Math.abs(geometry.navTop) < 1, file + ': classification nav should remain pinned');
        assert.ok(geometry.headingTop >= geometry.navBottom - 1, file + ': heading hidden by navigation');
        assert.equal(geometry.color, 'rgb(255, 255, 255)');
        assert.notEqual(geometry.background, 'rgb(255, 255, 255)');
      }
      // Native history, including Chinese hashes, must retain the right highlight.
      await page.goBack();
      await expectCurrent(links.at(-2));
      await page.goForward();
      await expectCurrent(links.at(-1));
      // Scrolling must override the URL's last clicked category, in both directions.
      for (const index of [1, 0, 1]) await scrollInside(links[index]);
      await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
      await expectCurrent(links.at(-1));
      // Clicking the same hash still navigates after manually scrolling away.
      await page.locator('.nav-links a').last().click();
      await expectCurrent(links.at(-1));
      await scrollInside(links[0]);
      await page.locator('.nav-links a').last().click();
      await expectCurrent(links.at(-1));
      checks++;
      console.log(`Passed ${file} at ${width}px`);
    }
  }
  // Direct article anchors and a changing nav height are separate from category clicks.
  for (const [file, anchor, href] of [['2026-10-12.html', 'opsft', '#后训练---Post-training'], ['navigation-fixture.html', 'paper-detail', '#中文分类']]) {
    await page.goto(base + '/llm_reports/' + file + '#' + anchor);
    await expectCurrent(href);
    for (const width of [1500, 390, 320]) {
      await page.setViewportSize({width, height: 900});
      await page.locator('.nav-links a').filter({hasText: file.startsWith('2026') ? '后训练' : '中文分类'}).click();
      await expectCurrent(href);
      assert.ok(await page.evaluate(href => document.getElementById(decodeURIComponent(href.slice(1))).getBoundingClientRect().top >= document.querySelector('.nav:has(.nav-links)').getBoundingClientRect().bottom, href));
    }
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({passed: true, pageChecks: checks, reports, widths: [1500, 768, 390, 320], checks: ['Template-generated page', 'Click and scroll highlighting', 'Visible active colors', 'Pinned navigation and visible headings', 'Chinese hashes and paper anchors', 'Native back/forward and same-hash clicks', 'Short last section', 'Responsive nav height'], errors}, null, 2));
} finally {
  await browser.close();
  await close();
}
