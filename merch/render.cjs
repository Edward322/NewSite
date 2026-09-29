// Рендер макетов в фото: node merch/render.cjs [фильтр]
// Нужен Playwright (npm i -g playwright или локально) и Chromium.
const path = require('path');
const fs = require('fs');

let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  ({ chromium } = require('/opt/node22/lib/node_modules/playwright'));
}

const SRC = path.join(__dirname, 'src');
const OUT = path.join(__dirname, 'photos');
const filter = process.argv[2] || '';

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const scenes = fs.readdirSync(SRC)
    .filter((f) => /^\d\d-.+\.html$/.test(f) && f.includes(filter))
    .sort();

  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1200 },
    deviceScaleFactor: 1.25,
  });

  for (const file of scenes) {
    await page.goto('file://' + path.join(SRC, file));
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(150);
    const out = path.join(OUT, file.replace(/\.html$/, '.jpg'));
    await page.locator('.scene').screenshot({ path: out, type: 'jpeg', quality: 90 });
    console.log('✓', path.relative(process.cwd(), out));
  }

  await browser.close();
})();
