// crop_svg.mjs — mermaid centres a short-wide graph in a tall canvas, which leaves
// dead whitespace in the rendered figure. Measure the real drawing bbox and report it.
import puppeteer from "puppeteer";
import { readdirSync } from "node:fs";

const dir = process.argv[2];
const files = readdirSync(dir).filter(f => f.endsWith(".svg") && /^\d\d-/.test(f));
const browser = await puppeteer.launch({ args: ["--no-sandbox"] });
for (const f of files) {
  const page = await browser.newPage();
  await page.goto(`file://${dir}/${f}`);
  const box = await page.evaluate(() => {
    const svg = document.querySelector("svg");
    const b = svg.getBBox();
    return { x: b.x, y: b.y, w: b.width, h: b.height };
  });
  console.log(JSON.stringify({ file: f, ...box }));
  await page.close();
}
await browser.close();
