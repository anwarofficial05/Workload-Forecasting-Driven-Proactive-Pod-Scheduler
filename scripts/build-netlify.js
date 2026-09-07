const fs = require('fs');
const path = require('path');

const rootDir = path.resolve(__dirname, '..');
const distDir = path.join(rootDir, 'dist');
const webStaticDir = path.join(rootDir, 'web', 'static');
const plotsDir = path.join(rootDir, 'evaluation', 'plots');
const pdfFile = path.join(rootDir, 'Project_Review_Demonstration_Guide.pdf');

console.log('[Netlify Build] Packaging Proactive Pod Scheduler static portal...');

if (!fs.existsSync(distDir)) {
  fs.mkdirSync(distDir, { recursive: true });
}
const distPlotsDir = path.join(distDir, 'plots');
if (!fs.existsSync(distPlotsDir)) {
  fs.mkdirSync(distPlotsDir, { recursive: true });
}

const filesToCopy = ['index.html', 'styles.css', 'app.js'];
let copiedFiles = 0;
filesToCopy.forEach(file => {
  const src = path.join(webStaticDir, file);
  const dest = path.join(distDir, file);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, dest);
    copiedFiles++;
  }
});
console.log('[Netlify Build] Synchronized ' + copiedFiles + ' web source files to dist/');

if (fs.existsSync(plotsDir)) {
  const plots = fs.readdirSync(plotsDir).filter(f => f.endsWith('.png'));
  plots.forEach(plot => {
    fs.copyFileSync(path.join(plotsDir, plot), path.join(distPlotsDir, plot));
  });
  console.log('[Netlify Build] Synchronized ' + plots.length + ' evaluation plots to dist/plots/');
}

if (fs.existsSync(pdfFile)) {
  fs.copyFileSync(pdfFile, path.join(distDir, 'Project_Review_Demonstration_Guide.pdf'));
  console.log('[Netlify Build] Bundled Project_Review_Demonstration_Guide.pdf in dist/');
}

const redirectsContent = '/*    /index.html   200\n';
fs.writeFileSync(path.join(distDir, '_redirects'), redirectsContent, 'utf8');
console.log('[Netlify Build] Generated dist/_redirects (200 rewrite rule)');

console.log('[Netlify Build] Deployment bundle ready in dist/ directory!');
