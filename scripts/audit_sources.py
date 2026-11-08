"""Read every archive source file, hash every resource, preserve attribution."""
import hashlib, json, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'external/raag-ratnakar-main'
rows = []
text = []
for p in sorted(src.rglob('*')):
    if not p.is_file():
        continue
    raw = p.read_bytes()
    rows.append({'path': str(p.relative_to(src)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    if p.suffix in {'.py','.jsx','.js','.css','.json','.md','.txt','.html','.ipynb'} or p.name == '.gitignore':
        content = raw.decode('utf-8', errors='replace')
        if p.suffix == '.ipynb':
            nb = json.loads(content)
            content = '\n'.join(''.join(c.get('source',[])) + '\nOUTPUT: ' + str(c.get('outputs',[])) for c in nb['cells'])
        text.append('\n## '+str(p.relative_to(src))+'\n'+content)
data = json.loads((src/'raga_sthayi_bhava_dataset_enriched.json').read_text())
links = [link for r in data for group in r['youtube_links'].values() for link in group]
report = {'files': rows, 'ragas':len(data),'youtube_link_entries':len(links),'unique_youtube_links':len(set(links)), 'metadata_keys': sorted(set(k for r in data for k in r)), 'license_files':[r['path'] for r in rows if any(x in r['path'].lower() for x in ['license','licence','copying'])]}
(ROOT/'reports/source_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(ROOT/'reports/repository_source_review.txt').write_text('\n'.join(text),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
