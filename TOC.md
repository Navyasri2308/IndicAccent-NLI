import re
from pathlib import Path

readme = Path("README.md").read_text(encoding="utf-8")
heads = re.findall(r'^(#{2,6})\s+(.*)$', readme, flags=re.MULTILINE)
toc = []
for level, text in heads:
    anchor = text.strip().lower()
    anchor = re.sub(r'[^\w\s-]', '', anchor)  # remove punctuation
    anchor = anchor.replace(' ', '-')
    indent = '  ' * (len(level)-2)
    toc.append(f"{indent}- [{text}](#{anchor})")
print("\n".join(toc))
