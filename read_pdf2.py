import fitz
doc = fitz.open(r'C:\Users\Uluxih\Zotero\storage\EYF374TE\A_three-dimensional_equivalent_continuum_constitutive1.pdf')
print(len(doc))
with open('equiv_continuum_text.txt', 'w', encoding='utf-8') as f:
    for p in doc:
        f.write(p.get_text())
        f.write('\n\n--PAGE BREAK--\n\n')
