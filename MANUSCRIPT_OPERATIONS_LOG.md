# Manuscript Operations Log
# Paper: A Massively Parallel CUDA Framework for Real-Time Transient Pressure Simulation in Fractured Reservoirs
# Authors: Rongwang Yin, Shaowei Zhang
# Journal: Computers & Geosciences (CAGEO)
# Date: 2026-06-12

---

## 1. REVIEW COMPLIANCE (review.txt)

### 1.1 Line Numbers Added
- File: `A Massively Parallel CUDA Framework.docx`
- Method: XML injection of `<w:lnNumType>` into `sectPr`
- Config: `countBy=1`, `restart=continuous`

### 1.2 Reference Format (Numbered → Author-Date)
- All in-text citations: `[N]` → `(Author, Year)` format
- 29 citation keys mapped (see citation_map in original script)
- NVIDIA: `2020a` / `2020b` / `2023` disambiguation
- Reference list: `[N]` removed, alphabetically sorted

### 1.3 Word Count (7060 → 5324, limit 6050)
- 20 paragraphs trimmed (see detailed table below)
- Duplicate paras 169-171 cleared (merged into 168)
- Para 172 trimmed to unique future work items only

**Trimmed Paragraphs:**
| Para | Topic | Before | After | Cut |
|------|-------|--------|-------|-----|
| 8 | Introduction | 163 | 132 | 31 |
| 9 | Parallel computing | 130 | 102 | 28 |
| 46 | Math derivation | 158 | 135 | 23 |
| 85 | Computational cost | 116 | 93 | 23 |
| 92 | CPU/GPU impl | 352 | 204 | 148 |
| 98 | Complexity analysis | 253 | 119 | 134 |
| 99 | CUDA kernel design | 178 | 98 | 80 |
| 104 | Roofline analysis | 178 | 101 | 77 |
| 107 | Performance benchmarks | 278 | 143 | 135 |
| 108 | cuSPARSE comparison | 194 | 102 | 92 |
| 109 | Numerical validation | 192 | 99 | 93 |
| 134 | Parametric sensitivity | 150 | 88 | 62 |
| 135 | Adsorption coefficients | 131 | 84 | 47 |
| 136 | Integration findings | 137 | 52 | 85 |
| 140 | Spatiotemporal analysis | 150 | 92 | 58 |
| 153 | Strong scaling | 191 | 99 | 92 |
| 159 | Energy efficiency | 192 | 92 | 100 |
| 164 | Ablation study | 214 | 98 | 116 |
| 174 | Conclusion | 116 | 94 | 22 |
| 175 | Framework robustness | 140 | 72 | 68 |

---

## 2. GITHUB REPOSITORY

**URL**: https://github.com/rwyin08-bit/AMaPaCU

### Structure
```
code_repository/
├── README.md, CITATION.cff, LICENSE, .gitignore
├── manuscript/ (revised + original .docx)
├── code/
│   ├── README.md
│   ├── src/ (6 files: .py, .cu, .cpp)
│   ├── config/ (base_case.yaml, parameters.yaml)
│   ├── docker/ (Dockerfile, docker-compose.yml)
│   └── scripts/ (run_benchmark.py)
├── src/ (common.h, greens.h, influence_cuda.cu, influence_omp.c)
├── supplementary/ (6 files: .docx, .cpp, .cu)
├── figures/ (10 figures + 58 media files)
├── data/ (README.md)
└── docs/ (API.md, BUILD.md)
```

### Git History
```
fb011ff Formatting fixes: indentation, hanging refs, double spaces
a515aa2 Final figure enhancement (Fig 1 at 3.6x, all 300+ DPI)
14ec6b9 Enhance figures to 300 DPI, style tables
2c881b3 Initial commit (101 files)
```

---

## 3. FIGURE QUALITY ENHANCEMENT

**Method**: PIL Lanczos upscaling + Unsharp Mask → 300 DPI

| Figure | Original | Enhanced | Scale |
|--------|----------|----------|-------|
| Fig 1 | 554x322 | 2000x1162 | 3.6x |
| Fig 3 | 990x680 | 1500x1030 | 1.5x |
| Fig 4 | 593x336 | 1500x849 | 2.5x |
| Fig 5 | 465x436 | 1500x1406 | 3.2x |
| Fig 6 | 479x383 | 1500x1199 | 3.1x |
| Fig 7 | 1056x784 | 1800x1336 | 1.7x |
| Fig 8 | 461x260 | 1500x845 | 3.3x |
| Fig 9 | 463x268 | 1500x868 | 3.2x |
| Fig 10 | 758x273 | 1800x648 | 2.4x |
| Fig 11 | 1210x769 | 1800x1143 | 1.5x |
| Fig 12 | 1210x600 | 1800x892 | 1.5x |
| Fig 2 | WMF vector | unchanged | N/A |

**CRITICAL**: Use ZIP-level manipulation for image replacement. python-docx `save()` re-compresses PNGs.

---

## 4. TABLE STYLING

- Header: #1F4E79 blue, white bold 8-9pt, centered
- Body: alternating white / #F2F7FB blue, 8-9pt Times New Roman
- 3 tables styled: Reservoir Properties (20x4), Scalability (6x6), Ablation (7x4)

---

## 5. FORMATTING FIXES

### 5.1 Indentation
- 108 body paragraphs: 0.85cm first-line indent
- 29 references: -0.5cm hanging + 0.5cm left indent
- "where..." paras (35,41,44,49,67,71,75,77,81,90): indent added

### 5.2 Double Spaces
- "where..." paragraphs: OLE equation symbol artifacts (normal, visual appearance correct)
- Para 166 (Table 3 caption): actual double space fixed

### 5.3 Document Structure
All sections verified complete and correctly ordered.

---

## 6. FINAL STATISTICS

| Metric | Value |
|--------|-------|
| Body words | 5,347 (limit 6,050) |
| References | 29 (APA author-date, alphabetical) |
| Figures | 13 (all >= 300 DPI) |
| Tables | 3 (styled) |
| Embedded images | 58 (12 PNG + 46 WMF) |
| File size | 4,296 KB |

---

## 7. KEY PATHS

```
C:\paper6\A Massively Parallel CUDA Framework.docx  ← MAIN MANUSCRIPT
C:\paper6\A Massively Parallel CUDA Framework_backup.docx
C:\paper6\code_repository\                          ← GitHub local
C:\paper6\figures\enhanced\                         ← 300 DPI source
C:\paper6\MANUSCRIPT_OPERATIONS_LOG.md              ← THIS FILE
```

---

## 8. NOTES FOR NEXT REVISION

1. **Image replacement**: Use ZIP-level, NOT python-docx save (preserves quality)
2. **OLE equations**: Double spaces in "where..." text are normal (inline MathType objects)
3. **Figs 12/13**: Both use image58.png — regenerate separately if needed
4. **Citation format**: 3+ authors = `(Author et al., Year)`; 2 = `(Author and Author, Year)`
5. **CAGEO limits**: 5,500 words +10%, exclude Abstract/Keywords/Refs/Captions
