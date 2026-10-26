# Changelogs for pdf-translation

> Created and Maintained by @erbanku and fellow AI agents

## 2026-10-23

- Fit isolated text-layer table paragraphs to detected source cell bounds, keep their original font-size ceiling and horizontal origin, reserve glyph ascenders/descenders, and prevent expansion across row borders. Keep body layout and the Dify API unchanged; add eight real-font cell regressions.
- Reject invalid model/OCR/callback URLs before queueing requests; preserve valid client URLs and use the existing service default when its environment value is blank.
- Record connection, service and empty-response failures in V3 paragraph tracking, and reject incomplete jobs before rendering/publication instead of returning the untranslated source as success. Add 24 API and failure-handling regressions.
- Fix Chinese output in the V3 text-layer engine: replace Chinese prompt examples, expand target language codes, validate automatic terms and cached/returned translations against the actual source, and block unresolved wrong-language results before rendering and publication.
- Localize content-filter notices and preserve explicitly scoped glossary names; add 39 offline V3 language regressions, including fallback and pre-render checks.
- Fix native text-layer parser imports to reuse bundled pdfminer glyph, encoding and font-metric tables; add a Docker parser startup check and PDF-font regression tests.
- Constrain bitstring to `>=4.3.0,<5`: BabelDOC uses BitStream, which was removed in 5.0. Split dependency installation and asset warmup into separate Docker build steps.
- Add automatic target-language fonts to OCR rendering: bundled Noto Serif/Sans regular, bold, italic and bold italic; Noto Naskh Arabic; regional CJK fallbacks.
- Share font selection between layout measurement and Typst/direct PDF output, and invalidate render/width caches when fonts or target language change.
- Install bundled fonts in Docker and default the font family to `auto`; preserve explicit font-family overrides.
- Preserve the requested target language through formula segmentation, OCR reconstruction, retries, caches and Typst rendering.
- Reject unexpected Chinese text for non-CJK targets while preserving source names and explicitly scoped glossary literals.
- Measure Latin and accented text in layout fitting, enable target-language hyphenation, and share emergency fitting across Typst output paths.
- Exclude unscoped Chinese glossary targets from non-CJK jobs in both translation engines; normalize locale aliases such as `it_IT`.
- Add multilingual regression tests and isolate existing test-module stubs.

## 08/06/2026

- Add root `IDENTITY.md` (VoltBox SWE Agent identity; agents must read and follow, must not edit) and link it from the top of `AGENTS.md`.
