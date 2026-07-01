# MNRAS Hardfix Revision Report, 2026-07-02

This report records the manuscript hardfix applied after the six-reviewer MNRAS-style audit.

## Inputs

- Previous review PDF: `z2quijote_full_manuscript_20260702_mnras_review_word_export.pdf`
- Previous review DOCX: `z2quijote_full_manuscript_20260702_mnras_review.docx`
- Maintained Markdown source: `z2quijote_full_manuscript_20260701.md`

## Outputs

- Hardfix Markdown: `z2quijote_full_manuscript_20260702_mnras_hardfix.md`
- Hardfix DOCX: `z2quijote_full_manuscript_20260702_mnras_hardfix.docx`
- Hardfix PDF: `z2quijote_full_manuscript_20260702_mnras_hardfix_word_export.pdf`
- Hardfix audit JSON: `z2quijote_mnras_hardfix_audit_20260702.json`
- Rendered page images: `rendered_pages_mnras_hardfix_20260702/page-01.png` through `page-23.png`
- Page overview: `rendered_pages_mnras_hardfix_20260702/contact_sheet_all_pages.png`

## Reviewer Issues Addressed

1. Title-page metadata was added. The current file contains author and affiliation placeholders because the final author list and institutional affiliations have not yet been supplied.
2. All display equations were given visible equation numbers in the Word/PDF output. The Markdown source also carries explicit LaTeX `\tag{n}` markers.
3. Figure order was repaired. Figure 12 and Figure 13 were moved after Figure 11 into a support-diagnostics subsection, so the visible sequence is now Figure 1 through Figure 13.
4. Figure alt text was added for every figure. The PDF text extraction contains 13 `Alt text:` paragraphs.
5. Data Availability was rewritten to remove local machine paths and to state a public repository, supplementary package, or stable identifier release plan before external submission.
6. Figure 6/Figure 7 crowding was reduced by separating the later figure placement.
7. Figure 8 caption was revised to state that lower values are better and to clarify the Sobol64 reference line.
8. Figure 12 and Figure 13 were resized to avoid clipping in the two-column Word/PDF export.
9. The z2 translation harness was updated so equation-number tags do not create false formula-loss alarms.

## Audit Results

- PDF page count: 23
- Display equations numbered: 102
- Display formula retention in harness: 102/102
- Missing display formulae after tag normalization: 0
- Inline math count: 501 in candidate versus 463 in the Chinese source
- Missing inline math count: 27, mainly repeated exact `p68`, `k`, and numerical-token count differences rather than lost display definitions
- Chinese residue: 0
- Figure captions in PDF text: Figure 1 through Figure 13, each once
- Alt-text paragraphs in PDF text: 13
- Local path residues in PDF text: 0
- Rendered page images: 23/23
- Translation harness pass rate: 87.50%
- Translation harness weighted score: 0.8571
- Unit test: `python -m pytest versions\z2quijote\tests\test_translation_harness_loop.py -q` passed with 3 tests

## Remaining Submission Items

The manuscript is now suitable for the next internal review pass, but before external submission the following placeholders must be replaced:

- Final author names
- Final affiliations
- Corresponding-author details
- Funding acknowledgement
- Author-specific contribution statement
- Stable public repository, DOI archive, or supplementary-material link

