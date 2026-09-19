# PO Pilot

BusyBee PO Pilot turns purchase orders into reviewed, validated records. Upload PDFs or images, extract structured fields with a vision-capable AI model, edit the result, resolve validation warnings, save approved data to a local dashboard, and download PO line items.

> **Privacy warning:** Do not upload confidential or sensitive company documents to this demo. Uploaded files are held only in memory by the app and are not deliberately logged or saved, but live extraction sends the document to the configured AI provider for processing. Review that provider's data policies before using real documents.

## Who it is for

PO Pilot is built for early-stage consumer-goods founders who manually transfer retailer purchase orders into spreadsheets. It removes the repetitive first-pass copying while keeping a human review step before export.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set your OpenAI API key in the same terminal. Never put it in `app.py` or commit it to source control.

```bash
export OPENAI_API_KEY="your-api-key-here"
```

The default model is `gpt-4.1-mini`. To use another compatible vision model, set `OPENAI_MODEL`:

```bash
export OPENAI_MODEL="gpt-4.1-mini"
```

Run the app:

```bash
streamlit run app.py
```

## What works

- Bulk drag-and-drop for PDF, PNG, JPG, and JPEG POs with automated sequential extraction and a per-document review queue
- Guided upload-to-review wizard that displays only the current stage instead of stacking the uploader and review workspace
- Duplicate protection that skips identical uploaded files, flags repeated PO numbers, and requires explicit verification before replacing a saved PO
- Vision-based PO extraction into a strict, validated schema
- Side-by-side, in-app preview of the original PDF or image and the editable extracted data
- Editable PO summary and line-item table
- Styled extract → review → validate → save workflow with per-document status
- Checks for missing identifiers, dates, invalid quantities/prices, and total mismatches
- Separate subtotal, shipping, tax, other-charge, and discount fields so grand totals reconcile correctly
- Local SQLite history for approved POs and line items (uploaded source files are not stored)
- A weekly dashboard for order value, tax, shipping, PO count, estimated time saved, and estimated labor savings at a clearly disclosed $20/hour data-entry rate
- Saved-order history with deletion controls
- Inventory ledger populated directly from saved PO line-item quantities
- Inventory dashboard with PO units and on-hand totals by SKU (or item description when no SKU is printed)
- Currency comparisons with a small tolerance for floating-point rounding
- Excel-ready CSV download with PO fields repeated on each line-item row
- Friendly extraction errors and a retry path
- A no-API synthetic sample with three items and an intentional discrepancy

## What is simulated

The **Use synthetic sample** option bypasses the API and loads hardcoded demonstration data. The data, company, SKUs, and PO number are all synthetic. Its third line has an intentionally incorrect line total, so validation reliably shows both a line-level mismatch and a PO-total mismatch. Approval/download stays disabled until the warnings are corrected.

There is no retailer integration, multi-user authentication, cloud deployment, or automatic Excel upload in this MVP. “Approve” means the reviewed structured data can be saved to the local SQLite dashboard and exported as a CSV download. Each saved PO adds its line-item quantities to inventory; deleting it reverses those inventory additions. Uploaded PDF and image files are never stored in the database.

## Where AI is used

AI is used only for document extraction. The uploaded document and extraction instructions are sent to a vision-capable OpenAI model, which is required to return structured data matching the `PurchaseOrder` schema. Pydantic validates that result before the UI displays it. Editing, validation rules, and CSV creation are local, deterministic application logic.

## Suggested 60-second demo

1. Open PO Pilot and explain the founder bottleneck in one sentence.
2. Select **Use synthetic sample**, then click **Extract PO**.
3. Point out the editable PO summary and three extracted line items.
4. Show the validation warnings caused by the intentional third-line discrepancy.
5. Change the third line total from `175.00` to `180.00`; the corrected line sum is `470.00`, clearing both warnings.
6. Click **Approve & Download CSV** and open the downloaded CSV in Excel or another spreadsheet app.
7. Mention that a real PDF or image follows the same flow when `OPENAI_API_KEY` is set.
