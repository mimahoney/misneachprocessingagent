"""PO Pilot: turn purchase orders into editable, validated CSV records."""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
CURRENCY_TOLERANCE = 0.02
SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg"]
DATABASE_PATH = Path(__file__).with_name("po_pilot.db")
LOGO_PATH = Path(__file__).with_name("assets") / "busybee-logo.jpg"
MINUTES_SAVED_PER_ORDER = 5
MINUTES_SAVED_PER_LINE_ITEM = 3
DATA_ENTRY_HOURLY_RATE = 20.00


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS purchase_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                po_number TEXT NOT NULL UNIQUE,
                buyer TEXT NOT NULL,
                order_date TEXT,
                ship_by_date TEXT,
                currency TEXT,
                subtotal REAL,
                shipping REAL,
                tax REAL,
                other_charges REAL,
                discount REAL,
                order_total REAL,
                source_filename TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS line_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                purchase_order_id INTEGER NOT NULL,
                sku TEXT,
                description TEXT,
                quantity REAL,
                unit_price REAL,
                line_total REAL,
                FOREIGN KEY (purchase_order_id) REFERENCES purchase_orders(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                merchant TEXT NOT NULL,
                receipt_date TEXT,
                currency TEXT,
                category TEXT NOT NULL,
                subtotal REAL,
                tax REAL,
                tip REAL,
                total REAL NOT NULL,
                payment_method TEXT,
                source_filename TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS receipt_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_id INTEGER NOT NULL,
                description TEXT,
                quantity REAL,
                amount REAL,
                FOREIGN KEY (receipt_id) REFERENCES receipts(id) ON DELETE CASCADE
            );
            """
        )


def save_approved_po(summary: dict, items: pd.DataFrame, source_filename: str) -> None:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO purchase_orders (
                po_number, buyer, order_date, ship_by_date, currency, subtotal,
                shipping, tax, other_charges, discount, order_total, source_filename
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary["po_number"].strip(),
                summary["buyer"].strip(),
                summary["order_date"] or None,
                summary["ship_by_date"] or None,
                summary["currency"].strip().upper() or None,
                as_number(summary.get("subtotal")),
                as_number(summary.get("shipping")),
                as_number(summary.get("tax")),
                as_number(summary.get("other_charges")),
                as_number(summary.get("discount")),
                as_number(summary.get("order_total")),
                source_filename,
            ),
        )
        order_id = cursor.lastrowid
        connection.executemany(
            """
            INSERT INTO line_items (
                purchase_order_id, sku, description, quantity, unit_price, line_total
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    order_id,
                    str(row.get("SKU") or ""),
                    str(row.get("Description") or ""),
                    as_number(row.get("Quantity")),
                    as_number(row.get("Unit Price")),
                    as_number(row.get("Line Total")),
                )
                for _, row in items.iterrows()
            ],
        )


def load_orders() -> pd.DataFrame:
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT po.id, po.po_number, po.buyer, po.order_date, po.ship_by_date,
                   po.currency, po.subtotal, po.shipping, po.tax, po.other_charges,
                   po.discount, po.order_total, po.source_filename, po.created_at,
                   COUNT(li.id) AS line_item_count
            FROM purchase_orders po
            LEFT JOIN line_items li ON li.purchase_order_id = po.id
            GROUP BY po.id
            ORDER BY po.created_at DESC, po.id DESC
            """,
            connection,
        )


def delete_order(order_id: int) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM purchase_orders WHERE id = ?", (order_id,))


def save_receipt(receipt: dict, items: pd.DataFrame, source_filename: str) -> None:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO receipts (
                merchant, receipt_date, currency, category, subtotal, tax, tip,
                total, payment_method, source_filename
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt["merchant"].strip(), receipt["receipt_date"] or None,
                receipt["currency"].strip().upper() or None, receipt["category"],
                as_number(receipt.get("subtotal")), as_number(receipt.get("tax")),
                as_number(receipt.get("tip")), as_number(receipt["total"]),
                receipt.get("payment_method") or None, source_filename,
            ),
        )
        receipt_id = cursor.lastrowid
        connection.executemany(
            "INSERT INTO receipt_items (receipt_id, description, quantity, amount) VALUES (?, ?, ?, ?)",
            [(receipt_id, str(row.get("Description") or ""), as_number(row.get("Quantity")),
              as_number(row.get("Amount"))) for _, row in items.iterrows()],
        )


def load_receipts() -> pd.DataFrame:
    with get_connection() as connection:
        return pd.read_sql_query(
            """SELECT r.*, COUNT(ri.id) AS item_count FROM receipts r
            LEFT JOIN receipt_items ri ON ri.receipt_id = r.id
            GROUP BY r.id ORDER BY r.created_at DESC, r.id DESC""",
            connection,
        )


def delete_receipt(receipt_id: int) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    line_total: Optional[float] = None


class PurchaseOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    po_number: Optional[str] = None
    buyer: Optional[str] = None
    order_date: Optional[str] = None
    ship_by_date: Optional[str] = None
    currency: Optional[str] = None
    subtotal: Optional[float] = None
    shipping: Optional[float] = None
    tax: Optional[float] = None
    other_charges: Optional[float] = None
    discount: Optional[float] = None
    order_total: Optional[float] = None
    line_items: list[LineItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("order_date", "ship_by_date")
    @classmethod
    def validate_iso_date(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            date.fromisoformat(value)
        return value

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and (len(value) != 3 or not value.isalpha()):
            raise ValueError("currency must be a three-letter code")
        return value.upper() if value else value


class ReceiptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: Optional[str] = None
    quantity: Optional[float] = None
    amount: Optional[float] = None


class ExpenseReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    merchant: Optional[str] = None
    receipt_date: Optional[str] = None
    currency: Optional[str] = None
    category: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    tip: Optional[float] = None
    total: Optional[float] = None
    payment_method: Optional[str] = None
    items: list[ReceiptItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


EXPENSE_CATEGORIES = [
    "Fuel", "Travel", "Meals", "Office supplies", "Shipping", "Software",
    "Marketing", "Professional services", "Utilities", "Equipment", "Other",
]


SYNTHETIC_SAMPLE = {
    "po_number": "SYN-PO-1042",
    "buyer": "Synthetic Corner Market",
    "order_date": "2026-09-15",
    "ship_by_date": "2026-09-25",
    "currency": "USD",
    "subtotal": 470.00,
    "shipping": 0.00,
    "tax": 0.00,
    "other_charges": 0.00,
    "discount": 0.00,
    "order_total": 470.00,
    "line_items": [
        {
            "sku": "SYN-ALM-12",
            "description": "Synthetic Almond Snack Packs",
            "quantity": 12,
            "unit_price": 10.00,
            "line_total": 120.00,
        },
        {
            "sku": "SYN-OAT-20",
            "description": "Synthetic Oat Bites",
            "quantity": 20,
            "unit_price": 8.50,
            "line_total": 170.00,
        },
        {
            "sku": "SYN-TEA-15",
            "description": "Synthetic Botanical Iced Tea",
            "quantity": 15,
            "unit_price": 12.00,
            "line_total": 175.00,
        },
    ],
    "notes": [
        "Synthetic demo data only.",
        "The third line total intentionally differs from quantity × unit price by $5.00.",
    ],
}


def extract_po_with_ai(file_bytes: bytes, filename: str, mime_type: str) -> PurchaseOrder:
    """Provider-specific extraction boundary; replace this function to change providers."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or use the synthetic sample."
        )

    encoded = base64.b64encode(file_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"
    if mime_type == "application/pdf":
        document_content = {
            "type": "input_file",
            "filename": filename,
            "file_data": data_url,
        }
    else:
        document_content = {
            "type": "input_image",
            "image_url": data_url,
            "detail": "high",
        }

    prompt = (
        "Extract this retailer purchase order. Return only data matching the supplied schema. "
        "Use null when a value is absent; do not guess. Dates must be YYYY-MM-DD. Currency must "
        "be a three-letter ISO code. Preserve each line item. Extract subtotal, shipping, tax, "
        "other charges, and discounts separately when printed; discounts must be positive values. "
        "The order total is the final grand total. Do not include markdown fences."
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    document_content,
                ],
            }
        ],
        text_format=PurchaseOrder,
    )
    if response.output_parsed is None:
        raise ValueError("The model did not return valid purchase-order JSON.")
    return response.output_parsed


def extract_receipt_with_ai(
    file_bytes: bytes, filename: str, mime_type: str
) -> ExpenseReceipt:
    """Extract and categorize an expense receipt with a replaceable provider boundary."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment first.")
    encoded = base64.b64encode(file_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"
    document = (
        {"type": "input_file", "filename": filename, "file_data": data_url}
        if mime_type == "application/pdf" or filename.lower().endswith(".pdf")
        else {"type": "input_image", "image_url": data_url, "detail": "high"}
    )
    prompt = (
        "Extract this business expense receipt. Use null for missing values and do not guess. "
        "Dates must be YYYY-MM-DD and currency a three-letter code. Categorize it as exactly one "
        f"of: {', '.join(EXPENSE_CATEGORIES)}. Amount is the full amount for each receipt item. "
        "Return only data matching the schema, without markdown fences."
    )
    response = OpenAI(api_key=api_key).responses.parse(
        model=MODEL,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt}, document,
        ]}],
        text_format=ExpenseReceipt,
    )
    if response.output_parsed is None:
        raise ValueError("The model did not return valid receipt data.")
    return response.output_parsed


def item_frame(po: PurchaseOrder) -> pd.DataFrame:
    columns = ["SKU", "Description", "Quantity", "Unit Price", "Line Total"]
    rows = [
        {
            "SKU": item.sku or "",
            "Description": item.description or "",
            "Quantity": item.quantity,
            "Unit Price": item.unit_price,
            "Line Total": item.line_total,
        }
        for item in po.line_items
    ]
    return pd.DataFrame(rows, columns=columns)


def is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value)


def as_number(value: object) -> Optional[float]:
    if is_blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_po(summary: dict, items: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if is_blank(summary["po_number"]):
        warnings.append("Missing PO number.")
    if is_blank(summary["buyer"]):
        warnings.append("Missing buyer/company.")
    if is_blank(summary["ship_by_date"]):
        warnings.append("Missing ship-by date.")

    numeric_columns = ["Quantity", "Unit Price", "Line Total"]
    for index, row in items.iterrows():
        row_number = index + 1
        if is_blank(row.get("SKU")):
            warnings.append(f"Line {row_number}: missing SKU.")
        values = {name: as_number(row.get(name)) for name in numeric_columns}
        if values["Quantity"] is None:
            warnings.append(f"Line {row_number}: quantity is missing or invalid.")
        elif values["Quantity"] < 1:
            warnings.append(f"Line {row_number}: quantity is less than 1.")
        if values["Unit Price"] is None:
            warnings.append(f"Line {row_number}: unit price is missing or invalid.")
        elif values["Unit Price"] < 0:
            warnings.append(f"Line {row_number}: unit price is negative.")
        if values["Line Total"] is None:
            warnings.append(f"Line {row_number}: line total is missing or invalid.")
        elif values["Quantity"] is not None and values["Unit Price"] is not None:
            expected = values["Quantity"] * values["Unit Price"]
            if abs(expected - values["Line Total"]) > CURRENCY_TOLERANCE:
                warnings.append(
                    f"Line {row_number}: quantity × unit price ({expected:.2f}) "
                    f"does not match line total ({values['Line Total']:.2f})."
                )

    po_total = as_number(summary.get("order_total"))
    subtotal = as_number(summary.get("subtotal"))
    line_totals = [as_number(value) for value in items.get("Line Total", [])]
    if line_totals and all(value is not None for value in line_totals):
        line_sum = sum(value for value in line_totals if value is not None)
        adjustments = {
            "shipping": as_number(summary.get("shipping")) or 0.0,
            "tax": as_number(summary.get("tax")) or 0.0,
            "other_charges": as_number(summary.get("other_charges")) or 0.0,
            "discount": as_number(summary.get("discount")) or 0.0,
        }
        has_adjustments = any(adjustments.values())
        comparison_total = subtotal
        comparison_label = "subtotal"
        if subtotal is None and not has_adjustments:
            comparison_total = po_total
            comparison_label = "PO total"
        if (
            comparison_total is not None
            and abs(line_sum - comparison_total) > CURRENCY_TOLERANCE
        ):
            warnings.append(
                f"Sum of line totals ({line_sum:.2f}) does not match "
                f"{comparison_label} ({comparison_total:.2f})."
            )

        if po_total is not None and (subtotal is not None or has_adjustments):
            total_base = subtotal if subtotal is not None else line_sum
            calculated_total = (
                total_base
                + adjustments["shipping"]
                + adjustments["tax"]
                + adjustments["other_charges"]
                - adjustments["discount"]
            )
            if abs(calculated_total - po_total) > CURRENCY_TOLERANCE:
                warnings.append(
                    f"Subtotal plus adjustments ({calculated_total:.2f}) does not match "
                    f"PO total ({po_total:.2f})."
                )
    return warnings


def csv_bytes(summary: dict, items: pd.DataFrame) -> bytes:
    records = []
    for _, item in items.iterrows():
        records.append(
            {
                "PO Number": summary["po_number"],
                "Buyer": summary["buyer"],
                "Order Date": summary["order_date"],
                "Ship By Date": summary["ship_by_date"],
                "Currency": summary["currency"],
                "SKU": item.get("SKU"),
                "Description": item.get("Description"),
                "Quantity": item.get("Quantity"),
                "Unit Price": item.get("Unit Price"),
                "Line Total": item.get("Line Total"),
            }
        )
    return pd.DataFrame(records).to_csv(index=False).encode("utf-8")


def render_document_preview(document: dict) -> None:
    """Display an uploaded document without writing it to disk."""
    file_bytes = document["bytes"]
    filename = document["filename"]
    mime_type = document["mime_type"]

    st.caption(filename)
    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        st.pdf(file_bytes, height=900, key=f"pdf-preview-{filename}")
        st.download_button(
            "Download original PDF",
            data=file_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.image(file_bytes, caption="Uploaded purchase order", use_container_width=True)


def load_result(po: PurchaseOrder, source: str, document: Optional[dict] = None) -> None:
    st.session_state.extracted_po = po.model_dump()
    st.session_state.line_items = item_frame(po)
    st.session_state.result_source = source
    st.session_state.source_document = document


def activate_queued_result(index: int) -> None:
    result = st.session_state.extraction_queue[index]
    load_result(
        PurchaseOrder.model_validate(result["po"]),
        result["source"],
        result["document"],
    )
    st.session_state.active_result_index = index


def workflow_steps(active_step: int) -> None:
    labels = ["Extract", "Review", "Validate", "Save"]
    steps = []
    for number, label in enumerate(labels, start=1):
        state = "complete" if number < active_step else "active" if number == active_step else ""
        marker = "✓" if number < active_step else str(number)
        steps.append(
            f'<div class="workflow-step {state}"><span>{marker}</span><strong>{label}</strong></div>'
        )
    st.markdown(
        f'<div class="workflow-steps">{"".join(steps)}</div>',
        unsafe_allow_html=True,
    )


def render_receipt_workflow() -> None:
    st.markdown(
        """<div class="section-heading"><span class="eyebrow">EXPENSE CAPTURE</span>
        <h2>Track expense receipts</h2><p>Upload gas, travel, meal, office, or other business receipts.</p></div>""",
        unsafe_allow_html=True,
    )
    files = st.file_uploader(
        "Upload expense receipts", type=SUPPORTED_TYPES, accept_multiple_files=True,
        key="receipt_uploader", help="Select one or more receipt PDFs or images.",
    )
    if st.button("Extract receipts", type="primary"):
        if not files:
            st.warning("Upload at least one receipt first.")
        else:
            results, failures = [], []
            progress = st.progress(0, text="Starting receipt extraction…")
            for position, uploaded in enumerate(files, start=1):
                progress.progress((position - 1) / len(files), text=f"Reading {uploaded.name}…")
                try:
                    raw = uploaded.getvalue()
                    receipt = extract_receipt_with_ai(raw, uploaded.name, uploaded.type)
                    results.append({"receipt": receipt.model_dump(), "document": {
                        "bytes": raw, "filename": uploaded.name, "mime_type": uploaded.type,
                    }, "status": "needs review"})
                except Exception:
                    failures.append(uploaded.name)
            progress.empty()
            if results:
                st.session_state.receipt_queue = results
                st.session_state.receipt_batch_id = st.session_state.get("receipt_batch_id", 0) + 1
                st.success(f"Extracted {len(results)} of {len(files)} receipts.")
            if failures:
                st.error("Could not extract: " + ", ".join(failures))

    queue = st.session_state.get("receipt_queue", [])
    if not queue:
        return
    batch = st.session_state.get("receipt_batch_id", 0)
    index = st.selectbox(
        "Receipt review queue", list(range(len(queue))),
        format_func=lambda value: f"{value + 1}. {queue[value]['document']['filename']} — {queue[value]['status']}",
        key=f"receipt_queue_{batch}",
    )
    result = queue[index]
    receipt = result["receipt"]
    key = f"receipt_{batch}_{index}"
    workflow_steps(4 if result["status"] == "saved" else 2)
    workspace = st.container(key="review_workspace")
    source, review = workspace.columns([1, 1.15], gap="large")
    with source:
        st.markdown('<div class="review-label">01 · RECEIPT IMAGE</div>', unsafe_allow_html=True)
        st.subheader("Original receipt")
        render_document_preview(result["document"])
    with review:
        st.markdown('<div class="review-label">02 · EXPENSE DATA</div>', unsafe_allow_html=True)
        st.subheader("Review and categorize")
        left, right = st.columns(2)
        with left:
            merchant = st.text_input("Merchant", receipt.get("merchant") or "", key=f"{key}_merchant")
            receipt_date = st.text_input("Receipt date", receipt.get("receipt_date") or "", key=f"{key}_date")
            currency = st.text_input("Currency", receipt.get("currency") or "", key=f"{key}_currency")
            subtotal = st.number_input("Subtotal", value=receipt.get("subtotal"), step=0.01,
                                       format="%.2f", key=f"{key}_subtotal")
        with right:
            extracted_category = receipt.get("category")
            category = st.selectbox(
                "Expense category", EXPENSE_CATEGORIES,
                index=EXPENSE_CATEGORIES.index(extracted_category) if extracted_category in EXPENSE_CATEGORIES else len(EXPENSE_CATEGORIES) - 1,
                key=f"{key}_category",
            )
            tax = st.number_input("Tax", value=receipt.get("tax"), step=0.01, format="%.2f", key=f"{key}_tax")
            tip = st.number_input("Tip", value=receipt.get("tip"), step=0.01, format="%.2f", key=f"{key}_tip")
            total = st.number_input("Total", value=receipt.get("total"), step=0.01, format="%.2f", key=f"{key}_total")
        payment_method = st.text_input(
            "Payment method", receipt.get("payment_method") or "", key=f"{key}_payment"
        )
        item_rows = [{"Description": item.get("description") or "", "Quantity": item.get("quantity"),
                      "Amount": item.get("amount")} for item in receipt.get("items", [])]
        items = st.data_editor(
            pd.DataFrame(item_rows, columns=["Description", "Quantity", "Amount"]),
            num_rows="dynamic", hide_index=True, use_container_width=True, key=f"{key}_items",
        )
        warnings = []
        if not merchant.strip(): warnings.append("Merchant is missing.")
        if not receipt_date.strip(): warnings.append("Receipt date is missing.")
        if total is None or total <= 0: warnings.append("Total must be greater than zero.")
        if subtotal is not None and total is not None:
            calculated = subtotal + (tax or 0) + (tip or 0)
            if abs(calculated - total) > CURRENCY_TOLERANCE:
                warnings.append(f"Subtotal plus tax and tip ({calculated:.2f}) does not match total ({total:.2f}).")
        if warnings:
            st.markdown(f'<div class="validation-summary warning"><strong>{len(warnings)} issues to resolve</strong><p>Check these values against the receipt.</p></div>', unsafe_allow_html=True)
            for warning in warnings: st.warning(warning)
        else:
            st.markdown('<div class="validation-summary pass"><strong>✓ Receipt validated</strong><p>This expense is ready to save.</p></div>', unsafe_allow_html=True)
        if st.button("Save expense receipt", type="primary", disabled=bool(warnings),
                     use_container_width=True, key=f"{key}_save"):
            data = {"merchant": merchant, "receipt_date": receipt_date, "currency": currency,
                    "category": category, "subtotal": subtotal, "tax": tax, "tip": tip,
                    "total": total, "payment_method": payment_method}
            save_receipt(data, items, result["document"]["filename"])
            result["status"] = "saved"
            st.session_state.save_message = f"Receipt from {merchant} was added to expenses."
            st.rerun()


def format_money(value: float, currency: Optional[str]) -> str:
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency or "", "")
    suffix = "" if symbol else f" {currency or ''}".rstrip()
    return f"{symbol}{value:,.2f}{suffix}"


def render_dashboard() -> None:
    st.markdown(
        """
        <div class="section-heading">
            <span class="eyebrow">OPERATIONS</span>
            <h2>This week at a glance</h2>
            <p>Your approved purchase orders, summarized automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    orders = load_orders()
    receipts = load_receipts()
    stored_order_label = "PO" if len(orders) == 1 else "POs"
    st.markdown(
        f"""
        <div class="database-status">
            <span class="status-dot"></span>
            <strong>SQLite connected</strong>
            <span>{len(orders)} {stored_order_label} · {len(receipts)} receipts stored locally</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if orders.empty:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">↗</div>
                <div>
                    <strong>Your dashboard is ready</strong>
                    <p>Process and save your first purchase order to see weekly totals here.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    created = pd.to_datetime(orders["created_at"], errors="coerce")
    today = pd.Timestamp.now().normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    weekly = orders.loc[created >= week_start].copy()

    currencies = weekly["currency"].dropna().unique().tolist()
    display_currency = currencies[0] if len(currencies) == 1 else None
    total_value = float(weekly["order_total"].fillna(0).sum())
    total_tax = float(weekly["tax"].fillna(0).sum())
    total_shipping = float(weekly["shipping"].fillna(0).sum())
    minutes_saved = int(
        len(weekly) * MINUTES_SAVED_PER_ORDER
        + weekly["line_item_count"].fillna(0).sum() * MINUTES_SAVED_PER_LINE_ITEM
    )
    labor_cost_saved = minutes_saved / 60 * DATA_ENTRY_HOURLY_RATE

    metric_columns = st.columns(3)
    metric_columns[0].metric("POs this week", f"{len(weekly):,}")
    metric_columns[1].metric(
        "Total order value",
        format_money(total_value, display_currency) if not weekly.empty else "—",
    )
    metric_columns[2].metric("Estimated time saved", f"{minutes_saved / 60:.1f} hrs")

    secondary_metrics = st.columns(3)
    secondary_metrics[0].metric(
        "Total tax",
        format_money(total_tax, display_currency) if not weekly.empty else "—",
    )
    secondary_metrics[1].metric(
        "Total shipping",
        format_money(total_shipping, display_currency) if not weekly.empty else "—",
    )
    secondary_metrics[2].metric(
        "Estimated labor saved",
        format_money(labor_cost_saved, "USD"),
        help=f"Estimated time saved × ${DATA_ENTRY_HOURLY_RATE:.0f}/hour data-entry rate.",
    )
    st.caption(
        "This week runs Monday through today. Time saved is estimated at 5 minutes per PO "
        f"plus 3 minutes per line item. Labor savings use a ${DATA_ENTRY_HOURLY_RATE:.0f}/hour "
        "data-entry rate."
    )
    if len(currencies) > 1:
        st.warning(
            "This week contains multiple currencies. Monetary totals are numeric sums and are "
            "shown without a currency symbol; no exchange-rate conversion is applied."
        )

    with st.expander("Saved purchase orders", expanded=True):
        display = orders.rename(
            columns={
                "po_number": "PO Number",
                "buyer": "Buyer",
                "order_date": "Order Date",
                "currency": "Currency",
                "subtotal": "Subtotal",
                "shipping": "Shipping",
                "tax": "Tax",
                "order_total": "Order Total",
                "line_item_count": "Items",
                "created_at": "Added",
            }
        )
        st.dataframe(
            display[
                [
                    "PO Number",
                    "Buyer",
                    "Order Date",
                    "Currency",
                    "Subtotal",
                    "Shipping",
                    "Tax",
                    "Order Total",
                    "Items",
                    "Added",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Subtotal": st.column_config.NumberColumn(format="%.2f"),
                "Shipping": st.column_config.NumberColumn(format="%.2f"),
                "Tax": st.column_config.NumberColumn(format="%.2f"),
                "Order Total": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        selected_id = st.selectbox(
            "Select a saved PO to delete",
            options=orders["id"].astype(int).tolist(),
            format_func=lambda order_id: (
                f"{orders.loc[orders['id'] == order_id, 'po_number'].iloc[0]} — "
                f"{orders.loc[orders['id'] == order_id, 'buyer'].iloc[0]}"
            ),
        )
        if st.button("Delete selected PO", type="secondary"):
            delete_order(int(selected_id))
            st.success("The selected PO and its line items were deleted.")
            st.rerun()

    st.markdown(
        """
        <div class="section-heading">
            <span class="eyebrow">EXPENSES</span>
            <h2>Receipt spending</h2>
            <p>Approved expenses grouped automatically for this week.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if receipts.empty:
        st.info("No saved receipts yet. Choose Expense receipts below to add one.")
    else:
        receipt_created = pd.to_datetime(receipts["created_at"], errors="coerce")
        weekly_receipts = receipts.loc[receipt_created >= week_start].copy()
        receipt_currency_values = weekly_receipts["currency"].dropna().unique().tolist()
        receipt_currency = receipt_currency_values[0] if len(receipt_currency_values) == 1 else None
        expense_columns = st.columns(3)
        expense_columns[0].metric("Receipts this week", f"{len(weekly_receipts):,}")
        expense_columns[1].metric(
            "Weekly expenses",
            format_money(float(weekly_receipts["total"].fillna(0).sum()), receipt_currency),
        )
        expense_columns[2].metric(
            "Receipt tax",
            format_money(float(weekly_receipts["tax"].fillna(0).sum()), receipt_currency),
        )
        category_totals = (
            weekly_receipts.groupby("category", as_index=False)["total"].sum()
            .sort_values("total", ascending=False)
            .rename(columns={"category": "Category", "total": "Total"})
        )
        st.dataframe(category_totals, use_container_width=True, hide_index=True)
        with st.expander("Saved expense receipts", expanded=False):
            st.dataframe(
                receipts[["merchant", "receipt_date", "category", "currency", "tax", "total", "created_at"]]
                .rename(columns={"merchant": "Merchant", "receipt_date": "Date", "category": "Category",
                                 "currency": "Currency", "tax": "Tax", "total": "Total", "created_at": "Added"}),
                use_container_width=True, hide_index=True,
            )
            receipt_id = st.selectbox(
                "Select a receipt to delete",
                receipts["id"].astype(int).tolist(),
                format_func=lambda value: f"{receipts.loc[receipts['id'] == value, 'merchant'].iloc[0]} — {receipts.loc[receipts['id'] == value, 'total'].iloc[0]:.2f}",
            )
            if st.button("Delete selected receipt", type="secondary"):
                delete_receipt(int(receipt_id))
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="BusyBee | PO Pilot", page_icon="🐝", layout="wide")
    initialize_database()
    st.markdown(
        """
        <style>
        :root {
            --ink: #090908;
            --muted: #58584f;
            --navy: #090908;
            --navy-soft: #22221f;
            --coral: #ffcb28;
            --coral-dark: #f1b900;
            --mint: #fff0a8;
            --line: #11110f;
            --surface: #ffffff;
            --canvas: #fffdf3;
        }
        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(255, 253, 243, 0.94);
            backdrop-filter: blur(12px);
        }
        [data-testid="stDecoration"] { display: none; }
        .main .block-container {
            max-width: 1500px;
            padding-top: 4rem;
            padding-bottom: 5rem;
        }
        .stMarkdown, .stMarkdown p, .stCaption, label,
        [data-testid="stWidgetLabel"] p {
            color: var(--ink);
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 54px;
            margin-bottom: 48px;
            color: white;
            background: #090908;
            border: 3px solid #090908;
            border-radius: 6px;
            box-shadow: 12px 12px 0 #ffcb28;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(330px, 0.75fr);
            gap: 48px;
            align-items: center;
        }
        .brand-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 34px;
            color: #ffcb28;
            font-size: 0.9rem;
            font-weight: 900;
            letter-spacing: 0.12em;
        }
        .brand-mark {
            display: inline-grid;
            place-items: center;
            width: 34px;
            height: 34px;
            border: 2px solid #ffcb28;
            border-radius: 50%;
            color: #090908;
            background: #ffcb28;
            font-size: 18px;
        }
        .hero h1 {
            margin: 0 0 12px;
            color: white !important;
            font-size: clamp(3.4rem, 6.5vw, 6.6rem);
            line-height: 0.88;
            letter-spacing: -0.075em;
            font-weight: 950;
        }
        .hero .tagline {
            max-width: 800px;
            margin: 0 0 18px;
            color: #ffffff !important;
            font-size: 1.55rem;
            line-height: 1.45;
            font-weight: 600;
        }
        .hero .persona {
            max-width: 880px;
            margin: 0;
            color: #d0d0c8 !important;
            font-size: 1.05rem;
            line-height: 1.6;
        }
        .logo-card {
            padding: 10px;
            background: #fffef8;
            border: 3px solid #ffcb28;
            box-shadow: 9px 9px 0 #ffcb28;
            transform: rotate(1.5deg);
        }
        .logo-card img { display: block; width: 100%; }
        .privacy-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 24px;
            padding: 8px 12px;
            color: #dce8f8;
            background: rgba(255,255,255,0.08);
            color: #090908;
            background: #ffcb28;
            border: 2px solid #ffcb28;
            border-radius: 2px;
            font-size: 0.86rem;
            font-weight: 800;
        }
        .section-heading { margin: 16px 0 24px; }
        .section-heading .eyebrow {
            display: inline-block;
            padding: 5px 9px;
            color: #090908;
            background: #ffcb28;
            font-size: 0.76rem;
            font-weight: 950;
            letter-spacing: 0.14em;
        }
        .section-heading h2 {
            margin: 4px 0 4px;
            color: var(--ink) !important;
            font-size: clamp(2.1rem, 4vw, 3.25rem);
            letter-spacing: -0.05em;
            font-weight: 950;
        }
        .section-heading p { margin: 0; color: var(--muted) !important; font-size: 1.08rem; }
        .database-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin: -6px 0 20px;
            padding: 8px 12px;
            color: #090908;
            background: #ffcb28;
            border: 2px solid #090908;
            border-radius: 2px;
            box-shadow: 3px 3px 0 #090908;
            font-size: 0.84rem;
        }
        .database-status .status-dot {
            width: 8px;
            height: 8px;
            background: #090908;
            border-radius: 50%;
            box-shadow: none;
        }
        .database-status span:last-child { color: #33332d; }
        .empty-state {
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 24px;
            margin-bottom: 14px;
            color: var(--ink);
            background: #ffffff;
            border: 3px solid #090908;
            border-radius: 4px;
            box-shadow: 8px 8px 0 #ffcb28;
        }
        .empty-state .empty-icon {
            display: grid;
            place-items: center;
            width: 44px;
            height: 44px;
            color: #090908;
            background: #ffcb28;
            border: 2px solid #090908;
            border-radius: 50%;
            font-size: 1.35rem;
        }
        .empty-state strong { font-size: 1rem; }
        .empty-state p { margin: 3px 0 0; color: var(--muted) !important; }
        [data-testid="stMetric"] {
            min-height: 124px;
            background: var(--surface);
            border: 2px solid var(--line);
            border-top: 9px solid #ffcb28;
            border-radius: 3px;
            padding: 18px 20px;
            box-shadow: 5px 5px 0 #090908;
        }
        [data-testid="stMetricLabel"] p { color: var(--muted) !important; }
        [data-testid="stMetricValue"] { color: var(--navy) !important; font-weight: 950; }
        [data-testid="stExpander"] {
            background: var(--surface);
            border: 2px solid var(--line);
            border-radius: 3px;
            box-shadow: 6px 6px 0 #ffcb28;
        }
        [data-testid="stExpander"] details > summary {
            color: #ffffff !important;
            background: #090908 !important;
        }
        [data-testid="stExpander"] details > summary p,
        [data-testid="stExpander"] details > summary span,
        [data-testid="stExpander"] details > summary svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-testid="stFileUploader"] {
            background: var(--surface);
            border: 2px solid var(--line);
            border-radius: 3px;
            padding: 14px;
            box-shadow: 6px 6px 0 #ffcb28;
        }
        [data-testid="stFileUploaderDropzone"] {
            color: var(--ink);
            background: #fffdf3;
            border: 2px dashed #090908;
            border-radius: 2px;
        }
        [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
        [data-baseweb="input"], [data-baseweb="select"] > div {
            background: white !important;
            border: 2px solid #090908 !important;
            border-radius: 2px !important;
        }
        [data-baseweb="input"] input { color: var(--ink) !important; }
        .st-key-review_workspace {
            padding: 18px;
            background: #ffcb28;
            border: 3px solid #090908;
            border-radius: 4px;
        }
        .st-key-review_workspace [data-testid="stColumn"] {
            padding: 18px;
            background: white;
            border: 2px solid var(--line);
            border-radius: 2px;
            box-shadow: 5px 5px 0 #090908;
        }
        .st-key-review_workspace [data-testid="stColumn"]:first-child {
            position: sticky;
            top: 4rem;
            align-self: flex-start;
        }
        .review-label {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            margin-bottom: 4px;
            padding: 5px 8px;
            color: #090908;
            background: #ffcb28;
            font-size: 0.78rem;
            font-weight: 950;
            letter-spacing: 0.1em;
        }
        .workflow-steps {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin: 20px 0 26px;
        }
        .workflow-step {
            display: flex;
            align-items: center;
            gap: 10px;
            min-height: 54px;
            padding: 10px 12px;
            color: #77776d;
            background: #ffffff;
            border: 2px solid #b9b9ae;
        }
        .workflow-step span {
            display: grid;
            place-items: center;
            width: 28px;
            height: 28px;
            flex: 0 0 28px;
            color: #ffffff;
            background: #85857c;
            border-radius: 50%;
            font-weight: 950;
        }
        .workflow-step.complete,
        .workflow-step.active {
            color: #090908;
            border-color: #090908;
        }
        .workflow-step.complete span { color: #090908; background: #ffcb28; }
        .workflow-step.active {
            background: #ffcb28;
            box-shadow: 4px 4px 0 #090908;
        }
        .workflow-step.active span { color: #ffcb28; background: #090908; }
        .queue-bar {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: center;
            margin: 8px 0 14px;
            padding: 14px 16px;
            color: #ffffff;
            background: #090908;
            border-left: 10px solid #ffcb28;
        }
        .queue-bar strong { font-size: 1.05rem; }
        .queue-bar span { color: #d5d5cc; }
        .validation-summary {
            margin: 8px 0 14px;
            padding: 16px 18px;
            border: 2px solid #090908;
            box-shadow: 4px 4px 0 #090908;
        }
        .validation-summary.pass { background: #ffcb28; }
        .validation-summary.warning { background: #ffffff; border-left: 10px solid #ffcb28; }
        .validation-summary strong { display: block; font-size: 1.1rem; }
        .validation-summary p { margin: 3px 0 0; color: #4e4e47 !important; }
        div.stButton > button, div.stDownloadButton > button {
            min-height: 42px;
            border: 2px solid #090908;
            border-radius: 2px;
            font-size: 1rem;
            font-weight: 900;
            box-shadow: 3px 3px 0 #090908;
        }
        div.stButton > button[kind="primary"] {
            color: #090908;
            background: var(--coral);
            border-color: #090908;
        }
        div.stButton > button[kind="primary"]:hover {
            background: var(--coral-dark);
            border-color: #090908;
            transform: translate(2px, 2px);
            box-shadow: 1px 1px 0 #090908;
        }
        div.stButton > button[kind="secondary"] {
            color: #090908 !important;
            background: #ffcb28 !important;
            border-color: #090908 !important;
        }
        div.stButton > button[kind="secondary"] p,
        div.stButton > button[kind="secondary"] span {
            color: #090908 !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            color: #ffffff !important;
            background: #090908 !important;
        }
        div.stButton > button[kind="secondary"]:hover p,
        div.stButton > button[kind="secondary"]:hover span {
            color: #ffffff !important;
        }
        div.stDownloadButton > button {
            color: var(--navy);
            background: white;
            border: 2px solid #090908;
        }
        hr { border-color: var(--line) !important; }
        h1, h2, h3 { color: var(--ink) !important; }
        @media (max-width: 700px) {
            .main .block-container { padding-top: 2rem; }
            .hero { padding: 30px 24px; }
            .hero-grid { grid-template-columns: 1fr; }
            .hero .tagline { font-size: 1.08rem; }
            .workflow-steps { grid-template-columns: repeat(2, 1fr); }
            .st-key-review_workspace [data-testid="stColumn"]:first-child {
                position: static;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logo_data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <div class="brand-row"><span class="brand-mark">B</span> BUSYBEE / PO PILOT</div>
                    <h1>Less busywork.<br>More business.</h1>
                    <p class="tagline">Turn retailer purchase orders into validated, Excel-ready records.</p>
                    <p class="persona">Built for early-stage consumer-goods founders who are done manually transferring retailer POs into spreadsheets.</p>
                    <div class="privacy-pill">● &nbsp;Documents are processed in memory and never saved</div>
                </div>
                <div class="logo-card">
                    <img src="data:image/jpeg;base64,{logo_data}" alt="BusyBee logo: a yellow bee carrying a document">
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_dashboard()
    if st.session_state.get("save_message"):
        st.success(st.session_state.pop("save_message"))
    st.divider()
    st.markdown(
        """
        <div class="section-heading">
            <span class="eyebrow">NEW PURCHASE ORDER</span>
            <h2>Process a purchase order</h2>
            <p>Upload a document, review the extraction, then save it to your dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload purchase orders",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Select one or more PDF, PNG, JPG, or JPEG files.",
    )
    use_sample = st.checkbox(
        "Use synthetic sample",
        help="Loads clearly labeled synthetic data and does not call the AI API.",
    )

    if st.button("Extract purchase orders", type="primary"):
        if use_sample:
            st.session_state.extraction_queue = [
                {
                    "po": SYNTHETIC_SAMPLE,
                    "source": "synthetic",
                    "document": None,
                    "status": "needs review",
                }
            ]
            st.session_state.extraction_batch_id = st.session_state.get(
                "extraction_batch_id", 0
            ) + 1
            activate_queued_result(0)
            st.success("Synthetic sample loaded. No document was uploaded or sent to an API.")
        elif not uploaded_files:
            st.warning("Upload one or more PDFs or images, or select “Use synthetic sample.”")
        else:
            results = []
            failures = []
            progress = st.progress(0, text="Starting batch extraction…")
            for position, uploaded_file in enumerate(uploaded_files, start=1):
                progress.progress(
                    (position - 1) / len(uploaded_files),
                    text=f"Extracting {uploaded_file.name} ({position} of {len(uploaded_files)})…",
                )
                try:
                    file_bytes = uploaded_file.getvalue()
                    po = extract_po_with_ai(
                        file_bytes, uploaded_file.name, uploaded_file.type
                    )
                    results.append(
                        {
                            "po": po.model_dump(),
                            "source": "ai",
                            "document": {
                            "bytes": file_bytes,
                            "filename": uploaded_file.name,
                            "mime_type": uploaded_file.type,
                            },
                            "status": "needs review",
                        },
                    )
                except RuntimeError as exc:
                    failures.append(f"{uploaded_file.name}: {exc}")
                except (ValidationError, json.JSONDecodeError, ValueError):
                    failures.append(f"{uploaded_file.name}: AI response was not valid PO data.")
                except Exception:
                    failures.append(f"{uploaded_file.name}: extraction failed; please retry.")
            progress.empty()
            if results:
                st.session_state.extraction_queue = results
                st.session_state.extraction_batch_id = st.session_state.get(
                    "extraction_batch_id", 0
                ) + 1
                activate_queued_result(0)
                st.success(
                    f"Extracted {len(results)} of {len(uploaded_files)} documents. "
                    "Review each PO in the queue below."
                )
            if failures:
                st.error(
                    "Some documents could not be extracted:\n\n- " + "\n- ".join(failures)
                )

    if "extracted_po" not in st.session_state:
        return

    queue = st.session_state.get("extraction_queue", [])
    if queue:
        active_index = st.session_state.get("active_result_index", 0)
        selected_index = st.selectbox(
            "Review queue",
            options=list(range(len(queue))),
            index=active_index,
            format_func=lambda index: (
                f"{index + 1}. "
                f"{queue[index]['document']['filename'] if queue[index]['document'] else 'Synthetic sample'} "
                f"— {queue[index]['status']}"
            ),
            key=f"queue_selector_{st.session_state.get('extraction_batch_id', 0)}",
        )
        if selected_index != active_index:
            activate_queued_result(selected_index)
        st.markdown(
            f"""
            <div class="queue-bar">
                <strong>Reviewing {selected_index + 1} of {len(queue)}</strong>
                <span>{sum(item['status'] == 'saved' for item in queue)} saved · {sum(item['status'] != 'saved' for item in queue)} remaining</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        workflow_steps(4 if queue[selected_index]["status"] == "saved" else 2)

    po_data = st.session_state.extracted_po
    widget_key = (
        f"batch_{st.session_state.get('extraction_batch_id', 0)}_"
        f"po_{st.session_state.get('active_result_index', 0)}"
    )
    if st.session_state.get("result_source") == "synthetic":
        st.info("Synthetic sample data — for demonstration only, not a real purchase order.")

    st.markdown(
        """
        <div class="section-heading">
            <span class="eyebrow">HUMAN REVIEW</span>
            <h2>Compare document and extraction</h2>
            <p>Check every extracted value against the original without leaving PO Pilot.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    review_workspace = st.container(key="review_workspace")
    source_panel, review_panel = review_workspace.columns([1, 1.15], gap="large")

    with source_panel:
        st.markdown('<div class="review-label">01 · SOURCE DOCUMENT</div>', unsafe_allow_html=True)
        st.subheader("Original purchase order")
        document = st.session_state.get("source_document")
        if document:
            render_document_preview(document)
        else:
            st.info(
                "No original document is used in synthetic demo mode. "
                "Upload a PDF or image to see it here beside the extracted data."
            )

    with review_panel:
        st.markdown('<div class="review-label">02 · EXTRACTED DATA</div>', unsafe_allow_html=True)
        st.subheader("Review and correct")
        first, second = st.columns(2)
        with first:
            po_number = st.text_input(
                "PO number", value=po_data.get("po_number") or "", key=f"{widget_key}_number"
            )
            order_date = st.text_input(
                "Order date", value=po_data.get("order_date") or "", key=f"{widget_key}_date"
            )
            currency = st.text_input(
                "Currency", value=po_data.get("currency") or "", key=f"{widget_key}_currency"
            )
            subtotal = st.number_input(
                "Subtotal",
                value=(
                    float(po_data["subtotal"])
                    if po_data.get("subtotal") is not None
                    else None
                ),
                step=0.01,
                format="%.2f",
                placeholder="Not found",
                key=f"{widget_key}_subtotal",
            )
            tax = st.number_input(
                "Tax",
                value=float(po_data["tax"]) if po_data.get("tax") is not None else None,
                step=0.01,
                format="%.2f",
                placeholder="Not found",
                key=f"{widget_key}_tax",
            )
            discount = st.number_input(
                "Discount",
                value=(
                    float(po_data["discount"])
                    if po_data.get("discount") is not None
                    else None
                ),
                step=0.01,
                format="%.2f",
                placeholder="Not found",
                key=f"{widget_key}_discount",
            )
        with second:
            buyer = st.text_input(
                "Buyer/company", value=po_data.get("buyer") or "", key=f"{widget_key}_buyer"
            )
            ship_by_date = st.text_input(
                "Ship-by date",
                value=po_data.get("ship_by_date") or "",
                key=f"{widget_key}_ship_date",
            )
            order_total = st.number_input(
                "Order total",
                value=float(po_data.get("order_total") or 0.0),
                step=0.01,
                format="%.2f",
                key=f"{widget_key}_total",
            )
            shipping = st.number_input(
                "Shipping",
                value=(
                    float(po_data["shipping"])
                    if po_data.get("shipping") is not None
                    else None
                ),
                step=0.01,
                format="%.2f",
                placeholder="Not found",
                key=f"{widget_key}_shipping",
            )
            other_charges = st.number_input(
                "Other charges",
                value=(
                    float(po_data["other_charges"])
                    if po_data.get("other_charges") is not None
                    else None
                ),
                step=0.01,
                format="%.2f",
                placeholder="Not found",
                key=f"{widget_key}_other_charges",
            )

        summary = {
            "po_number": po_number,
            "buyer": buyer,
            "order_date": order_date,
            "ship_by_date": ship_by_date,
            "currency": currency,
            "subtotal": subtotal,
            "shipping": shipping,
            "tax": tax,
            "other_charges": other_charges,
            "discount": discount,
            "order_total": order_total,
        }

        st.subheader("Extracted line items")
        edited_items = st.data_editor(
            st.session_state.line_items,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Quantity": st.column_config.NumberColumn(format="%.2f"),
                "Unit Price": st.column_config.NumberColumn(format="$%.2f"),
                "Line Total": st.column_config.NumberColumn(format="$%.2f"),
            },
            key=f"{widget_key}_line_items",
        )

        st.subheader("Validation")
        warnings = validate_po(summary, edited_items)
        if warnings:
            st.markdown(
                f"""
                <div class="validation-summary warning">
                    <strong>{len(warnings)} issue{'s' if len(warnings) != 1 else ''} to resolve</strong>
                    <p>Compare the highlighted checks with the source document, then correct the fields above.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for warning in warnings:
                st.warning(warning)
        else:
            st.markdown(
                """
                <div class="validation-summary pass">
                    <strong>✓ Validation passed</strong>
                    <p>Totals reconcile and required fields are present. This PO is ready to save.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        save_column, download_column = st.columns(2)
        with save_column:
            if st.button(
                "Save approved PO",
                type="primary",
                disabled=bool(warnings),
                help="Resolve validation warnings before saving." if warnings else None,
                use_container_width=True,
                key=f"{widget_key}_save",
            ):
                source_document = st.session_state.get("source_document")
                source_filename = (
                    source_document["filename"] if source_document else "Synthetic sample"
                )
                try:
                    save_approved_po(summary, edited_items, source_filename)
                    queue = st.session_state.get("extraction_queue", [])
                    if queue:
                        queue[st.session_state.get("active_result_index", 0)]["status"] = "saved"
                    st.session_state.save_message = (
                        f"PO {po_number} was saved and added to this week's dashboard."
                    )
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error(
                        "That PO number is already saved. Delete the existing record first "
                        "or use a different PO number."
                    )
        with download_column:
            st.download_button(
                "Approve & Download CSV",
                data=csv_bytes(summary, edited_items),
                file_name=f"{po_number or 'purchase-order'}.csv",
                mime="text/csv",
                disabled=bool(warnings),
                help="Resolve validation warnings before approval." if warnings else None,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
