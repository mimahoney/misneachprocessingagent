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
MINUTES_SAVED_PER_ORDER = 5
MINUTES_SAVED_PER_LINE_ITEM = 3


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
    stored_order_label = "PO" if len(orders) == 1 else "POs"
    st.markdown(
        f"""
        <div class="database-status">
            <span class="status-dot"></span>
            <strong>SQLite connected</strong>
            <span>{len(orders)} {stored_order_label} stored locally</span>
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
        return

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

    metric_columns = st.columns(5)
    metric_columns[0].metric("POs this week", f"{len(weekly):,}")
    metric_columns[1].metric(
        "Total order value",
        format_money(total_value, display_currency) if not weekly.empty else "—",
    )
    metric_columns[2].metric(
        "Total tax",
        format_money(total_tax, display_currency) if not weekly.empty else "—",
    )
    metric_columns[3].metric(
        "Total shipping",
        format_money(total_shipping, display_currency) if not weekly.empty else "—",
    )
    metric_columns[4].metric("Estimated time saved", f"{minutes_saved / 60:.1f} hrs")
    st.caption(
        "This week runs Monday through today. Time saved is estimated at 5 minutes per PO "
        "plus 3 minutes per line item."
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


def main() -> None:
    st.set_page_config(page_title="PO Pilot", page_icon="📦", layout="wide")
    initialize_database()
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #64748b;
            --navy: #14213d;
            --navy-soft: #203354;
            --coral: #ff6b4a;
            --coral-dark: #e95738;
            --mint: #ccf4e2;
            --line: #e4e9f1;
            --surface: #ffffff;
            --canvas: #f4f7fb;
        }
        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(244, 247, 251, 0.92);
            backdrop-filter: blur(12px);
        }
        [data-testid="stDecoration"] { display: none; }
        .main .block-container {
            max-width: 1440px;
            padding-top: 4.5rem;
            padding-bottom: 5rem;
        }
        .stMarkdown, .stMarkdown p, .stCaption, label,
        [data-testid="stWidgetLabel"] p {
            color: var(--ink);
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 44px 48px;
            margin-bottom: 32px;
            color: white;
            background: linear-gradient(130deg, #111d36 0%, #1e3357 68%, #28506a 100%);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(20, 33, 61, 0.18);
        }
        .hero::after {
            content: "";
            position: absolute;
            width: 340px;
            height: 340px;
            right: -110px;
            top: -180px;
            border-radius: 50%;
            background: rgba(204, 244, 226, 0.12);
        }
        .brand-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 28px;
            font-weight: 750;
            letter-spacing: 0.02em;
        }
        .brand-mark {
            display: inline-grid;
            place-items: center;
            width: 34px;
            height: 34px;
            border-radius: 10px;
            color: var(--navy);
            background: var(--mint);
            font-size: 18px;
        }
        .hero h1 {
            margin: 0 0 12px;
            color: white !important;
            font-size: clamp(2.4rem, 5vw, 4.4rem);
            line-height: 0.98;
            letter-spacing: -0.055em;
        }
        .hero .tagline {
            max-width: 800px;
            margin: 0 0 18px;
            color: #e7eef9 !important;
            font-size: 1.35rem;
            line-height: 1.45;
        }
        .hero .persona {
            max-width: 880px;
            margin: 0;
            color: #b9c7dc !important;
            font-size: 0.98rem;
        }
        .privacy-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 24px;
            padding: 8px 12px;
            color: #dce8f8;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 999px;
            font-size: 0.82rem;
        }
        .section-heading { margin: 8px 0 20px; }
        .section-heading .eyebrow {
            color: var(--coral);
            font-size: 0.73rem;
            font-weight: 800;
            letter-spacing: 0.14em;
        }
        .section-heading h2 {
            margin: 4px 0 4px;
            color: var(--ink) !important;
            font-size: 1.8rem;
            letter-spacing: -0.025em;
        }
        .section-heading p { margin: 0; color: var(--muted) !important; }
        .database-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin: -6px 0 20px;
            padding: 8px 12px;
            color: #315047;
            background: #effaf5;
            border: 1px solid #d5eee3;
            border-radius: 999px;
            font-size: 0.8rem;
        }
        .database-status .status-dot {
            width: 8px;
            height: 8px;
            background: #24a673;
            border-radius: 50%;
            box-shadow: 0 0 0 3px rgba(36, 166, 115, 0.13);
        }
        .database-status span:last-child { color: #58736b; }
        .empty-state {
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 24px;
            margin-bottom: 14px;
            color: var(--ink);
            background: linear-gradient(110deg, #ffffff, #f0fbf6);
            border: 1px solid #dcece5;
            border-radius: 16px;
            box-shadow: 0 6px 20px rgba(23, 32, 51, 0.05);
        }
        .empty-state .empty-icon {
            display: grid;
            place-items: center;
            width: 44px;
            height: 44px;
            color: #126044;
            background: var(--mint);
            border-radius: 13px;
            font-size: 1.35rem;
        }
        .empty-state strong { font-size: 1rem; }
        .empty-state p { margin: 3px 0 0; color: var(--muted) !important; }
        [data-testid="stMetric"] {
            min-height: 124px;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 8px 22px rgba(23, 32, 51, 0.055);
        }
        [data-testid="stMetricLabel"] p { color: var(--muted) !important; }
        [data-testid="stMetricValue"] { color: var(--navy) !important; }
        [data-testid="stExpander"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(23, 32, 51, 0.045);
        }
        [data-testid="stFileUploader"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px;
        }
        [data-testid="stFileUploaderDropzone"] {
            color: var(--ink);
            background: #f8fafc;
            border: 1.5px dashed #b7c3d5;
            border-radius: 12px;
        }
        [data-testid="stFileUploaderDropzone"] * { color: var(--ink) !important; }
        [data-baseweb="input"], [data-baseweb="select"] > div {
            background: white !important;
            border-color: #d8e0eb !important;
        }
        [data-baseweb="input"] input { color: var(--ink) !important; }
        .st-key-review_workspace {
            padding: 18px;
            background: #e9eef5;
            border: 1px solid #d9e1ec;
            border-radius: 20px;
        }
        .st-key-review_workspace [data-testid="stColumn"] {
            padding: 18px;
            background: white;
            border: 1px solid var(--line);
            border-radius: 15px;
            box-shadow: 0 8px 22px rgba(23, 32, 51, 0.045);
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
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.1em;
        }
        div.stButton > button, div.stDownloadButton > button {
            min-height: 42px;
            border-radius: 10px;
            font-weight: 700;
        }
        div.stButton > button[kind="primary"] {
            color: white;
            background: var(--coral);
            border-color: var(--coral);
        }
        div.stButton > button[kind="primary"]:hover {
            background: var(--coral-dark);
            border-color: var(--coral-dark);
        }
        div.stDownloadButton > button {
            color: var(--navy);
            background: white;
            border: 1px solid #bcc8d8;
        }
        hr { border-color: var(--line) !important; }
        h1, h2, h3 { color: var(--ink) !important; }
        @media (max-width: 700px) {
            .main .block-container { padding-top: 2rem; }
            .hero { padding: 30px 24px; border-radius: 18px; }
            .hero .tagline { font-size: 1.08rem; }
            .st-key-review_workspace [data-testid="stColumn"]:first-child {
                position: static;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
            <div class="brand-row"><span class="brand-mark">P</span> PO PILOT</div>
            <h1>Purchase orders,<br>ready for takeoff.</h1>
            <p class="tagline">Turn retailer purchase orders into validated, Excel-ready records.</p>
            <p class="persona">Built for early-stage consumer-goods founders who are done manually transferring retailer POs into spreadsheets.</p>
            <div class="privacy-pill">● &nbsp;Documents are processed in memory and never saved</div>
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

    uploaded_file = st.file_uploader(
        "Upload a purchase order",
        type=SUPPORTED_TYPES,
        help="Accepted formats: PDF, PNG, JPG, and JPEG.",
    )
    use_sample = st.checkbox(
        "Use synthetic sample",
        help="Loads clearly labeled synthetic data and does not call the AI API.",
    )

    if st.button("Extract PO", type="primary"):
        if use_sample:
            load_result(PurchaseOrder.model_validate(SYNTHETIC_SAMPLE), "synthetic")
            st.success("Synthetic sample loaded. No document was uploaded or sent to an API.")
        elif uploaded_file is None:
            st.warning("Upload a PDF or image, or select “Use synthetic sample.”")
        else:
            try:
                with st.spinner("Reading the purchase order and extracting its fields…"):
                    file_bytes = uploaded_file.getvalue()
                    po = extract_po_with_ai(
                        file_bytes, uploaded_file.name, uploaded_file.type
                    )
                    load_result(
                        po,
                        "ai",
                        {
                            "bytes": file_bytes,
                            "filename": uploaded_file.name,
                            "mime_type": uploaded_file.type,
                        },
                    )
                st.success("Extraction complete. Review and edit the results below.")
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                st.error(
                    "The AI response could not be validated as purchase-order data. "
                    "Please try extraction again."
                )
                st.caption(str(exc))
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception:
                st.error(
                    "We couldn't extract this document. Check your API key and connection, "
                    "then try again. You can also use the synthetic sample."
                )

    if "extracted_po" not in st.session_state:
        return

    po_data = st.session_state.extracted_po
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
            po_number = st.text_input("PO number", value=po_data.get("po_number") or "")
            order_date = st.text_input("Order date", value=po_data.get("order_date") or "")
            currency = st.text_input("Currency", value=po_data.get("currency") or "")
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
            )
            tax = st.number_input(
                "Tax",
                value=float(po_data["tax"]) if po_data.get("tax") is not None else None,
                step=0.01,
                format="%.2f",
                placeholder="Not found",
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
            )
        with second:
            buyer = st.text_input("Buyer/company", value=po_data.get("buyer") or "")
            ship_by_date = st.text_input(
                "Ship-by date", value=po_data.get("ship_by_date") or ""
            )
            order_total = st.number_input(
                "Order total",
                value=float(po_data.get("order_total") or 0.0),
                step=0.01,
                format="%.2f",
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
            key="line_item_editor",
        )

        st.subheader("Validation")
        warnings = validate_po(summary, edited_items)
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No validation problems found.")

        save_column, download_column = st.columns(2)
        with save_column:
            if st.button(
                "Save approved PO",
                type="primary",
                disabled=bool(warnings),
                help="Resolve validation warnings before saving." if warnings else None,
                use_container_width=True,
            ):
                source_document = st.session_state.get("source_document")
                source_filename = (
                    source_document["filename"] if source_document else "Synthetic sample"
                )
                try:
                    save_approved_po(summary, edited_items, source_filename)
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
