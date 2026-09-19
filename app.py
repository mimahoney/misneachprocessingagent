"""PO Pilot: turn purchase orders into editable, validated CSV records."""

from __future__ import annotations

import base64
import json
import os
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
CURRENCY_TOLERANCE = 0.02
SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg"]


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
        "be a three-letter ISO code. Preserve each line item. Do not include markdown fences."
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

    po_total = as_number(summary["order_total"])
    line_totals = [as_number(value) for value in items.get("Line Total", [])]
    if po_total is not None and line_totals and all(value is not None for value in line_totals):
        line_sum = sum(value for value in line_totals if value is not None)
        if abs(line_sum - po_total) > CURRENCY_TOLERANCE:
            warnings.append(
                f"Sum of line totals ({line_sum:.2f}) does not match PO total ({po_total:.2f})."
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


def load_result(po: PurchaseOrder, source: str) -> None:
    st.session_state.extracted_po = po.model_dump()
    st.session_state.line_items = item_frame(po)
    st.session_state.result_source = source


def main() -> None:
    st.set_page_config(page_title="PO Pilot", page_icon="📦", layout="wide")
    st.title("PO Pilot")
    st.subheader("Turn retailer purchase orders into validated, Excel-ready records.")
    st.write(
        "Built for early-stage consumer-goods founders who manually transfer retailer "
        "purchase orders into spreadsheets."
    )
    st.caption("Uploads are processed in memory and are not logged or saved by this app.")

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
                    po = extract_po_with_ai(
                        uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type
                    )
                    load_result(po, "ai")
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

    st.header("Review PO summary")
    first, second, third = st.columns(3)
    with first:
        po_number = st.text_input("PO number", value=po_data.get("po_number") or "")
        order_date = st.text_input("Order date", value=po_data.get("order_date") or "")
    with second:
        buyer = st.text_input("Buyer/company", value=po_data.get("buyer") or "")
        ship_by_date = st.text_input("Ship-by date", value=po_data.get("ship_by_date") or "")
    with third:
        currency = st.text_input("Currency", value=po_data.get("currency") or "")
        order_total = st.number_input(
            "Order total",
            value=float(po_data.get("order_total") or 0.0),
            step=0.01,
            format="%.2f",
        )

    summary = {
        "po_number": po_number,
        "buyer": buyer,
        "order_date": order_date,
        "ship_by_date": ship_by_date,
        "currency": currency,
        "order_total": order_total,
    }

    st.header("Review line items")
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

    st.header("Validation")
    warnings = validate_po(summary, edited_items)
    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No validation problems found.")

    st.download_button(
        "Approve & Download CSV",
        data=csv_bytes(summary, edited_items),
        file_name=f"{po_number or 'purchase-order'}.csv",
        mime="text/csv",
        type="primary",
        disabled=bool(warnings),
        help="Resolve validation warnings before approval." if warnings else None,
    )


if __name__ == "__main__":
    main()
