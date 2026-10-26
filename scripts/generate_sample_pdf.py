"""
Utility script to generate a sample contract PDF for quick testing and benchmarking.
"""

from pathlib import Path


def generate_sample_pdf(output_path: str = "sample_contract.pdf"):
    # Clean PDF 1.4 binary structure
    content_lines = [
        "BT",
        "/F1 14 Tf",
        "72 720 Td",
        "(ENTERPRISE CLOUD SERVICES AGREEMENT) Tj",
        "0 -28 Td",
        "/F1 11 Tf",
        "(1. PARTIES AND EFFECTIVE DATE) Tj",
        "0 -16 Td",
        "(This Agreement is entered between Acme Global Inc. and CloudProvider Technologies LLC.) Tj",
        "0 -24 Td",
        "(2. SCOPE OF SERVICES) Tj",
        "0 -16 Td",
        "(CloudProvider shall provide high-availability Kubernetes infrastructure and 24/7 monitoring.) Tj",
        "0 -24 Td",
        "(3. PAYMENT AND INVOICING TERMS) Tj",
        "0 -16 Td",
        "(All invoices are due and payable Net 30 days from the invoice issuance date.) Tj",
        "0 -16 Td",
        "(A late fee of 1.5% per month applies to overdue balances exceeding 45 days.) Tj",
        "0 -24 Td",
        "(4. TERMINATION AND CANCELLATION CLAUSE) Tj",
        "0 -16 Td",
        "(Either party may terminate this agreement without cause upon thirty (30) days prior written notice.) Tj",
        "0 -16 Td",
        "(Immediate termination is permitted in the event of a material breach uncured after 15 days.) Tj",
        "0 -24 Td",
        "(5. CONFIDENTIALITY AND DATA PROTECTION) Tj",
        "0 -16 Td",
        "(All customer proprietary data must be encrypted in transit using TLS 1.3 and at rest with AES-256.) Tj",
        "ET",
    ]
    stream_data = "\n".join(content_lines).encode("latin-1")
    stream_length = len(stream_data)

    pdf_parts = []
    pdf_parts.append(b"%PDF-1.4\n")

    # Object 1: Catalog
    offset_1 = sum(len(p) for p in pdf_parts)
    pdf_parts.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    # Object 2: Pages
    offset_2 = sum(len(p) for p in pdf_parts)
    pdf_parts.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

    # Object 3: Page
    offset_3 = sum(len(p) for p in pdf_parts)
    pdf_parts.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )

    # Object 4: Stream
    offset_4 = sum(len(p) for p in pdf_parts)
    pdf_parts.append(
        f"4 0 obj\n<< /Length {stream_length} >>\nstream\n".encode("latin-1")
        + stream_data
        + b"\nendstream\nendobj\n"
    )

    # Object 5: Font
    offset_5 = sum(len(p) for p in pdf_parts)
    pdf_parts.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    # xref table
    xref_offset = sum(len(p) for p in pdf_parts)
    xref = (
        f"xref\n"
        f"0 6\n"
        f"0000000000 65535 f \n"
        f"{offset_1:010d} 00000 n \n"
        f"{offset_2:010d} 00000 n \n"
        f"{offset_3:010d} 00000 n \n"
        f"{offset_4:010d} 00000 n \n"
        f"{offset_5:010d} 00000 n \n"
        f"trailer\n"
        f"<< /Size 6 /Root 1 0 R >>\n"
        f"startxref\n"
        f"{xref_offset}\n"
        f"%%EOF\n"
    ).encode("latin-1")
    pdf_parts.append(xref)

    out_file = Path(output_path)
    out_file.write_bytes(b"".join(pdf_parts))
    print(f"Sample PDF successfully generated at: {out_file.resolve()}")


if __name__ == "__main__":
    generate_sample_pdf()
