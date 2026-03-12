"""Password gate for admin tabs."""
from __future__ import annotations

import streamlit as st


def check_admin_password() -> bool:
    """Render sidebar password field and return True if correct."""
    pwd = st.sidebar.text_input("Admin", type="password", key="admin_password")
    expected = st.secrets.get("ADMIN_PASSWORD", "ricky2026")
    return pwd == expected and pwd != ""
