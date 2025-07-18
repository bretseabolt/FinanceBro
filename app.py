import streamlit as st

main_page = st.Page(
    "ui_pages/main_page.py",
    title="Finance Wizard",
    default=True,
    icon="💰"
)
dashboard_page = st.Page(
    "ui_pages/data_viewer.py",
    title="Finance Viewer",
    icon="📊"
)

visualization_page = st.Page(
    "ui_pages/visualization_page.py",
    title="Visualizations",
    icon="📈"
)

pages = [main_page, dashboard_page, visualization_page]

pg = st.navigation(pages)
pg.run()