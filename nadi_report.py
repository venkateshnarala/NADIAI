"""
report.py
NADI AI - PDF Report Assembly Module
--------------------------------------
Builds the full technical PDF report from the outputs of data_collec.py,
quality.py, statisticaltests.py, distfit.py and ratingcurve.py, using
plot.py for all figures.

Uses reportlab Platypus (SimpleDocTemplate + Frame/canvas onPage callbacks)
so every page automatically gets the "NADI AI | <Station Name>" header and
a page-number footer, without having to remember to add it in every section.
"""

import os
import tempfile

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

import nadi_plot as pl
import nadi_quality as ql
import nadi_statisticaltests as st
import nadi_distfit as df
import nadi_ratingcurve as rc
import nadi_data_collec as dc

# Path to the NADI AI logo used on the title page and thank-you page.
# If the file is missing, the report still builds fine (logo is skipped).

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "NADI AI LOGO.jpg"

# ---------------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------------
PRIMARY_BLUE = colors.HexColor("#0B5394")
LIGHT_BLUE = colors.HexColor("#3D85C6")
HEADER_BG = colors.HexColor("#D9E7F5")
TABLE_GRID = colors.HexColor("#A9C4E0")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="NadiTitle", fontSize=26, leading=30, alignment=TA_CENTER,
                           textColor=PRIMARY_BLUE, fontName="Helvetica-Bold", spaceAfter=6))
styles.add(ParagraphStyle(name="NadiSubtitle", fontSize=13, leading=16, alignment=TA_CENTER,
                           textColor=colors.HexColor("#444444"), spaceAfter=4))
styles.add(ParagraphStyle(name="SectionHeading", fontSize=15, leading=18,
                           textColor=PRIMARY_BLUE, fontName="Helvetica-Bold",
                           spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle(name="SubHeading", fontSize=12, leading=15,
                           textColor=LIGHT_BLUE, fontName="Helvetica-Bold",
                           spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name="BodyJustify", fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY,
                           spaceAfter=6))
styles.add(ParagraphStyle(name="SmallNote", fontSize=8, leading=11,
                           textColor=colors.HexColor("#777777"), alignment=TA_LEFT))
styles.add(ParagraphStyle(name="WarningNote", fontSize=9, leading=12,
                           textColor=colors.HexColor("#CC4125"), fontName="Helvetica-Oblique",
                           spaceBefore=4, spaceAfter=6))
styles.add(ParagraphStyle(name="FormulaLine", fontSize=9.5, leading=14,
                           fontName="Courier", textColor=colors.HexColor("#1B4F72"),
                           leftIndent=18, spaceAfter=3))

# ---------------------------------------------------------------------------
# HEADER / FOOTER (drawn on every page except the title page)
# ---------------------------------------------------------------------------
class _HeaderFooterCanvas:
    """Callable used as onPage for SimpleDocTemplate to stamp header/footer."""

    def __init__(self, station_name):
        self.station_name = station_name

    def __call__(self, canvas, doc):
        canvas.saveState()
        width, height = A4
        # header
        canvas.setFillColor(PRIMARY_BLUE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(2 * cm, height - 1.3 * cm, "NADI AI")
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(width - 2 * cm, height - 1.3 * cm,
                                f"Station: {self.station_name}")
        canvas.setStrokeColor(LIGHT_BLUE)
        canvas.setLineWidth(0.8)
        canvas.line(2 * cm, height - 1.5 * cm, width - 2 * cm, height - 1.5 * cm)
        # footer
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawCentredString(width / 2, 1 * cm, f"Page {doc.page}")
        canvas.drawString(2 * cm, 1 * cm, "AI-generated report - verify before use")
        canvas.restoreState()

def _title_page_canvas(canvas, doc):
    """No header/footer on the title page itself."""
    pass

# ---------------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------------

def _fmt(val, decimals=2):
    """Format a numeric value safely, returning 'N/A' for NaN/None."""
    try:
        import numpy as np
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if isinstance(val, str):
            return val
        return f"{val:,.{decimals}f}"
    except Exception:
        return str(val)

def _std_table(data_rows, col_widths=None, header=True, font_size=8.5):
    """Build a styled reportlab Table from a list-of-lists."""
    t = Table(data_rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.5, TABLE_GRID),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style_cmds))
    return t

def _wrapping_table(data_rows, col_widths, header=True, font_size=7.2):
    """
    Build a table where every cell is a Paragraph, so long text (e.g. source
    citations like 'ESRI Land Cover (Karra et al., 2021)') wraps within its
    column instead of overflowing outside the table borders.
    """
    cell_style = ParagraphStyle(
        name="TableCell", fontSize=font_size, leading=font_size + 2.5,
        fontName="Helvetica", textColor=colors.black, wordWrap="CJK"
    )
    header_style = ParagraphStyle(
        name="TableCellHeader", fontSize=font_size, leading=font_size + 2.5,
        fontName="Helvetica-Bold", textColor=colors.white
    )
    wrapped_rows = []
    for r_idx, row in enumerate(data_rows):
        wrapped_row = []
        for cell in row:
            style = header_style if (header and r_idx == 0) else cell_style
            wrapped_row.append(Paragraph(_esc(str(cell)), style))
        wrapped_rows.append(wrapped_row)

    t = Table(wrapped_rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.5, TABLE_GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style_cmds += [("BACKGROUND", (0, 0), (-1, 0), PRIMARY_BLUE)]
    t.setStyle(TableStyle(style_cmds))
    return t

def _fig_to_image(fig, tmpdir, name, width=15.5 * cm):
    path = os.path.join(tmpdir, f"{name}.png")
    pl.save_fig(fig, path, dpi=150)
    img = Image(path, width=width, height=width * 0.55)
    img.hAlign = "CENTER"
    return img

def _esc(text):
    """Escape XML-special characters so reportlab's Paragraph parser doesn't
    misinterpret <, >, & (common in formula strings like 'x < y') as markup."""
    if text is None:
        return ""
    text = str(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _get_logo(width=3.2 * cm):
    """Return a reportlab Image of the NADI AI logo, or None if not found/unreadable."""
    try:
        if os.path.isfile(LOGO_PATH):
            img = Image(LOGO_PATH, width=width, height=width)
            img.hAlign = "CENTER"
            return img
    except Exception:
        pass
    return None

def _test_desc_block(desc):
    """Render a definition/purpose/formula/standard-values/assumptions block as flowables."""
    flow = [Paragraph(_esc(desc["title"]), styles["SubHeading"])]
    flow.append(Paragraph(f"<b>Definition:</b> {_esc(desc['definition'])}", styles["BodyJustify"]))
    flow.append(Paragraph(f"<b>Purpose:</b> {_esc(desc['purpose'])}", styles["BodyJustify"]))
    if "formula_lines" in desc:
        flow.append(Paragraph("<b>Formula:</b>", styles["BodyJustify"]))
        for line in desc["formula_lines"]:
            flow.append(Paragraph(_esc(line), styles["FormulaLine"]))
        flow.append(Spacer(1, 4))
    elif "formula" in desc:
        flow.append(Paragraph(f"<b>Formula:</b> {_esc(desc['formula'])}", styles["BodyJustify"]))
    if "standard_values" in desc:
        flow.append(Paragraph(f"<b>Standard values / significance level:</b> {_esc(desc['standard_values'])}", styles["BodyJustify"]))
    if "parameter_explanation" in desc:
        flow.append(Paragraph(f"<b>Parameter notes:</b> {_esc(desc['parameter_explanation'])}", styles["BodyJustify"]))
    if "assumptions" in desc:
        flow.append(Paragraph(f"<b>Assumptions:</b> {_esc(desc['assumptions'])}", styles["BodyJustify"]))
    return flow

# ---------------------------------------------------------------------------
# MAIN REPORT BUILDER
# ---------------------------------------------------------------------------

def generate_report(station_data, output_path):
    """
    station_data: dict returned by data_collec.get_station_data()
    output_path: full path (including .pdf) to write the report to

    Returns output_path on success.
    """
    tmpdir = tempfile.mkdtemp(prefix="nadiai_")
    station_name = station_data["meta"].get("cwc_site_name", "Unknown Station")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
        title=f"NADI AI Report - {station_name}", author="NADI AI"
    )

    story = []

    # ================= TITLE PAGE =================
    story += _build_title_page(station_data)
    story.append(PageBreak())

    # ================= 1. STATION INFORMATION =================
    story += _build_station_info(station_data)
    story.append(PageBreak())

    # ================= 2. OVERVIEW OF DATA =================
    story += _build_overview(station_data, tmpdir)
    story.append(PageBreak())

    sufficient = station_data["sufficient_data"]
    ams = station_data["ams"]
    ams_vals = ams["ann_max"].values if not ams.empty else []
    years = ams["year"].values if not ams.empty else []

    if sufficient:
        # ================= 3. OUTLIER DETECTION =================
        story += _build_outlier_section(ams, ams_vals, tmpdir)
        story.append(PageBreak())

        # ================= 4. CHANGE POINT DETECTION =================
        story += _build_changepoint_section(ams, ams_vals, years, tmpdir)
        story.append(PageBreak())

        # ================= 5. TREND TESTS (AMS, Annual Mean, Monsoon Mean) =================
        story += _build_trend_section(
            ams, ams_vals, tmpdir,
            series_key="ann_max",
            ylabel="Annual maximum discharge (m3/s)",
            heading="5. Mann-Kendall Trend Test - Annual Maximum Series (AMS)",
            series_desc="annual maximum flow",
        )
        story.append(PageBreak())

        annual_mean_flow = station_data.get("annual_mean_flow")
        monsoon_mean_flow = station_data.get("monsoon_mean_flow")
        if (annual_mean_flow is not None and not annual_mean_flow.empty) or \
           (monsoon_mean_flow is not None and not monsoon_mean_flow.empty):
            story += _build_additional_trend_sections(annual_mean_flow, monsoon_mean_flow, tmpdir)
            story.append(PageBreak())

        # ================= 8, 9, 10: DISTRIBUTION FITTING, GOF, DESIGN MAGNITUDES =================
        story += _build_distribution_sections(ams_vals, tmpdir)
        story.append(PageBreak())
    else:
        story.append(Paragraph("Statistical Tests and Frequency Analysis", styles["SectionHeading"]))
        story.append(Paragraph(
            "Sufficient data is not available for this station to perform outlier detection, "
            "change-point detection, trend analysis, or flood-frequency (distribution fitting) "
            "analysis. A minimum of 10 years of data, each with at least 50% daily data "
            "availability, is required. Only the data overview above is presented for this "
            "station.", styles["WarningNote"]))
        story.append(PageBreak())

    # ================= RATING CURVE (only for checked stations) =================
    story += _build_rating_curve_section(station_data, tmpdir)
    story.append(PageBreak())

    # ================= REFERENCES =================
    story += _build_references_page()
    story.append(PageBreak())

    # ================= THANK YOU =================
    story += _build_thankyou_page()

    header_footer = _HeaderFooterCanvas(station_name)

    # first page (title) gets no header/footer; subsequent pages do
    def _on_first_page(canvas, doc_):
        _title_page_canvas(canvas, doc_)

    def _on_later_pages(canvas, doc_):
        header_footer(canvas, doc_)

    doc.build(story, onFirstPage=_on_first_page, onLaterPages=_on_later_pages)
    return output_path

# ---------------------------------------------------------------------------
# SECTION BUILDERS
# ---------------------------------------------------------------------------

def _build_title_page(station_data):
    meta = station_data["meta"]
    flow = []
    flow.append(Spacer(1, 1.3 * cm))
    logo = _get_logo(width=3.4 * cm)
    if logo is not None:
        flow.append(logo)
        flow.append(Spacer(1, 0.5 * cm))
    flow.append(Paragraph("NADI AI", styles["NadiTitle"]))
    flow.append(Paragraph(
        "AI-Assisted Hydrological Analysis Report for Streamflow analysis "
        "and Design Flood Estimation", styles["NadiSubtitle"]))
    flow.append(Spacer(1, 1.2 * cm))
    flow.append(Paragraph(
        f"<b>Station:</b> {meta.get('cwc_site_name', 'N/A')}", styles["NadiSubtitle"]))
    flow.append(Paragraph(
        f"<b>River Basin:</b> {meta.get('river_basin', 'N/A')} &nbsp;|&nbsp; "
        f"<b>River:</b> {meta.get('cwc_river', 'N/A')}", styles["NadiSubtitle"]))
    flow.append(Spacer(1, 1.6 * cm))
    flow.append(Paragraph(
        "Developed by <b>Narala Venkatesh</b>, M.Tech Water Resources Engineering, "
        "NIT Warangal", styles["NadiSubtitle"]))
    flow.append(Spacer(1, 0.5 * cm))
    flow.append(Paragraph(
        "<i>NADI AI is currently under development and has not yet been fully "
        "validated by experienced hydrologists. Please interpret the results "
        "with caution and verify them before practical use.</i>",
        styles["NadiSubtitle"]))
    flow.append(Spacer(1, 1.4 * cm))
    flow.append(Paragraph("Acknowledgement", styles["SubHeading"]))
    flow.append(Paragraph(
        "NADI AI gratefully acknowledges the authors of the datasets used in "
        "this platform. Daily water level data were obtained from the dataset "
        "published by Solanki, H. and Mishra, V. (2026). Streamflow and "
        "catchment attribute data were sourced from the CAMELS-IND dataset "
        "developed by Mangukiya, N. K., Kumar, K. B., Dey, P., Sharma, S., "
        "Bejagam, V., Mujumdar, P. P., and Sharma, A. (2025). We sincerely "
        "appreciate the authors for making these high-quality datasets openly "
        "available, which have been instrumental in the development of NADI AI "
        "and continue to support hydrological research and applications.",
        styles["BodyJustify"]))
    flow.append(Spacer(1, 1.2 * cm))
    flow.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated with the assistance of an "
        "automated (AI-supported) analysis pipeline and may contain errors or "
        "omissions. Please review all results carefully before using them for any "
        "engineering, planning, or design decision.", styles["WarningNote"]))
    flow.append(Spacer(1, 0.6 * cm))
    flow.append(Paragraph(
        "For suggestions, corrections, or feedback, please write to: "
        "<b>venkateshnarala387@gmail.com</b><br/>"
        "LinkedIn: "
        "<b>www.linkedin.com/in/venkatesh-narala3</b>",
        styles["SmallNote"]))
    return flow

def _build_station_info(station_data):
    meta = station_data["meta"]
    land = station_data["land"]
    topo = station_data["topo"]
    flow = [Paragraph("1. Station Information", styles["SectionHeading"])]

    # ---- summary table ----
    summary_rows = [
        ["Attribute", "Value", "Attribute", "Value"],
        ["Station Name", str(meta.get("cwc_site_name", "N/A")),
         "River Basin", str(meta.get("river_basin", "N/A"))],
        ["River / Tributary", str(meta.get("cwc_river", "N/A")),
         f"Flow Availability (%) ({dc.FLOW_RECORD_PERIOD})", _fmt(meta.get("flow_availability"), 1)],
        ["CWC Latitude", _fmt(topo.get("cwc_lat"), 4),
         "CWC Longitude", _fmt(topo.get("cwc_lon"), 4)],
        ["Catchment Area - CWC (km2)", _fmt(topo.get("cwc_area"), 1),
         "", ""],
    ]
    flow.append(_std_table(summary_rows, col_widths=[4.6 * cm, 3.2 * cm, 4.6 * cm, 3.2 * cm]))
    flow.append(Spacer(1, 10))

    # ---- topography / land cover table ----
    flow.append(Paragraph("Catchment Topography and Land Cover", styles["SubHeading"]))
    topo_land_rows = [
        ["Attribute", "Value", "Attribute", "Value"],
        ["Elevation Mean (m)", _fmt(topo.get("elev_mean"), 1),
         "Elevation Median (m)", _fmt(topo.get("elev_median"), 1)],
        ["Elevation Min (m)", _fmt(topo.get("elev_min"), 1),
         "Elevation Max (m)", _fmt(topo.get("elev_max"), 1)],
        ["Slope Mean (%)", _fmt(topo.get("slope_mean"), 2),
         "Slope Median (%)", _fmt(topo.get("slope_median"), 2)],
        ["Slope Min (%)", _fmt(topo.get("slope_min"), 2),
         "Slope Max (%)", _fmt(topo.get("slope_max"), 2)],
        ["Water Fraction", _fmt(land.get("water_frac"), 3),
         "Trees Fraction", _fmt(land.get("trees_frac"), 3)],
        ["Crops Fraction", _fmt(land.get("crops_frac"), 3),
         "Built-up Fraction", _fmt(land.get("built_area_frac"), 3)],
        ["Bare Fraction", _fmt(land.get("bare_frac"), 3),
         "Range Fraction", _fmt(land.get("range_frac"), 3)],
        ["Dominant Land Cover", str(land.get("dom_land_cover", "N/A")),
         "Dominant LC Fraction", _fmt(land.get("dom_land_cover_frac"), 3)],
        ["LAI Mean", _fmt(land.get("lai_mean"), 2),
         "LAI Min , LAI Max", f"{_fmt(land.get('lai_min'), 2)} / {_fmt(land.get('lai_max'), 2)}"],
    ]
    flow.append(_std_table(topo_land_rows, col_widths=[4.2 * cm, 3.6 * cm, 4.2 * cm, 3.6 * cm]))
    flow.append(Spacer(1, 10))

    # ---- attribute description table (reference) ----
    flow.append(Paragraph("Attribute Definitions and Data Sources", styles["SubHeading"]))
    desc_rows = [["Attribute", "Description", "Source"]]
    key_order = ["gauge_id", "ghi_stn_id", "cwc_site_name", "river_basin", "cwc_river",
                 "flow_availability", "cwc_lat", "cwc_lon", "ghi_lat", "ghi_lon",
                 "elev_mean", "elev_median", "elev_min", "elev_max", "slope_mean",
                 "slope_median", "slope_min", "slope_max", "cwc_area", "ghi_area",
                 "gauge_elevation", "water_frac", "trees_frac", "flooded_veg_frac",
                 "crops_frac", "built_area_frac", "bare_frac", "range_frac",
                 "dom_land_cover", "dom_land_cover_frac", "lai_mean", "lai_min", "lai_max", "lai_diff"]
    for key in key_order:
        if key in dc.ATTRIBUTE_DESCRIPTIONS:
            label, unit, source = dc.ATTRIBUTE_DESCRIPTIONS[key]
            desc_rows.append([key, f"{label} ({unit})" if unit != "-" else label, source])
    tbl = _wrapping_table(desc_rows, col_widths=[3.0 * cm, 8.5 * cm, 4.5 * cm], font_size=7.2)
    flow.append(tbl)

    return flow

def _build_overview(station_data, tmpdir):
    flow = [Paragraph("2. Overview of Data", styles["SectionHeading"])]
    yearly_avail = station_data["yearly_avail"]
    usable_years = station_data["usable_years"]
    ams = station_data["ams"]
    basic_stats = station_data["basic_stats"]

    flow.append(Paragraph(
        f"Daily streamflow data (period of record: {dc.FLOW_RECORD_PERIOD}) were examined "
        f"for the full period of record. Years with at least 50% daily data availability "
        f"were retained as 'valid years for analysis' (a common minimum-completeness "
        f"threshold in hydrological practice). "
        f"<b>{len(usable_years)} valid year(s)</b> were identified"
        + (f": {', '.join(str(y) for y in usable_years)}." if usable_years else "."),
        styles["BodyJustify"]))
    flow.append(Paragraph(
        "<b>Note:</b> A year is considered valid for analysis only if at least 50% of its "
        "daily streamflow data is available. A minimum of 10 valid years is further "
        "required before outlier detection, change-point detection, trend analysis, or "
        "flood-frequency (distribution fitting) analysis is performed.", styles["WarningNote"]))

    if not yearly_avail.empty:
        flow.append(_fig_to_image(pl.plot_yearly_availability(yearly_avail), tmpdir, "avail"))
    flow.append(Spacer(1, 8))

    if not station_data["daily_usable"].empty:
        flow.append(Paragraph("Mean Monthly and Mean Annual Flow", styles["SubHeading"]))
        flow.append(_fig_to_image(pl.plot_monthly_mean_flow(station_data["monthly_mean_flow"]), tmpdir, "monthly"))
        flow.append(Spacer(1, 6))
        flow.append(_fig_to_image(pl.plot_annual_mean_flow(station_data["annual_mean_flow"]), tmpdir, "annualmean"))
        flow.append(Spacer(1, 8))

        flow.append(Paragraph("Flow Duration Curve (FDC)", styles["SubHeading"]))
        flow.append(_fig_to_image(pl.plot_fdc(station_data["fdc"]), tmpdir, "fdc"))
        flow.append(Spacer(1, 6))
        flow.append(Paragraph("Flow Duration Curve (Logarithmic Y-axis)", styles["SubHeading"]))
        flow.append(_fig_to_image(pl.plot_fdc_log(station_data["fdc"]), tmpdir, "fdc_log"))
        flow.append(Spacer(1, 8))

        fdc_table = dc.fdc_percentile_table(station_data["fdc"])
        if not fdc_table.empty:
            flow.append(Paragraph("Discharge at Standard Exceedance Percentiles", styles["SubHeading"]))
            fdc_rows = [["Exceedance Probability (%)", "Discharge (m3/s)"]]
            for _, r in fdc_table.iterrows():
                fdc_rows.append([f"{int(r['exceedance_prob'])}%", _fmt(r["discharge"], 2)])
            flow.append(_std_table(fdc_rows, col_widths=[7 * cm, 5 * cm]))
            flow.append(Spacer(1, 8))

    if not ams.empty:
        flow.append(Paragraph("Annual Maximum Series (AMS)", styles["SubHeading"]))
        flow.append(_fig_to_image(pl.plot_ams_series(ams), tmpdir, "ams"))
        flow.append(Spacer(1, 6))
        ams_stats = basic_stats["ams"]
        stat_rows = [
            ["Statistic", "Value (m3/s)"],
            ["Number of years (n)", str(ams_stats["n"])],
            ["Mean", _fmt(ams_stats["mean"])],
            ["Maximum", _fmt(ams_stats["max"])],
            ["Minimum", _fmt(ams_stats["min"])],
            ["Standard Deviation", _fmt(ams_stats["std"])],
            ["Coefficient of Variation", _fmt(ams_stats["cv"], 3)],
            ["Skewness", _fmt(ams_stats["skew"], 3)],
        ]
        flow.append(_std_table(stat_rows, col_widths=[7 * cm, 5 * cm]))

    # ---- Water level data availability (single plot, MSL) ----
    wl_avail = station_data.get("waterlevel_yearly_avail")
    if wl_avail is not None and not wl_avail.empty:
        flow.append(Spacer(1, 10))
        flow.append(Paragraph("Water Level Data Availability", styles["SubHeading"]))
        flow.append(Paragraph(
            "Water level (stage) observations at this station were converted to Mean Sea "
            "Level (MSL) datum. The plot below shows the annual data availability of the "
            "water level record, used for rating-curve construction where applicable.",
            styles["BodyJustify"]))
        flow.append(_fig_to_image(pl.plot_waterlevel_availability(wl_avail), tmpdir, "wl_avail"))

    if not station_data["sufficient_data"]:
        flow.append(Spacer(1, 8))
        flow.append(Paragraph(
            "Note: The number of valid years for analysis is below the minimum of 10 "
            "required for robust statistical testing and flood-frequency analysis. Only "
            "this data overview is presented for this station.", styles["WarningNote"]))

    return flow

def _build_outlier_section(ams, ams_vals, tmpdir):
    flow = [Paragraph("3. Outlier Detection Tests", styles["SectionHeading"])]

    # --- IQR ---
    iqr_res = ql.iqr_outlier_test(ams_vals)
    flow += _test_desc_block(ql.IQR_DESCRIPTION)
    iqr_rows = [
        ["Parameter", "Value"],
        ["Q1 (25th percentile)", _fmt(iqr_res["q1"])],
        ["Q3 (75th percentile)", _fmt(iqr_res["q3"])],
        ["IQR", _fmt(iqr_res["iqr"])],
        ["Lower Fence", _fmt(iqr_res["lower_fence"])],
        ["Upper Fence", _fmt(iqr_res["upper_fence"])],
        ["High Outliers Detected", str(len(iqr_res["high_outliers"]))],
        ["Low Outliers Detected", str(len(iqr_res["low_outliers"]))],
    ]
    flow.append(_std_table(iqr_rows, col_widths=[7 * cm, 5 * cm]))
    flow.append(Spacer(1, 6))
    flow.append(_fig_to_image(
        pl.plot_outliers(ams, iqr_res["low_outliers"], iqr_res["high_outliers"], "IQR Test"),
        tmpdir, "iqr"))
    if len(iqr_res["high_outliers"]) > 0 or len(iqr_res["low_outliers"]) > 0:
        flow.append(Paragraph(
            "Note: The IQR test flagged one or more potential outliers. Please cross-check "
            "these years manually against station records for data-entry errors or genuine "
            "extreme events.", styles["WarningNote"]))
    flow.append(Spacer(1, 12))

    # --- Grubbs-Beck ---
    gb_res = ql.grubbs_beck_test(ams_vals)
    flow += _test_desc_block(ql.GRUBBS_BECK_DESCRIPTION)
    gb_rows = [
        ["Parameter", "Value"],
        ["Sample Size (n)", str(gb_res["n"])],
        ["Mean of log10(Q)", _fmt(gb_res["log_mean"], 4)],
        ["Std. Dev. of log10(Q)", _fmt(gb_res["log_std"], 4)],
        ["Critical K value (alpha=0.10)", _fmt(gb_res["K_critical"], 4)],
        ["High Outliers Detected", str(len(gb_res["high_outliers"]))],
        ["Low Outliers Detected", str(len(gb_res["low_outliers"]))],
    ]
    flow.append(_std_table(gb_rows, col_widths=[7 * cm, 5 * cm]))
    if len(gb_res["high_outliers"]) > 0 or len(gb_res["low_outliers"]) > 0:
        flow.append(Paragraph(
            "Note: The Grubbs Beck test flagged one or more potential outliers. As per "
            "USGS Bulletin 17B/17C practice, please review these values manually before "
            "deciding whether to retain, adjust, or treat them as historical/censored data.",
            styles["WarningNote"]))
    return flow

def _build_changepoint_section(ams, ams_vals, years, tmpdir):
    flow = [Paragraph("4. Change Point Detection Tests", styles["SectionHeading"])]

    # --- Pettitt ---
    pt_res = ql.pettitt_test(ams_vals, years)
    flow += _test_desc_block(ql.PETTITT_DESCRIPTION)
    pt_rows = [
        ["Parameter", "Value"],
        ["K statistic", _fmt(pt_res["K_stat"], 2)],
        ["Approx. p-value", _fmt(pt_res["p_value"], 4)],
        ["Significance level (alpha)", "0.05"],
        ["Detected Change Year", str(pt_res["change_year"]) if pt_res["change_year"] else "N/A"],
        ["Statistically Significant?", "Yes" if pt_res["significant"] else "No"],
    ]
    flow.append(_std_table(pt_rows, col_widths=[7 * cm, 5 * cm]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph("Pettitt Test Statistic vs. Year", styles["SubHeading"]))
    flow.append(_fig_to_image(
        pl.plot_pettitt_statistic(years, pt_res.get("U_series", []), pt_res["change_year"]),
        tmpdir, "pettitt_stat"))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph("Annual Maximum Series with Change Point", styles["SubHeading"]))
    flow.append(_fig_to_image(pl.plot_pettitt_ams(ams, pt_res["change_year"]), tmpdir, "pettitt_ams"))
    if pt_res["significant"]:
        flow.append(Paragraph(
            f"Note: A statistically significant change point was detected around "
            f"{pt_res['change_year']}. Please cross-check this against any known catchment "
            f"changes (e.g., dam construction, land-use change, station relocation).",
            styles["WarningNote"]))
    flow.append(Spacer(1, 12))

    # --- CUSUM ---
    cs_res = ql.cusum_test(ams_vals, years)
    flow += _test_desc_block(ql.CUSUM_DESCRIPTION)
    cs_rows = [
        ["Parameter", "Value"],
        ["Maximum |CUSUM|", _fmt(cs_res["max_abs_cusum"], 2)],
        ["Indicated Change Year", str(cs_res["change_year"]) if cs_res["change_year"] else "N/A"],
    ]
    flow.append(_std_table(cs_rows, col_widths=[7 * cm, 5 * cm]))
    flow.append(Spacer(1, 6))
    flow.append(_fig_to_image(pl.plot_cusum(years, cs_res["cusum"], cs_res["change_year"]), tmpdir, "cusum"))
    flow.append(Paragraph(
        "Note: CUSUM is used here as a qualitative, exploratory cross-check alongside the "
        "Pettitt test. If both tests point to a similar year, this strengthens confidence "
        "that a genuine shift may be present; please verify manually.", styles["WarningNote"]))
    return flow

def _build_trend_section(series_df, series_vals, tmpdir, series_key, ylabel, heading, series_desc):
    """
    Generic Mann-Kendall trend section builder, reused for the AMS, annual
    mean flow, and monsoon (JJAS) mean flow series.

    series_df: DataFrame with columns 'year', <series_key>
    series_vals: the array of values (same as series_df[series_key].values)
    """
    flow = [Paragraph(heading, styles["SectionHeading"])]
    mk_res = st.mann_kendall_test(series_vals)
    flow += _test_desc_block(st.MANN_KENDALL_DESCRIPTION)
    mk_rows = [
        ["Parameter", "Value"],
        ["Trend", str(mk_res["trend"]).capitalize()],
        ["Significance level (alpha)", "0.05"],
        ["Statistically Significant?", "Yes" if mk_res["h"] else "No"],
        ["p-value", _fmt(mk_res["p"], 4)],
        ["Z statistic", _fmt(mk_res["z"], 3)],
        ["Kendall's Tau", _fmt(mk_res["Tau"], 3)],
        ["Sen's Slope (units/year)", _fmt(mk_res["slope"], 3)],
    ]
    flow.append(_std_table(mk_rows, col_widths=[7 * cm, 5 * cm]))
    flow.append(Spacer(1, 6))

    if mk_res.get("slope") is not None and mk_res["slope"] == mk_res["slope"]:  # not NaN
        flow.append(_fig_to_image(
            pl.plot_mann_kendall(series_df, mk_res["slope"], mk_res["intercept"], mk_res["trend"],
                                  value_col=series_key, ylabel=ylabel,
                                  title=heading.split(". ", 1)[-1]),
            tmpdir, f"mk_{series_key}_{heading[:2].strip()}"))
    flow.append(Spacer(1, 6))

    if mk_res["h"]:
        direction = "increasing" if mk_res["trend"] == "increasing" else "decreasing"
        flow.append(Paragraph(
            f"<b>Analysis:</b> The Mann-Kendall test indicates a statistically significant "
            f"{direction} trend in {series_desc} at this station (p = {_fmt(mk_res['p'],4)}), "
            f"with a Sen's slope of {_fmt(mk_res['slope'],3)} m3/s per year. This may have "
            f"implications for the assumption of stationarity in flood-frequency analysis and "
            f"should be considered when interpreting design flood estimates.", styles["BodyJustify"]))
        flow.append(Paragraph(
            "Note: The standard flood-frequency analysis presented in this report assumes "
            "the annual maximum series is stationary (i.e., its statistical properties do "
            "not change over time). Since a significant trend was detected in this series, "
            "this assumption may not strictly hold, and design flood estimates should be "
            "interpreted with appropriate caution.", styles["WarningNote"]))
    else:
        flow.append(Paragraph(
            f"<b>Analysis:</b> No statistically significant monotonic trend was detected in "
            f"{series_desc} at the 5% significance level (p = {_fmt(mk_res['p'],4)}). The "
            f"stationarity assumption underlying standard flood-frequency analysis therefore "
            f"appears reasonable for this dataset.", styles["BodyJustify"]))
        flow.append(Paragraph(
            "Note: The flood-frequency analysis presented in this report assumes the "
            "annual maximum series is stationary. This assumption is supported by the "
            "trend test result above.", styles["WarningNote"]))
    return flow

def _build_compact_trend_block(series_df, series_key, ylabel, sub_heading, series_desc, tmpdir, fig_tag):
    """
    Compact Mann-Kendall trend block (no definition/theory - that is already
    given once under the AMS trend test section). Just the sub-heading,
    results table, plot, and a short analysis line.
    """
    series_vals = series_df[series_key].values if series_df is not None and not series_df.empty else []
    flow = [Paragraph(sub_heading, styles["SubHeading"])]
    mk_res = st.mann_kendall_test(series_vals)
    mk_rows = [
        ["Parameter", "Value"],
        ["Trend", str(mk_res["trend"]).capitalize()],
        ["Significance level (alpha)", "0.05"],
        ["Statistically Significant?", "Yes" if mk_res["h"] else "No"],
        ["p-value", _fmt(mk_res["p"], 4)],
        ["Z statistic", _fmt(mk_res["z"], 3)],
        ["Kendall's Tau", _fmt(mk_res["Tau"], 3)],
        ["Sen's Slope (units/year)", _fmt(mk_res["slope"], 3)],
    ]
    flow.append(_std_table(mk_rows, col_widths=[7 * cm, 5 * cm]))
    flow.append(Spacer(1, 6))

    if mk_res.get("slope") is not None and mk_res["slope"] == mk_res["slope"]:  # not NaN
        flow.append(_fig_to_image(
            pl.plot_mann_kendall(series_df, mk_res["slope"], mk_res["intercept"], mk_res["trend"],
                                  value_col=series_key, ylabel=ylabel, title=sub_heading),
            tmpdir, fig_tag))
    flow.append(Spacer(1, 6))

    if mk_res["h"]:
        direction = "increasing" if mk_res["trend"] == "increasing" else "decreasing"
        flow.append(Paragraph(
            f"<b>Analysis:</b> The Mann-Kendall test indicates a statistically significant "
            f"{direction} trend in {series_desc} at this station (p = {_fmt(mk_res['p'],4)}), "
            f"with a Sen's slope of {_fmt(mk_res['slope'],3)} m3/s per year.", styles["BodyJustify"]))
    else:
        flow.append(Paragraph(
            f"<b>Analysis:</b> No statistically significant monotonic trend was detected in "
            f"{series_desc} at the 5% significance level (p = {_fmt(mk_res['p'],4)}).",
            styles["BodyJustify"]))
    flow.append(Spacer(1, 10))
    return flow

def _build_additional_trend_sections(annual_mean_flow, monsoon_mean_flow, tmpdir):
    """
    Section 5 continued: compact trend analyses for annual mean flow (5.1)
    and monsoon (JJAS) mean flow (5.2). No repeated Mann-Kendall definition/
    theory - that is presented once under the AMS trend test above.
    """
    flow = []
    if annual_mean_flow is not None and not annual_mean_flow.empty:
        flow += _build_compact_trend_block(
            annual_mean_flow, "mean_flow", "Mean annual flow (m3/s)",
            "5.1 Annual Mean Flow Trend Analysis", "annual mean flow",
            tmpdir, "mk_annual_mean")
    if monsoon_mean_flow is not None and not monsoon_mean_flow.empty:
        flow += _build_compact_trend_block(
            monsoon_mean_flow, "mean_flow", "Mean monsoon (Jun-Sep) flow (m3/s)",
            "5.2 Monsoon Mean Flow Trend Analysis", "monsoon (June-September) mean flow",
            tmpdir, "mk_monsoon_mean")
    return flow

def _build_distribution_sections(ams_vals, tmpdir):
    flow = [Paragraph("6. Distribution Fitting", styles["SectionHeading"])]
    results = df.fit_all_distributions(ams_vals)
    if results.empty:
        flow.append(Paragraph("Distribution fitting could not be completed for this dataset.",
                               styles["WarningNote"]))
        return flow

    flow.append(Paragraph(
        "Six candidate probability distributions commonly used in flood-frequency analysis "
        "were fitted to the annual maximum series using up to three parameter-estimation "
        "methods each: Method of Moments (MOM), L-Moments, and Maximum Likelihood Estimation "
        "(MLE).", styles["BodyJustify"]))
    fit_rows = [["Distribution", "Method", "Fitted Parameters"]]
    for _, row in results.iterrows():
        param_str = ", ".join(f"{k}={_fmt(v, 3)}" for k, v in row["params"].items())
        fit_rows.append([row["distribution"], row["method"], param_str])
    flow.append(_std_table(fit_rows, col_widths=[4 * cm, 3 * cm, 9 * cm], font_size=7.5))
    flow.append(PageBreak())

    # ---------------- 9. GOODNESS OF FIT ----------------
    flow.append(Paragraph("7. Goodness-of-Fit (GoF) Tests", styles["SectionHeading"]))
    flow.append(Paragraph(
        "Goodness-of-fit tests quantify how well each fitted distribution reproduces the "
        "observed annual maximum series. Five complementary tests are used so that both "
        "overall fit and tail behaviour (most relevant for design flood estimation) are "
        "assessed.", styles["BodyJustify"]))
    for key in ["KS", "Chi2", "AD", "AIC", "RMSE"]:
        d = df.GOF_DESCRIPTIONS[key]
        flow.append(Paragraph(_esc(d["title"]), styles["SubHeading"]))
        flow.append(Paragraph(f"<b>Definition:</b> {_esc(d['definition'])}", styles["BodyJustify"]))
        flow.append(Paragraph(f"<b>Purpose:</b> {_esc(d['purpose'])}", styles["BodyJustify"]))
        flow.append(Paragraph(f"<b>Interpretation:</b> {_esc(d['standard_values'])}", styles["BodyJustify"]))
    flow.append(Spacer(1, 8))

    flow.append(Paragraph("Ranking Table (Composite Rank across all 5 GoF Tests)", styles["SubHeading"]))
    rank_rows = [["Rank", "Distribution", "Method", "KS", "Chi2", "AD", "AIC", "RMSE"]]
    for _, row in results.iterrows():
        rank_rows.append([
            str(row["overall_rank"]), row["distribution"], row["method"],
            _fmt(row["ks_stat"], 4), _fmt(row["chi2_stat"], 2), _fmt(row["ad_stat"], 3),
            _fmt(row["aic"], 1), _fmt(row["rmse"], 2)
        ])
    flow.append(_std_table(rank_rows, col_widths=[1.3 * cm, 3.5 * cm, 2.2 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.8 * cm, 1.7 * cm], font_size=7))
    flow.append(PageBreak())

    top5 = results.head(5).reset_index(drop=True)
    sorted_vals, plotting_pos = df.get_plotting_positions(ams_vals)
    curves = df.build_distribution_curves(top5, ams_vals)
    flow.append(Paragraph("Top 5 Fitted Distributions vs. Observed Data", styles["SubHeading"]))
    flow.append(_fig_to_image(pl.plot_top_distributions(sorted_vals, plotting_pos, curves), tmpdir, "topdist"))
    flow.append(Spacer(1, 8))

    # ---- Performance metrics table for the top-5 fitted distributions ----
    flow.append(Paragraph("Performance Metrics - Top 5 Fitted Distributions", styles["SubHeading"]))
    perf_rows = [["Rank", "Distribution", "Method", "R\u00b2", "NSE", "RMSE", "MAE"]]
    for _, row in top5.iterrows():
        perf_rows.append([
            str(row["overall_rank"]), row["distribution"], row["method"],
            _fmt(row.get("r2"), 3), _fmt(row.get("nse"), 3),
            _fmt(row.get("rmse"), 2), _fmt(row.get("mae"), 2)
        ])
    flow.append(_std_table(perf_rows, col_widths=[1.3 * cm, 3.8 * cm, 2.5 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm], font_size=7.5))

    best = results.iloc[0]
    best_label = f"{best['distribution']} ({best['method']})"
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(
        f"<b>Best-fit distribution (lowest composite rank score):</b> {best_label}, with "
        f"KS = {_fmt(best['ks_stat'],4)}, AD = {_fmt(best['ad_stat'],3)}, "
        f"AIC = {_fmt(best['aic'],1)}, RMSE = {_fmt(best['rmse'],2)} m3/s, "
        f"R\u00b2 = {_fmt(best.get('r2'),3)}.", styles["BodyJustify"]))
    flow.append(PageBreak())

    # ---------------- 10. DESIGN MAGNITUDES ----------------
    flow.append(Paragraph("8. Design Flood Magnitudes", styles["SectionHeading"]))
    flow.append(Paragraph(
        "Using the top-5 ranked distribution/method combinations, discharge quantiles were "
        "estimated for standard return periods up to 1000 years.", styles["BodyJustify"]))
    qtable = df.estimate_quantiles(top5)
    labels = [f"{r['distribution']} ({r['method']})" for _, r in top5.iterrows()]
    q_rows = [["Return Period (yr)"] + labels]
    for _, r in qtable.iterrows():
        q_rows.append([str(int(r["return_period"]))] + [_fmt(r[l], 1) for l in labels])
    q_col_widths = [2.2 * cm] + [(15.5 - 2.2) / len(labels) * cm] * len(labels)
    flow.append(_wrapping_table(q_rows, col_widths=q_col_widths, font_size=6.8))
    flow.append(Spacer(1, 10))

    flow.append(Paragraph("Design Flood Magnitude vs. Return Period (Top 5 Distributions)", styles["SubHeading"]))
    flow.append(_fig_to_image(pl.plot_quantile_vs_return_period(qtable, labels), tmpdir, "quantiles"))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(f"Best-Fit Distribution Design Curve: {best_label}", styles["SubHeading"]))
    flow.append(_fig_to_image(pl.plot_best_fit_quantiles(qtable, best_label), tmpdir, "bestfit"))

    return flow

def _build_rating_curve_section(station_data, tmpdir):
    """
    Rating-curve section. Only stations in the checked RATING_CURVE_STATIONS
    list get a full fitted analysis with plots and performance metrics; all
    other stations get just the definition/method note (no fitting, no plots).
    """
    flow = [Paragraph("Rating Curve Analysis", styles["SectionHeading"])]
    flow.append(Paragraph(rc.RATING_CURVE_DEFINITION, styles["BodyJustify"]))

    has_rc = station_data.get("has_rating_curve", False)
    stage_discharge = station_data.get("stage_discharge")

    if not has_rc:
        flow.append(Paragraph(
            "A checked, reliable stage-discharge relationship is not currently available "
            "for this station, so a fitted rating curve is not presented in this report.",
            styles["WarningNote"]))
        return flow

    flow.append(Paragraph(
        "<b>Note:</b> Water level data used in this analysis were converted to Mean Sea "
        "Level (MSL) prior to fitting.", styles["WarningNote"]))

    if stage_discharge is None or stage_discharge.empty:
        flow.append(Paragraph(
            "No overlapping water level and discharge observations were found for this "
            "station, so a rating curve could not be fitted.", styles["WarningNote"]))
        return flow

    fit_results = rc.fit_rating_curves(stage_discharge)
    if fit_results.empty:
        flow.append(Paragraph(
            "Rating-curve fitting could not be completed for this station (insufficient "
            "overlapping data).", styles["WarningNote"]))
        return flow

    flow.append(Spacer(1, 8))
    flow.append(_fig_to_image(pl.plot_rating_curve(stage_discharge, fit_results, top_n=3), tmpdir, "rc_top3"))
    flow.append(Spacer(1, 8))

    flow.append(Paragraph("Performance Metrics - Top 3 Fitted Rating Curves", styles["SubHeading"]))
    perf_rows = [["Rank", "Equation", "Form", "R\u00b2", "NSE", "RMSE", "MAE"]]
    for _, row in fit_results.head(3).iterrows():
        perf_rows.append([
            str(row["rank"]), row["label"], row["form"],
            _fmt(row["r2"], 3), _fmt(row["nse"], 3), _fmt(row["rmse"], 2), _fmt(row["mae"], 2)
        ])
    flow.append(_std_table(perf_rows, col_widths=[1.3 * cm, 3.3 * cm, 4.7 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm], font_size=7))
    flow.append(Spacer(1, 10))

    best = fit_results.iloc[0]
    param_str = ", ".join(f"{k}={_fmt(v, 5)}" for k, v in best["params"].items() if k != "h_shift")
    flow.append(Paragraph("Final Selected Rating Curve Equation", styles["SubHeading"]))
    flow.append(Paragraph(
        f"<b>Selected form:</b> {best['label']} &nbsp; <b>Equation:</b> {best['form']} "
        f"&nbsp; <b>Fitted parameters:</b> {param_str}", styles["BodyJustify"]))
    flow.append(Paragraph(
        f"R\u00b2 = {_fmt(best['r2'],3)}, NSE = {_fmt(best['nse'],3)}, "
        f"RMSE = {_fmt(best['rmse'],2)} m3/s, MAE = {_fmt(best['mae'],2)} m3/s.",
        styles["BodyJustify"]))
    flow.append(_fig_to_image(pl.plot_best_rating_curve(stage_discharge, fit_results), tmpdir, "rc_best"))

    return flow

def _build_references_page():
    flow = [Paragraph("References", styles["SectionHeading"])]
    flow.append(Paragraph(
        "1. Mangukiya, N. K., Kumar, K. B., Dey, P., Sharma, S., Bejagam, V., Mujumdar, P. P., "
        "and Sharma, A.: CAMELS-INDIA: hydrometeorological time series and catchment "
        "attributes for 472 catchments in Peninsular India, Earth Syst. Sci. Data Discuss. "
        "[preprint], https://doi.org/10.5194/essd-2024-379, in review, 2024.", styles["BodyJustify"]))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(
        "2. Solanki, H., &amp; Mishra, V. (2026). Filling streamflow data gaps in Indian "
        "catchments using machine learning and K-means clustering. Water Resources "
        "Research, 62, e2025WR042210. https://doi.org/10.1029/2025WR042210. Data "
        "downloaded from: https://doi.org/10.5281/zenodo.18613693.", styles["BodyJustify"]))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph(
        "<b>Note:</b> This report is AI-generated and may contain errors. It has "
        "been developed by an M.Tech student and should be independently verified before use "
        "in any decision making process. For suggestions or to report issues, "
        "please contact venkateshnarala387@gmail.com.", styles["WarningNote"]))
    return flow

def _build_thankyou_page():
    flow = [Spacer(1, 4 * cm)]
    logo = _get_logo(width=3 * cm)
    if logo is not None:
        flow.append(logo)
        flow.append(Spacer(1, 0.6 * cm))
    flow.append(Paragraph("Thank You for Using NADI AI", styles["NadiTitle"]))
    flow.append(Spacer(1, 0.5 * cm))
    flow.append(Paragraph(
        "We hope this report supports your hydrological analysis. "
        "Your feedback is genuinely valued.",
        styles["NadiSubtitle"]))
    flow.append(Spacer(1, 1 * cm))
    flow.append(Paragraph(
        "Developed by Narala Venkatesh, M.Tech Water Resources Engineering, NIT Warangal",
        styles["NadiSubtitle"]))
    flow.append(Paragraph("venkateshnarala387@gmail.com", styles["NadiSubtitle"]))
    return flow
