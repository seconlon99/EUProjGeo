import marimo

__generated_with = "0.13.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd# Load the dataset
    data_path = "./data/merged_organization_project_data.csv"
    data = pd.read_csv(data_path)
    return data, mo, pd


@app.cell
def _(data):
    data.columns
    return


@app.cell
def _(data, mo):
    domain_selector = mo.ui.multiselect(
        options=sorted(data["final_domain"].dropna().unique().tolist()),
        label="Domains"
    )

    method_selector = mo.ui.multiselect(
        options=sorted(data["final_method"].dropna().unique().tolist()),
        label="Methods"
    )


    return domain_selector, method_selector


@app.cell
def _(domain_selector, method_selector):
    selected_domains = domain_selector.value
    selected_methods = method_selector.value

    return selected_domains, selected_methods


@app.cell
def _(
    brics_countries,
    country_group_filter,
    data,
    eu_countries,
    nato_countries,
    selected_domains,
    selected_methods,
):
    filtered_data = data.copy()

    if selected_domains:
        filtered_data = filtered_data[filtered_data["final_domain"].isin(selected_domains)]

    if selected_methods:
        filtered_data = filtered_data[filtered_data["final_method"].isin(selected_methods)]

    selected_group = country_group_filter.value

    # Determine which countries to include
    if selected_group == "--":
        allowed_countries = filtered_data["iso3c"].unique()
    elif selected_group == "NATO":
        allowed_countries = nato_countries
    elif selected_group == "EU":
        allowed_countries = eu_countries
    elif selected_group == "BRICS":
        allowed_countries = brics_countries
    elif selected_group == "Non-NATO":
        allowed_countries = set(filtered_data["iso3c"].unique()) - nato_countries
    elif selected_group == "Non-EU":
        allowed_countries = set(filtered_data["iso3c"].unique()) - eu_countries
    else:
        allowed_countries = filtered_data["iso3c"].unique()

    # Filter rows
    filtered_data = filtered_data[filtered_data["iso3c"].isin(allowed_countries)]

    return (filtered_data,)


@app.cell
def _(mo):
    strength_slider = mo.ui.slider(
        start=0,
        stop=100,
        step=1,
        value=50,  # Default: top 50% strongest
        label="Minimum Collaboration Strength Percentile"
    )
    return (strength_slider,)


@app.cell
def _():
    nato_countries = {
        "USA", "CAN", "GBR", "FRA", "DEU", "ITA", "ESP", "POL", "NLD", "BEL",
        "TUR", "GRC", "NOR", "DNK", "CZE", "SVK", "HUN", "PRT", "ROU", "HRV",
        "BGR", "EST", "LVA", "LTU", "SVN", "ALB", "ISL", "LUX", "MNE", "MKD"
    }

    eu_countries = {
        "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
        "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
        "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"
    }

    brics_countries = {"BRA", "RUS", "IND", "CHN", "ZAF"}

    return brics_countries, eu_countries, nato_countries


@app.cell
def _(filtered_data, highlight_country, pd, selected_domains, strength_slider):
    import plotly.graph_objects as go
    import numpy as np

    # Compute ties between countries based only on the selected topic
    ties = {}
    for project_id, group in filtered_data.groupby("projectID"):
        countries = group["iso3c"].dropna().unique()
        if len(countries) <= 1:
            continue  # skip single-country projects
        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                pair = tuple(sorted([countries[i], countries[j]]))
                ties[pair] = ties.get(pair, 0) + 1

    # Country positions (mean coordinates by country)
    country_positions = filtered_data.groupby("iso3c")[["cultdist", "IdealPointAll"]].mean().reset_index()
    country_positions.columns = ["country", "cultdist", "IdealPointAll"]

    # Organization count per country
    country_org_counts = filtered_data.groupby("iso3c").size().reset_index(name="org_count")
    country_positions = country_positions.merge(country_org_counts, left_on="country", right_on="iso3c", how="left")
    country_positions["marker_size"] = np.log1p(country_positions["org_count"]) * 10

    # Random country color assignment
    country_colors = {
        country: f"rgba({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)}, 0.8)"
        for country in country_positions["country"]
    }

    # Construct edge dataframe

    threshold_percentile = strength_slider.value

    edge_df = pd.DataFrame([
        {"c1": c1, "c2": c2, "weight": w}
        for (c1, c2), w in ties.items()
    ])

    # Guard against empty or missing weights
    if not edge_df.empty and "weight" in edge_df.columns:
        threshold_percentile = strength_slider.value
        weight_threshold = np.percentile(edge_df["weight"], threshold_percentile)
        edge_df_filtered = edge_df[edge_df["weight"] >= weight_threshold]
    else:
        edge_df_filtered = pd.DataFrame(columns=["c1", "c2", "weight"])

    # Prepare edge coordinates for plotting
    edge_x = []
    edge_y = []
    edge_colors = []

    for _, row in edge_df_filtered.iterrows():
        c1, c2, weight = row["c1"], row["c2"], row["weight"]
        if c1 in country_positions["country"].values and c2 in country_positions["country"].values:
            x0 = country_positions[country_positions["country"] == c1]["cultdist"].values[0]
            y0 = country_positions[country_positions["country"] == c1]["IdealPointAll"].values[0]
            x1 = country_positions[country_positions["country"] == c2]["cultdist"].values[0]
            y1 = country_positions[country_positions["country"] == c2]["IdealPointAll"].values[0]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_colors.append(weight)

    # Normalize edge weights for coloring
    max_weight = max(edge_colors) if edge_colors else 1
    edge_colors_normalized = [w / max_weight for w in edge_colors]

    # Hover text setup (institution names and collaborations)
    institutions_by_country = (
        filtered_data.groupby("iso3c")["name"].nunique().to_dict()
    )

    collab_counts = {}
    for _, group in filtered_data.groupby("projectID"):
        orgs = group[["name", "iso3c"]].dropna().drop_duplicates()
        countries = orgs["iso3c"].unique()
        if len(countries) <= 1:
            continue
        for org in orgs["name"]:
            collab_counts[org] = collab_counts.get(org, 0) + 1

    hover_labels = []
    for _, row in country_positions.iterrows():
        iso = row["country"]
        institutions = filtered_data[filtered_data["iso3c"] == iso]["name"].dropna().unique()
        num_institutions = institutions_by_country.get(iso, 0)

        # Filter institutions with at least 1 international project
        inst_collabs = {
            inst: collab_counts.get(inst, 0)
            for inst in institutions
            if collab_counts.get(inst, 0) > 0
        }

        if inst_collabs:
            max_collab = max(inst_collabs.values())
            top_insts = [f"{inst} ({count})" for inst, count in inst_collabs.items() if count == max_collab]
            top_text = "<br>".join(top_insts)
            detail_text = f"Those with the most international Projects are:<br>{top_text}"
        else:
            detail_text = "No institutions with international Projects"

        label = (
            f"<b>{iso}</b><br>"
            f"{num_institutions} institution(s) are working on this topic.<br>"
            f"{detail_text}"
        )
        hover_labels.append(label)

    # ---- Plot ----
    fig = go.Figure()

    # Edges
    for i in range(len(edge_colors)):
        strength = edge_colors_normalized[i]
        color = f"rgba(0, 0, 0, {0.1 + 0.9 * strength})"  # Vary opacity only
        fig.add_trace(go.Scatter(
            x=edge_x[i * 3:i * 3 + 2],
            y=edge_y[i * 3:i * 3 + 2],
            line=dict(width=2, color=color),
            hoverinfo="none",
            mode="lines"
        ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=country_positions["cultdist"],
        y=country_positions["IdealPointAll"],
        mode="markers",
        marker=dict(
            size=country_positions["marker_size"],
            color=[country_colors[country] for country in country_positions["country"]],
            line=dict(width=2, color="black")
        ),
        hovertext=hover_labels,
        hoverinfo="text"
    ))

    # Layout
    fig.update_layout(
        title=f"Collaboration Network for Domain: {selected_domains}<br>Showing top {100 - threshold_percentile}% strongest ties",
        xaxis_title="Cultural Distance",
        yaxis_title="Geopolitical Alignment",
    showlegend=False,
    xaxis=dict(
        title="Cultural Distance from Western Liberal Norms",
        tickvals=[0.26, 0.36],
        ticktext=["Non Western Norms", "Towards Western Norms"],
        range=[0.25, 0.37],  # Force a consistent visual range
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        title="Geopolitical Alignment",
        tickvals=[country_positions["IdealPointAll"].min(), country_positions["IdealPointAll"].max()],
        ticktext=["Less US aligned", "More US aligned"])
    )

    # ---- Highlight Selected Country and Its Edges in Red ----
    highlight = highlight_country.value

    if highlight and highlight in country_positions["country"].values:
        # Highlight the selected node
        pos = country_positions[country_positions["country"] == highlight]
        fig.add_trace(go.Scatter(
            x=pos["cultdist"],
            y=pos["IdealPointAll"],
            mode="markers",
            marker=dict(
                size=pos["marker_size"].values[0] + 6,
                color="rgba(255,0,0,0.6)",
                line=dict(width=3, color="red")
            ),
            hoverinfo="skip",
            showlegend=False
        ))

        # Highlight connected edges
        for _, row in edge_df_filtered.iterrows():
            if highlight in [row["c1"], row["c2"]]:
                c1, c2 = row["c1"], row["c2"]
                if c1 in country_positions["country"].values and c2 in country_positions["country"].values:
                    x0 = country_positions[country_positions["country"] == c1]["cultdist"].values[0]
                    y0 = country_positions[country_positions["country"] == c1]["IdealPointAll"].values[0]
                    x1 = country_positions[country_positions["country"] == c2]["cultdist"].values[0]
                    y1 = country_positions[country_positions["country"] == c2]["IdealPointAll"].values[0]
                    fig.add_trace(go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(width=3, color="red"),
                        hoverinfo="none",
                        showlegend=False
                    ))

    fig.show()

    return


@app.cell
def _(domain_selector, method_selector, mo):
    mo.hstack([
        domain_selector, 
        method_selector,
    ])

    return


@app.cell
def _(country_group_filter, highlight_country, mo, strength_slider):
    mo.hstack([
        country_group_filter,
        highlight_country,
        strength_slider,
    ])
    return


@app.cell
def _(filtered_data, mo):
    visible_countries = sorted(filtered_data["iso3c"].dropna().unique())

    highlight_country = mo.ui.dropdown(
        options=visible_countries,
        label="Highlight Country"
    )

    return (highlight_country,)


@app.cell
def _(mo):
    country_group_filter = mo.ui.dropdown(
        options=["NATO", "EU", "BRICS", "Non-NATO", "Non-EU"],
        label="Show Countries"
    )

    return (country_group_filter,)


if __name__ == "__main__":
    app.run()
