"""Gradio demo: python app.py"""

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from flat_price.location import fill_location, load_location_table
from flat_price.plots import CITY_NAMES
from flat_price.predictor import PricePredictor

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "flat-price"
predictor = PricePredictor.load(MODEL_DIR)
locations = load_location_table(MODEL_DIR / "locations.csv")

CITIES = {CITY_NAMES.get(c, c.capitalize()): c for c in sorted(locations["city"].unique())}
TYPES = {
    "Block of flats": "blockOfFlats",
    "Apartment building": "apartmentBuilding",
    "Tenement": "tenement",
    "Unknown": None,
}
CONDITIONS = {"Unknown": None, "Premium": "premium", "Low (needs renovation)": "low"}
MATERIALS = {"Unknown": None, "Brick": "brick", "Concrete slab": "concreteSlab"}
AMENITIES = ["Parking space", "Balcony", "Elevator", "Security", "Storage room"]
AMENITY_COLUMNS = ["hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom"]
# median absolute percentage error of the model on the test set
METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text())
TYPICAL_ERROR = METRICS["neural network"]["MdAPE"] / 100


def estimate(
    city,
    area,
    rooms,
    floor,
    floor_count,
    build_year,
    building_type,
    condition,
    material,
    centre_distance,
    amenities,
):
    floor = floor or 1
    row = {
        "city": CITIES[city],
        "squareMeters": area,
        "rooms": rooms,
        "floor": floor,
        "floorCount": max(floor_count or floor, floor),
        "buildYear": build_year or None,
        "centreDistance": centre_distance,
        "monthIndex": predictor.latest_month_index,
        "type": TYPES[building_type],
        "ownership": "condominium",
        "buildingMaterial": MATERIALS[material],
        "condition": CONDITIONS[condition],
        **{
            col: "yes" if name in amenities else "no"
            for name, col in zip(AMENITIES, AMENITY_COLUMNS, strict=True)
        },
        **fill_location(CITIES[city], centre_distance, locations),
    }
    price = float(predictor.predict(pd.DataFrame([row]))[0])
    low, high = price * (1 - TYPICAL_ERROR), price * (1 + TYPICAL_ERROR)
    return (
        f"## {price:,.0f} PLN\n"
        f"**{price / area:,.0f} PLN/m²** · typical range {low:,.0f} – {high:,.0f} PLN\n\n"
        f"Half of the test set predictions were within ±{TYPICAL_ERROR:.0%} of the listed price."
    ).replace(",", " ")


with gr.Blocks(title="flat-price-nn") as demo:
    gr.Markdown(
        "# Apartment price estimator\n"
        "Neural network written from scratch in NumPy, trained on 65k listings from 15 Polish "
        "cities (Otodom, Aug 2023 – Jun 2024)."
    )
    with gr.Row():
        with gr.Column():
            city = gr.Dropdown(list(CITIES), value="Kraków", label="City")
            centre = gr.Slider(0, 15, value=3, step=0.5, label="Distance to city centre [km]")
            area = gr.Slider(25, 150, value=55, step=1, label="Area [m²]")
            rooms = gr.Slider(1, 6, value=3, step=1, label="Rooms")
            with gr.Row():
                floor = gr.Number(value=3, precision=0, minimum=1, label="Floor")
                floor_count = gr.Number(value=5, precision=0, minimum=1, label="Floors in building")
                year = gr.Number(value=2005, precision=0, label="Year built (0 = unknown)")
        with gr.Column():
            building_type = gr.Radio(list(TYPES), value="Block of flats", label="Building type")
            material = gr.Radio(list(MATERIALS), value="Unknown", label="Building material")
            condition = gr.Radio(list(CONDITIONS), value="Unknown", label="Condition")
            amenities = gr.CheckboxGroup(
                AMENITIES, value=["Balcony", "Elevator"], label="Amenities"
            )
            result = gr.Markdown()
    inputs = [
        city,
        area,
        rooms,
        floor,
        floor_count,
        year,
        building_type,
        condition,
        material,
        centre,
        amenities,
    ]
    for component in inputs:
        component.change(estimate, inputs, result)
    demo.load(estimate, inputs, result)

if __name__ == "__main__":
    demo.launch()
