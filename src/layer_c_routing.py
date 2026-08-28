"""
layer_c_routing.py
===================
Inventory-Routing Problem (IRP) via decomposition:
    Layer A/B (s,S) policy -> delivery requests -> Capacitated VRP -> routes
    -> realized delivery cost -> fed back into Layer A's fixed order cost K
    -> re-solve -> repeat until stable (2-3 iterations typically).

VRP solve uses Google OR-Tools (PATH_CHEAPEST_ARC + capacity dimension).
Vehicle capacity parsing and haversine distance are used as-is.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGISTICS_PATH = PROJECT_ROOT / "data" / "raw" / "Transportation__Logistics_Tracking_Dataset.xlsx"

EARTH_RADIUS_KM = 6371.0
# SKU units have no mass in train.csv; 10 kg/unit makes (s,S) replenishments
# sit in the same order of magnitude as the MT figures on vehicle types.
KG_PER_UNIT = 10.0
COST_PER_KM = 1.50
FIXED_COST_PER_ROUTE = 50.0


def haversine_distance_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def parse_vehicle_capacity_tonnes(vehicle_type: str) -> float | None:
    """
    Extract a tonnage figure from free-text Vehicle Type strings such as
    '32 FT Multi-Axle 14MT - HCV' -> 14.0, '20 FT SXL Container' -> None
    (no explicit tonnage in the name - would need a lookup table for these).
    Returns None when no tonnage pattern is found; caller should decide a
    fallback (e.g. median known capacity) rather than silently guessing.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*MT", str(vehicle_type), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def build_delivery_requests(layer_a_policies: pd.DataFrame,
                             current_inventory: pd.DataFrame,
                             day: pd.Timestamp) -> pd.DataFrame:
    """
    For a given `day`, compare `current_inventory` (store, item, on_hand)
    against each pair's reorder point `s` from `layer_a_policies`. Any pair
    with on_hand <= s becomes a delivery request of quantity (S - on_hand).

    Expected output columns: [store, item, quantity, requested_date]
    This feeds directly into build_vrp_model() below.
    """
    policies = layer_a_policies[["store", "item", "s", "S"]].copy()
    inventory = current_inventory[["store", "item", "on_hand"]].copy()
    merged = inventory.merge(policies, on=["store", "item"], how="inner")
    if merged.empty:
        logger.warning("No overlapping (store, item) pairs between inventory and policies")
        return pd.DataFrame(columns=["store", "item", "quantity", "requested_date"])

    # Never-order policies use s = -1; they must not generate a request.
    triggered = merged[(merged["on_hand"] <= merged["s"]) & (merged["s"] >= 0)].copy()
    triggered["quantity"] = (triggered["S"] - triggered["on_hand"]).clip(lower=0).astype(int)
    triggered = triggered[triggered["quantity"] > 0]
    triggered["requested_date"] = pd.Timestamp(day)
    out = triggered[["store", "item", "quantity", "requested_date"]].reset_index(drop=True)
    logger.info(
        "Delivery requests for %s: %d SKUs across %d stores (from %d inventory rows)",
        pd.Timestamp(day).date(), len(out), out["store"].nunique() if len(out) else 0, len(merged),
    )
    return out


def _distance_matrix_meters(latitudes, longitudes) -> list[list[int]]:
    n = len(latitudes)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            km = haversine_distance_km(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            meters = int(round(km * 1000))
            matrix[i][j] = meters
            matrix[j][i] = meters
    return matrix


def _aggregate_store_demand_kg(delivery_requests: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        delivery_requests.groupby("store", as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "units"})
    )
    grouped["demand_kg"] = (grouped["units"] * KG_PER_UNIT).round().astype(int)
    return grouped


def _split_oversize_stops(store_demand: pd.DataFrame, max_capacity_kg: int) -> pd.DataFrame:
    """Split a store whose demand exceeds the largest truck into multiple drops."""
    rows = []
    for _, row in store_demand.iterrows():
        remaining = int(row["demand_kg"])
        drop = 0
        while remaining > 0:
            chunk = remaining if remaining <= max_capacity_kg else max_capacity_kg
            rows.append({
                "store": int(row["store"]),
                "drop": drop,
                "units": int(round(chunk / KG_PER_UNIT)),
                "demand_kg": chunk,
            })
            remaining -= chunk
            drop += 1
    return pd.DataFrame(rows)


def _extract_routes(manager, routing, solution, labels, demands_kg, capacities_kg):
    routes = []
    stop_rows = []
    for vehicle_id in range(routing.vehicles()):
        index = routing.Start(vehicle_id)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue
        path_labels = []
        load_kg = 0
        dist_m = 0
        seq = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            path_labels.append(labels[node])
            load_kg += demands_kg[node]
            stop_rows.append({
                "vehicle": vehicle_id,
                "seq": seq,
                "stop": labels[node],
                "demand_kg": demands_kg[node],
                "cumul_load_kg": load_kg,
            })
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            dist_m += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            seq += 1
        end_node = manager.IndexToNode(index)
        path_labels.append(labels[end_node])
        stop_rows.append({
            "vehicle": vehicle_id,
            "seq": seq,
            "stop": labels[end_node],
            "demand_kg": 0,
            "cumul_load_kg": load_kg,
        })
        dist_km = dist_m / 1000.0
        routes.append({
            "vehicle": vehicle_id,
            "capacity_kg": capacities_kg[vehicle_id],
            "stops": path_labels,
            "n_stops": max(len(path_labels) - 2, 0),
            "load_kg": load_kg,
            "distance_km": round(dist_km, 2),
            "cost": round(FIXED_COST_PER_ROUTE + COST_PER_KM * dist_km, 2),
        })
    return routes, pd.DataFrame(stop_rows)


def build_vrp_model(delivery_requests: pd.DataFrame,
                     store_coordinates: pd.DataFrame,
                     depot_coordinates: tuple[float, float],
                     vehicle_capacities: list[float],
                     time_limit_seconds: int = 8):
    """
    Build and solve a Capacitated VRP using OR-Tools.

      1. Build a distance matrix via haversine_distance_km() between the
         depot and every store with a nonzero delivery request that day.
      2. Use ortools.constraint_solver.routing:
         - RoutingIndexManager(num_locations, num_vehicles, depot_index)
         - RoutingModel + distance callback (registered via
           routing.RegisterTransitCallback)
         - AddDimensionWithVehicleCapacity for the tonnage constraint,
           using parse_vehicle_capacity_tonnes() output as vehicle capacities
           and aggregated delivery quantity (converted to kg) as demand
           per stop.
      3. Solve with routing.SolveWithParameters(search_parameters),
         first_solution_strategy = PATH_CHEAPEST_ARC as a fast baseline.
      4. Extract routes + total distance -> convert to a realized cost
         (e.g. $/km * distance + per-route fixed cost) -> this is the
         feedback signal for Layer A's K parameter.

    Reference: https://developers.google.com/optimization/routing/cvrp
    """
    if delivery_requests is None or delivery_requests.empty:
        logger.info("No delivery requests — skipping VRP")
        return {
            "status": "NO_REQUESTS",
            "routes": [],
            "stops": pd.DataFrame(),
            "total_distance_km": 0.0,
            "total_cost": 0.0,
            "n_vehicles_used": 0,
        }

    coords = store_coordinates[["store", "latitude", "longitude"]].drop_duplicates("store")
    store_demand = _aggregate_store_demand_kg(delivery_requests)
    store_demand = store_demand.merge(coords, on="store", how="inner")
    missing = set(delivery_requests["store"].unique()) - set(store_demand["store"].unique())
    if missing:
        raise ValueError(f"store_coordinates missing stores with requests: {sorted(missing)}")

    capacities_t = [float(c) for c in vehicle_capacities if c is not None and c > 0]
    if not capacities_t:
        raise ValueError("vehicle_capacities must contain at least one positive tonnage")
    capacities_kg = [int(round(c * 1000)) for c in capacities_t]
    max_cap_kg = max(capacities_kg)

    stops = _split_oversize_stops(store_demand, max_cap_kg)
    stops = stops.merge(coords, on="store", how="left")

    while len(capacities_kg) < len(stops):
        capacities_kg.append(max_cap_kg)

    latitudes = [float(depot_coordinates[0])] + stops["latitude"].astype(float).tolist()
    longitudes = [float(depot_coordinates[1])] + stops["longitude"].astype(float).tolist()
    labels = ["depot"] + [
        f"store {int(r.store)}" + (f" (drop {int(r.drop) + 1})" if int(r.drop) else "")
        for r in stops.itertuples()
    ]
    demands_kg = [0] + stops["demand_kg"].astype(int).tolist()
    distance_m = _distance_matrix_meters(latitudes, longitudes)

    manager = pywrapcp.RoutingIndexManager(len(labels), len(capacities_kg), 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_m[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_index):
        return demands_kg[manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,
        capacities_kg,
        True,
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(int(time_limit_seconds))

    logger.info(
        "Solving CVRP: %d stops, %d vehicles, total demand %.1f t",
        len(stops), len(capacities_kg), sum(demands_kg) / 1000.0,
    )
    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        logger.warning("VRP solver returned no solution")
        return {
            "status": "INFEASIBLE",
            "routes": [],
            "stops": pd.DataFrame(),
            "total_distance_km": 0.0,
            "total_cost": 0.0,
            "n_vehicles_used": 0,
        }

    routes, stops_df = _extract_routes(
        manager, routing, solution, labels, demands_kg, capacities_kg
    )
    total_km = float(sum(r["distance_km"] for r in routes))
    total_cost = float(sum(r["cost"] for r in routes))
    result = {
        "status": "SOLVED",
        "routes": routes,
        "stops": stops_df,
        "total_distance_km": round(total_km, 2),
        "total_cost": round(total_cost, 2),
        "n_vehicles_used": len(routes),
        "cost_per_km": COST_PER_KM,
        "fixed_cost_per_route": FIXED_COST_PER_ROUTE,
        "kg_per_unit": KG_PER_UNIT,
    }
    logger.info(
        "VRP solved: %d routes, %.1f km, cost %.2f",
        result["n_vehicles_used"], result["total_distance_km"], result["total_cost"],
    )
    return result


def get_store_coordinates(path: Path = LOGISTICS_PATH) -> pd.DataFrame:
    """
    Real coordinates available in the logistics dataset, keyed by location
    name (NOT by the Kaggle store 1-10 IDs - no shared key exists, same
    caveat as lead time in data_prep.py). Use this as a REALISTIC coordinate
    pool to assign to the 10 abstract store IDs (e.g. by sampling 10 distinct
    destination locations), documented as a modeling choice, not real GPS
    per Kaggle store.
    """
    df = pd.read_excel(path, sheet_name="Primary Data")
    coords = df[[
        "Destination Location", "Destination Location Latitude", "Destination Location Longitude"
    ]].drop_duplicates(subset=["Destination Location"]).dropna()
    coords.columns = ["location_name", "latitude", "longitude"]
    logger.info("Loaded %d distinct destination coordinates from logistics dataset", len(coords))
    return coords


if __name__ == "__main__":
    logger.info("layer_c_routing: helpers + OR-Tools CVRP solver")
