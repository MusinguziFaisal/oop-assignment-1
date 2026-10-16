"""Interactive command-line front end: ``python -m src.microgrid_cli``."""
from src.microgrid import MicroGrid, CostModel, prompt_demand


def main() -> None:
    grid = MicroGrid()
    info = grid.check_well_posed()
    print(f"System OK: det={info['determinant']:.2f}, cond={info['condition_number']:.2f}")
    d1 = prompt_demand("Daytime load D1 (kWh): ")
    d2 = prompt_demand("Critical-equipment load D2 (kWh): ")
    report = grid.handle_infeasible([[d1], [d2]])
    x, y = report.handled[:, 0]
    if report.n_infeasible:
        print("Exact solution needs a negative source; using non-negative least squares instead.")
        print(f"Unmet load (kWh): D1 {report.unmet[0, 0]:.2f}, D2 {report.unmet[1, 0]:.2f}")
    cost = CostModel().daily_cost(report.handled)[0]
    print(f"Solar x = {x:.2f} kWh, battery y = {y:.2f} kWh, cost = UGX {cost:,.0f}")


if __name__ == "__main__":
    main()
