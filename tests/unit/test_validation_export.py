
import csv

from gelation_sim.export import export_validation


def test_export_validation(tmp_path):
    """Check validation CSV files and their contents."""
    rows = [
        (
            "step_000000", 0.0, 0.01, 0.2, 0.0,
            10, 200, 0, 1.0, None, None, None,
            None, None,
        ),
    ]
    constants = [
        ("Seed", 42, None),
        ("kd", 0.001, "1/s"),
        ("Mn", None, "g/mol"),
    ]

    constants_path, equations_path = export_validation(
        rows, constants, tmp_path
    )

    with constants_path.open(newline="", encoding="utf-8") as handle:
        constant_data = list(csv.reader(handle))

    with equations_path.open(newline="", encoding="utf-8") as handle:
        equation_data = list(csv.reader(handle))

    assert constants_path.name == "validation_constants.csv"
    assert equations_path.name == "validation_equations.csv"
    assert constants_path.is_file()
    assert equations_path.is_file()

    assert constant_data == [
        ["Value", "", "Units"],
        ["Seed", "42", ""],
        ["kd", "0.001", "1/s"],
        ["Mn", "", "g/mol"],
    ]

    assert len(equation_data[0]) == 14
    assert equation_data[0][0] == ""
    assert equation_data[0][11] == (
        "Simulated polymerization rate (mol/(L*s))"
    )
    assert equation_data[0][12].startswith("Theoretical")
    assert equation_data[0][13].startswith("Theoretical")

    assert equation_data[1][0] == "step_000000"
    assert equation_data[1][1] == "0.0"
    assert equation_data[1][9:] == ["", "", "", "", ""]
