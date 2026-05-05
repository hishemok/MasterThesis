from __future__ import annotations

from pathlib import Path

import retry2
from remake_majoranas4 import make_majoranas_for_B_and_C_with_projection_dim as make_v4_majoranas


OUTPUT_PATH = Path(__file__).with_name("retry2_remake4_diagonal_minus_dims_8_32_56_80_256_512.txt")
SEED_PATH = Path(__file__).with_name("retry2_remake4_diagonal_minus_dims_8_32_56_80.txt")
U_VALUES = [0.0, 0.1, 2.0]
PROJECTION_LEVELS = [8, 32, 56, 80, 256, 512]


def make_v4_diagonal_minus(*args, **kwargs):
    kwargs["tocheck"] = "Minus"
    kwargs["transition_mode"] = "diagonal"
    return make_v4_majoranas(*args, **kwargs)


def load_seeded_results():
    if OUTPUT_PATH.exists():
        return retry2.load_results_table(OUTPUT_PATH)
    if SEED_PATH.exists():
        rows = retry2.load_results_table(SEED_PATH)
        retry2.save_results_table(rows, OUTPUT_PATH)
        print(f"seeded {len(rows)} rows from {SEED_PATH}")
        return rows
    return []


def main():
    retry2.make_majoranas_for_B_and_C_with_projection_dim = make_v4_diagonal_minus
    retry2.RESULTS_OUTPUT_PATH = OUTPUT_PATH

    results = load_seeded_results()
    done = {retry2.result_key(row) for row in results}
    if results:
        print(f"loaded {len(results)} existing rows from {OUTPUT_PATH}")

    for u_value in U_VALUES:
        for projection_level in PROJECTION_LEVELS:
            key = (float(u_value), int(projection_level))
            if key in done:
                print(f"skipping existing row: U={u_value}, projection level={projection_level}")
                continue
            row = retry2.run_one_case(u_value, projection_level)
            row["tocheck"] = "MinusDiagonal"
            results.append(row)
            results.sort(key=retry2.result_key)
            retry2.save_results_table(results, OUTPUT_PATH)
            done.add(key)
            print(f"saved {len(results)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
