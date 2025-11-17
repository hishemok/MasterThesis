  # U_vals = [1.0]
    # for U in U_vals:
    #     print(f"\nOptimizing with U fixed to {U} for n=2,3,4")
    #     U = [U] * (2 - 1)  # max n=4
    #     fixed_params = {
    #         "U": torch.tensor(U)
    #         # "t": torch.tensor(1.0)
    #     }
    #     somefixed(n=2, fixed=fixed_params, optimize_method="basin_hopping", with_restarts=True)

    #     U = [U[0]] * (3 - 1)
    #     fixed_params = {
    #         "U": torch.tensor(U)
    #         # "t": torch.tensor(1.0)
    #     }
    #     somefixed(n=3, fixed=fixed_params)
    #     U = [U[0]] * (4 - 1)
    #     fixed_params = {
    #         "U": torch.tensor(U)
    #         # "t": torch.tensor(1.0)
    #     }
    #     somefixed(n=4, fixed=fixed_params)
