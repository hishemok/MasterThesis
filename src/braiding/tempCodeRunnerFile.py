    # #Plot gap
    # plt.figure(figsize=(7,4))
    # plt.plot(Time_array, gap)
    # plt.yscale("log")
    # plt.xlabel("Time")
    # plt.ylabel("Gap to excited states")
    # plt.title("Adiabatic gap")
    # plt.grid(True)
    # plt.show()

    # print("Minimum gap:", gap.min())


    # #Plot energy spectrum
    # plt.figure(figsize=(7,4))
    # for j in range(2):
    #     plt.plot(Time_array, E_low[:, j], label=f"State {j}")
    # plt.xlabel("Time")
    # plt.ylabel("Energy")
    # plt.title("Low-energy spectrum")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # # Plot majorana correlation matrix elements for each PMM
    # for i in range(majorana_correlations.shape[1]): 
    #     plt.figure(figsize=(7,4))
    #     for a in range(majorana_correlations.shape[2]):
    #         for b in range(majorana_correlations.shape[3]):
    #             plt.plot(Time_array, majorana_correlations[:, i, a, b], label=f"C[{a},{b}]")
    #     plt.xlabel("Time")
    #     plt.ylabel("Majorana Correlation C_ab")
    #     plt.title(f"Majorana Correlation Matrix Elements – PMM {i}")
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.grid(True)
    #     plt.show()
    

    # #plot local occupations for each PMM
    # for i in range(local_occupations.shape[1]):
    #     plt.figure(figsize=(7,4))
    #     for j in range(local_occupations.shape[2]):
    #         plt.plot(Time_array, local_occupations[:, i, j], label=f"Site {j}")
    #     plt.xlabel("Time")
    #     plt.ylabel("Local occupation ⟨n⟩")
    #     plt.title(f"Local occupations – PMM {i}")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
    
    # # Plot entanglement entropy for each PMM
    # for i in range(entropy.shape[1]):
    #     plt.figure(figsize=(7,4))
    #     plt.plot(Time_array, entropy[:, i])
    #     plt.xlabel("Time")
    #     plt.ylabel("Entanglement Entropy S")
    #     plt.title(f"Entanglement Entropy – PMM {i}")
    #     plt.grid(True)
    #     plt.show()


    # # Plot pulses
    # t_AB = np.array([p["t_AB"] for p in all_pulses])
    # d_AB = np.array([p["d_AB"] for p in all_pulses])
    # t_BC = np.array([p["t_BC"] for p in all_pulses])
    # d_BC = np.array([p["d_BC"] for p in all_pulses])
    # eps_B = np.array([p["eps_B"] for p in all_pulses])

    # plt.subplots(figsize=(7,6), nrows=2, ncols=1, sharex=True)
    # plt.subplot(2,1,1)
    # plt.plot(Time_array, t_AB, label="t A↔B")
    # plt.plot(Time_array, t_BC, label="t B↔C")
    # plt.plot(Time_array, eps_B, "--", label="ε B")
    # plt.ylabel("Tunneling amplitude")
    # plt.title("Tunneling pulses")
    # plt.legend()
    # plt.grid(True)
    # plt.subplot(2,1,2)
    # plt.plot(Time_array, d_AB, label="Δ A↔B")
    # plt.plot(Time_array, d_BC, label="Δ B↔C")
    # plt.plot(Time_array, eps_B, "--", label="ε B")
    # plt.xlabel("Time")
    # plt.ylabel("Pairing / detuning amplitude")
    # plt.title("Pairing & detuning pulses")
    # plt.legend()
    # plt.grid(True