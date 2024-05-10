nums = [
        [98.1, 0.1],
        [146.8, 0.1],
        [322, 1],
        [672, 1],
        [983, 1],
        [1488, 1 ],
        [2140, 10],
        [2650, 10],
        [3200, 10],
    ]

ri = 4.038474345
ri_error = 0.314254368

for R, R_error in nums:
    if R == 2140:
        R_ri = R + ri
        R_ri_error = R_error + ri_error

        R_ri_abs = abs(R_ri**(-2))

        print(f"{R_ri**(-1)} ± {(R_ri_abs*R_error)+(R_ri_abs*ri_error)} Ω")
        print(R_ri_abs * R_error)
        print(R_ri_abs * ri_error)

