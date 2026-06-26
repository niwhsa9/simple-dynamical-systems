import pandas as pd
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    #df = pd.read_csv(sys.argv[1])
    df = pd.read_csv(sys.stdin)
    #df = df[df["b"] == 0]  # Filter for first (and only) batch
    plt.figure()
    for b in df["b"].unique():
        df_b = df[df["b"] == b]
        plt.plot(df_b["x"], df_b["y"], label=f"Batch {b}")
    #plt.plot(df["x"], df["y"], label="CEM Trajectory")
    plt.show()


