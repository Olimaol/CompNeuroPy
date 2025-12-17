"""
Script for defining the combined samplers to sample the weights of striatal connections.

Data extracted from: https://docs.google.com/spreadsheets/d/1JWMdAMtYBd6oKmrmsM45UX2C_uB_c0CYwPKW3zXR-ys/edit?usp=sharing
"""

# components:

components_spn_spn = [
    {"type": "uniform", "params": {"min": 0.18, "max": 0.37}, "weight": 87.0},
    {
        "type": "trunc_gaussian",
        "params": {"mean": 0.42, "std": 0.25, "min": 0, "max": 0.42 + 0.25 * 6},
        "weight": 23.0,
    },
    {
        "type": "trunc_gaussian",
        "params": {"mean": 0.75, "std": 0.57, "min": 0.07, "max": 1.81},
        "weight": 26.0,
    },
]

components_fsi_spn = [
    {
        "type": "histogram",
        "params": {
            "edges": [
                0,
                1.428571429,
                2.857142857,
                4.285714286,
                5.714285714,
                7.142857143,
                8.571428571,
                10,
                11.42857143,
                12.85714286,
                14.28571429,
                15.71428571,
                17.14285714,
                18.57142857,
                20,
                21.42857143,
                22.85714286,
                24.28571429,
                25.71428571,
                27.14285714,
                28.57142857,
                30,
                31.42857143,
                32.85714286,
                34.28571429,
                35.71428571,
                37.14285714,
                38.57142857,
                40,
                41.42857143,
                42.85714286,
                44.28571429,
                45.71428571,
                47.14285714,
                48.57142857,
                50,
                51.42857143,
                52.85714286,
                54.28571429,
                55.71428571,
                57.14285714,
                58.57142857,
            ],
            "counts": [
                16,
                21,
                6,
                6,
                1,
                2,
                0,
                0,
                0,
                1,
                2,
                3,
                2,
                2,
                3,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
            ],
        },
        "weight": 75,
    },
    {
        "type": "trunc_gaussian",
        "params": {"mean": 1.57, "std": 2.68, "min": 0, "max": 1.57 + 2.68 * 6},
        "weight": 31.0,
    },
    {
        "type": "trunc_gaussian",
        "params": {"mean": 3.84, "std": 3.04, "min": 0.64, "max": 8.14},
        "weight": 9.0,
    },
]

components_fsi_fsi = [
    {
        "type": "trunc_gaussian",
        "params": {"mean": 1.1, "std": 1.5, "min": 0, "max": 1.1 + 1.5 * 6},
        "weight": 6.0,
    },
]

if __name__ == "__main__":
    ############ Example usage and plotting ############

    from CompNeuroPy import CombinedSampler
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    rng = np.random.default_rng(seed=42)

    # sampler for SPN-SPN
    sampler_spn_spn = CombinedSampler(
        components=components_spn_spn,
        rng=rng,
    )

    # sampler for FSI-SPN
    sampler_fsi_spn = CombinedSampler(
        components=components_fsi_spn,
        rng=rng,
    )

    # sampler for FSI-FSI
    sampler_fsi_fsi = CombinedSampler(
        components=components_fsi_fsi,
        rng=rng,
    )

    def sample_and_plot(
        samplers: list[tuple[str, CombinedSampler]],
        n_samples: int = 10000,
        save: bool = True,
        show: bool = False,
    ) -> str | None:
        """Sample from an arbitrary list of samplers and plot histograms.

        Args:
            samplers: List of (title, sampler) tuples.
            n_samples: Number of samples to draw from each sampler.
            save: If True, save figure to PNG in the output directory.
            show: If True, display the plot window (may block in some environments).

        Returns:
            Path to the saved figure if ``save`` is True, else None.
        """
        # Default samplers if none provided
        if len(samplers) == 0:
            raise ValueError("samplers list is empty")

        # Draw samples for each sampler
        sampled_data: list[tuple[str, np.ndarray]] = []
        for title, s in samplers:
            sampled_data.append((title, s.sample(n_samples)))

        n = len(sampled_data)
        # Determine subplot grid (try to be roughly square)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 4.5, rows * 3.8),
            constrained_layout=True,
        )

        # Normalize axes to 2D array for simple indexing
        if isinstance(axes, plt.Axes):  # single plot case
            axes_arr = np.array([[axes]])
        else:
            axes_arr = np.atleast_2d(axes)

        def _stats_text(x: np.ndarray) -> str:
            return (
                f"n={x.size}\n"
                f"mean={np.mean(x):.3f}\nstd={np.std(x):.3f}\n"
                f"min={np.min(x):.3f}\nmax={np.max(x):.3f}"
            )

        # Plot each sampler's histogram
        for idx, (title, data) in enumerate(sampled_data):
            r = idx // cols
            c = idx % cols
            ax = axes_arr[r, c]
            ax.hist(data, bins=50, alpha=0.9, edgecolor="white")
            ax.set_title(title)
            ax.set_xlabel("Weight [nS]")
            ax.set_ylabel("Count")
            ax.text(
                0.98,
                0.98,
                _stats_text(data),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cccccc"
                ),
            )

        # Hide any unused axes
        for extra_idx in range(n, rows * cols):
            r = extra_idx // cols
            c = extra_idx % cols
            axes_arr[r, c].set_visible(False)

        saved_path: str | None = None
        if save:
            output_dir = "connectivity_fits"
            os.makedirs(output_dir, exist_ok=True)
            saved_path = os.path.join(
                output_dir, f"sampler_histograms_{n}_samplers.png"
            )
            fig.savefig(saved_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return saved_path

    custom_out = sample_and_plot(
        n_samples=10000,
        samplers=[
            ("SPN → SPN", sampler_spn_spn),
            ("FSI → SPN", sampler_fsi_spn),
            ("FSI → FSI", sampler_fsi_fsi),
        ],
        save=True,
        show=True,
    )
