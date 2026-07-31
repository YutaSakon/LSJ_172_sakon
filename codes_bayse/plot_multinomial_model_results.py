summary = az.summary(
    idata,
    var_names=["intercept", "beta", "sigma_group"],
    round_to=3,
)

display(summary) if IN_COLAB else print(summary)
summary.to_csv(summary_csv_path)
idata.to_netcdf(idata_netcdf_path)

print("Saved summary:", summary_csv_path)
print("Saved InferenceData:", idata_netcdf_path)

az.plot_trace(idata, var_names=["intercept", "sigma_group"])

coef_rows = []
class_labels = list(y_encoder.classes_)
for predictor_idx, predictor_name in enumerate(x_cols_model):
    for logit_idx in range(2):
        coef_rows.append(
            {
                "parameter": f"beta[{predictor_idx},{logit_idx}]",
                "predictor": predictor_name,
                "contrast": f"{class_labels[logit_idx + 1]} vs {class_labels[0]}",
            }
        )

coef_map = pd.DataFrame(coef_rows)
display(coef_map) if IN_COLAB else print(coef_map)

import matplotlib.pyplot as plt

beta_summary = az.summary(
    idata,
    var_names=["beta"],
    hdi_prob=0.95,
    round_to=4,
).reset_index(names="parameter")

# ArviZ usually names entries like "beta[0, 0]". Normalize defensively in case
# another version omits spaces after commas.
beta_summary["parameter"] = beta_summary["parameter"].str.replace(", ", ",", regex=False)
coef_map_for_merge = coef_map.copy()
coef_map_for_merge["parameter"] = coef_map_for_merge["parameter"].str.replace(", ", ",", regex=False)

coef_plot_df = beta_summary.merge(coef_map_for_merge, on="parameter", how="left")
coef_plot_df = coef_plot_df.sort_values(["contrast", "predictor"]).reset_index(drop=True)

display(coef_plot_df) if IN_COLAB else print(coef_plot_df)

if coef_plot_df["contrast"].isna().any():
    unmatched = coef_plot_df.loc[coef_plot_df["contrast"].isna(), "parameter"].tolist()
    raise ValueError(f"Could not map these beta parameters to predictor names: {unmatched}")

for contrast, plot_df in coef_plot_df.groupby("contrast", sort=False):
    plot_df = plot_df.sort_values("mean")
    y_pos = np.arange(len(plot_df))

    lower_err = plot_df["mean"] - plot_df["hdi_2.5%"]
    upper_err = plot_df["hdi_97.5%"] - plot_df["mean"]

    fig_height = max(5, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9, fig_height))

    ax.errorbar(
        plot_df["mean"],
        y_pos,
        xerr=[lower_err, upper_err],
        fmt="o",
        color="black",
        ecolor="gray",
        elinewidth=1.5,
        capsize=3,
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["predictor"])
    ax.set_xlabel("Posterior mean and 95% credible interval, log-odds scale")
    ax.set_title(contrast)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

beta_summary = az.summary(
    idata,
    var_names=["beta"],
    hdi_prob=0.95,
    round_to=4,
).reset_index(names="parameter")

beta_summary["parameter"] = beta_summary["parameter"].str.replace(", ", ",", regex=False)

coef_map_for_merge = coef_map.copy()
coef_map_for_merge["parameter"] = coef_map_for_merge["parameter"].str.replace(", ", ",", regex=False)

coef_plot_df = beta_summary.merge(coef_map_for_merge, on="parameter", how="left")

if coef_plot_df["contrast"].isna().any():
    unmatched = coef_plot_df.loc[coef_plot_df["contrast"].isna(), "parameter"].tolist()
    raise ValueError(f"Could not map these beta parameters: {unmatched}")

# 元の変数順を保持
predictor_order = {name: i for i, name in enumerate(x_cols_model)}
coef_plot_df["predictor_order"] = coef_plot_df["predictor"].map(predictor_order)

coef_plot_df = coef_plot_df.sort_values(
    ["contrast", "predictor_order"]
).reset_index(drop=True)

plt.style.use("default")

for contrast, plot_df in coef_plot_df.groupby("contrast", sort=False):
    plot_df = plot_df.sort_values("predictor_order", ascending=True)

    y_pos = np.arange(len(plot_df))

    means = plot_df["mean"].to_numpy()
    lower = plot_df["hdi_2.5%"].to_numpy()
    upper = plot_df["hdi_97.5%"].to_numpy()

    lower_err = means - lower
    upper_err = upper - means

    fig_height = max(5, 0.42 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8.5, fig_height), dpi=140)

    # 95%信用区間
    ax.hlines(
        y=y_pos,
        xmin=lower,
        xmax=upper,
        color="#4A4A4A",
        linewidth=2.2,
        alpha=0.85,
    )

    # 事後平均
    ax.scatter(
        means,
        y_pos,
        s=42,
        color="#0072BD",      # MATLAB blue
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    # 0ライン
    ax.axvline(
        0,
        color="#A2142F",      # MATLAB red
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["predictor"], fontsize=10)

    # 上から元の変数順で見せる
    ax.invert_yaxis()

    ax.set_xlabel("Posterior coefficient, log-odds scale", fontsize=11)
    ax.set_title(contrast, fontsize=13, fontweight="bold")

    ax.grid(axis="x", color="0.85", linewidth=0.8)
    ax.grid(axis="y", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # 少し余白を取る
    x_min = np.nanmin(lower)
    x_max = np.nanmax(upper)
    x_pad = 0.08 * (x_max - x_min) if x_max > x_min else 0.5
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    fig.tight_layout()
    plt.show()

condition_entropy_values = [-1.0, 0.0, 1.0]
condition_c_ratio_values = [-1.0, 0.0, 1.0]
surprisal_grid = np.linspace(-4.0, 6.0, 100)

# Set this to a corpus label if you want predictions for a non-reference corpus.
# None means the reference corpus level after dummy coding.
condition_corpus = None


def make_design_matrix_for_conditions(rows):
    raw_numeric = pd.DataFrame(0.0, index=np.arange(len(rows)), columns=numeric_design_cols)

    for row_idx, row in enumerate(rows):
        raw_numeric.loc[row_idx, "c_ratio"] = row["c_ratio"]
        raw_numeric.loc[row_idx, "sc_onset_entropy"] = row["sc_onset_entropy"]
        raw_numeric.loc[row_idx, "sc_onset_surprisal"] = row["sc_onset_surprisal"]

    a, b, c = interaction_cols
    raw_numeric[f"{a}:{b}"] = raw_numeric[a] * raw_numeric[b]
    raw_numeric[f"{a}:{c}"] = raw_numeric[a] * raw_numeric[c]
    raw_numeric[f"{b}:{c}"] = raw_numeric[b] * raw_numeric[c]
    raw_numeric[f"{a}:{b}:{c}"] = raw_numeric[a] * raw_numeric[b] * raw_numeric[c]
    raw_numeric = raw_numeric[numeric_design_cols]

    if standardize_numeric_predictors:
        numeric_part = pd.DataFrame(
            scaler.transform(raw_numeric),
            columns=numeric_design_cols,
            index=raw_numeric.index,
        )
    else:
        numeric_part = raw_numeric.copy()

    categorical_cols = [col for col in x_cols_model if col not in numeric_design_cols]
    categorical_part = pd.DataFrame(0.0, index=raw_numeric.index, columns=categorical_cols)

    if condition_corpus is not None:
        corpus_col = f"corpus_{condition_corpus}"
        if corpus_col not in categorical_part.columns:
            raise ValueError(
                f"{corpus_col} is not in the dummy-coded corpus columns: "
                f"{list(categorical_part.columns)}"
            )
        categorical_part[corpus_col] = 1.0

    design = pd.concat([numeric_part, categorical_part], axis=1)
    return design[x_cols_model].to_numpy(dtype="float32")


condition_rows = []
for entropy_value in condition_entropy_values:
    for c_ratio_value in condition_c_ratio_values:
        for onset_surprisal_value in surprisal_grid:
            condition_rows.append(
                {
                    "sc_onset_entropy": entropy_value,
                    "c_ratio": c_ratio_value,
                    "sc_onset_surprisal": onset_surprisal_value,
                    "condition": (
                        f"entropy = {entropy_value:g}, "
                        f"comp = {c_ratio_value:g}"
                    ),
                }
            )

condition_df = pd.DataFrame(condition_rows)
X_cond = make_design_matrix_for_conditions(condition_rows)

posterior = idata.posterior
beta_samples = (
    posterior["beta"]
    .stack(sample=("chain", "draw"))
    .transpose("sample", "beta_dim_0", "beta_dim_1")
    .values
)
intercept_samples = (
    posterior["intercept"]
    .stack(sample=("chain", "draw"))
    .transpose("sample", "intercept_dim_0")
    .values
)

eta_samples = intercept_samples[:, None, :] + np.einsum(
    "np,spk->snk",
    X_cond,
    beta_samples,
)
baseline_logits = np.zeros((eta_samples.shape[0], eta_samples.shape[1], 1))
logits_samples = np.concatenate([baseline_logits, eta_samples], axis=2)
logits_samples = logits_samples - logits_samples.max(axis=2, keepdims=True)
prob_samples = np.exp(logits_samples)
prob_samples = prob_samples / prob_samples.sum(axis=2, keepdims=True)

conditional_rows = []
for row_idx, row in condition_df.iterrows():
    for class_idx, class_label in enumerate(class_labels):
        draws = prob_samples[:, row_idx, class_idx]
        hdi = az.hdi(draws, hdi_prob=0.95)
        conditional_rows.append(
            {
                "sc_onset_entropy": row["sc_onset_entropy"],
                "c_ratio": row["c_ratio"],
                "sc_onset_surprisal": row["sc_onset_surprisal"],
                "condition": row["condition"],
                "class": class_label,
                "mean": draws.mean(),
                "hdi_2.5%": hdi[0],
                "hdi_97.5%": hdi[1],
            }
        )

conditional_effects_df = pd.DataFrame(conditional_rows)
display(conditional_effects_df.head()) if IN_COLAB else print(conditional_effects_df.head())

conditions = conditional_effects_df["condition"].drop_duplicates().tolist()
fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
axes = axes.ravel()

for ax, condition in zip(axes, conditions):
    plot_df = conditional_effects_df[conditional_effects_df["condition"] == condition]

    for class_label in class_labels:
        class_df = plot_df[plot_df["class"] == class_label].sort_values(
            "sc_onset_surprisal"
        )
        x_values = class_df["sc_onset_surprisal"].to_numpy()
        mean_values = class_df["mean"].to_numpy()
        lower_values = class_df["hdi_2.5%"].to_numpy()
        upper_values = class_df["hdi_97.5%"].to_numpy()

        ax.plot(x_values, mean_values, label=str(class_label))
        ax.fill_between(x_values, lower_values, upper_values, alpha=0.15)

    ax.set_title(condition)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

for ax in axes[len(conditions):]:
    ax.axis("off")

for ax in axes[-3:]:
    ax.set_xlabel("sc_onset_surprisal")

for ax in axes[::3]:
    ax.set_ylabel("Predicted probability")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(class_labels))
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

single_effect_predictors = [
    "words_before_trigger",
    "sc_wordcount",
    "c_ratio",
    "s_distance",
    "c_surprisal",
    "sc_onset_entropy",
    "sc_onset_surprisal",
]
single_effect_grid = np.linspace(-4.0, 4.0, 100)


def make_design_matrix_for_single_effect(predictor_name, grid_values):
    raw_numeric = pd.DataFrame(
        0.0,
        index=np.arange(len(grid_values)),
        columns=numeric_design_cols,
    )
    raw_numeric[predictor_name] = grid_values

    a, b, c = interaction_cols
    raw_numeric[f"{a}:{b}"] = raw_numeric[a] * raw_numeric[b]
    raw_numeric[f"{a}:{c}"] = raw_numeric[a] * raw_numeric[c]
    raw_numeric[f"{b}:{c}"] = raw_numeric[b] * raw_numeric[c]
    raw_numeric[f"{a}:{b}:{c}"] = raw_numeric[a] * raw_numeric[b] * raw_numeric[c]
    raw_numeric = raw_numeric[numeric_design_cols]

    if standardize_numeric_predictors:
        numeric_part = pd.DataFrame(
            scaler.transform(raw_numeric),
            columns=numeric_design_cols,
            index=raw_numeric.index,
        )
    else:
        numeric_part = raw_numeric.copy()

    categorical_cols = [col for col in x_cols_model if col not in numeric_design_cols]
    categorical_part = pd.DataFrame(0.0, index=raw_numeric.index, columns=categorical_cols)

    design = pd.concat([numeric_part, categorical_part], axis=1)
    return design[x_cols_model].to_numpy(dtype="float32")


single_effect_rows = []

for predictor_name in single_effect_predictors:
    X_single = make_design_matrix_for_single_effect(
        predictor_name,
        single_effect_grid,
    )

    eta_samples = intercept_samples[:, None, :] + np.einsum(
        "np,spk->snk",
        X_single,
        beta_samples,
    )
    baseline_logits = np.zeros((eta_samples.shape[0], eta_samples.shape[1], 1))
    logits_samples = np.concatenate([baseline_logits, eta_samples], axis=2)
    logits_samples = logits_samples - logits_samples.max(axis=2, keepdims=True)
    prob_samples = np.exp(logits_samples)
    prob_samples = prob_samples / prob_samples.sum(axis=2, keepdims=True)

    for grid_idx, grid_value in enumerate(single_effect_grid):
        for class_idx, class_label in enumerate(class_labels):
            draws = prob_samples[:, grid_idx, class_idx]
            hdi = az.hdi(draws, hdi_prob=0.95)
            single_effect_rows.append(
                {
                    "predictor": predictor_name,
                    "value": grid_value,
                    "class": class_label,
                    "mean": draws.mean(),
                    "hdi_2.5%": hdi[0],
                    "hdi_97.5%": hdi[1],
                }
            )

single_effects_df = pd.DataFrame(single_effect_rows)
display(single_effects_df.head()) if IN_COLAB else print(single_effects_df.head())

n_predictor_plots = len(single_effect_predictors)
n_cols = 2
n_rows = int(np.ceil(n_predictor_plots / n_cols))
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(14, 4 * n_rows),
    sharey=True,
)
axes = np.asarray(axes).ravel()

for ax, predictor_name in zip(axes, single_effect_predictors):
    plot_df = single_effects_df[single_effects_df["predictor"] == predictor_name]

    for class_label in class_labels:
        class_df = plot_df[plot_df["class"] == class_label].sort_values("value")
        x_values = class_df["value"].to_numpy()
        mean_values = class_df["mean"].to_numpy()
        lower_values = class_df["hdi_2.5%"].to_numpy()
        upper_values = class_df["hdi_97.5%"].to_numpy()

        ax.plot(x_values, mean_values, label=str(class_label))
        ax.fill_between(x_values, lower_values, upper_values, alpha=0.15)

    ax.set_title(predictor_name)
    ax.set_xlabel(predictor_name)
    ax.set_ylabel("Predicted probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

for ax in axes[n_predictor_plots:]:
    ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(class_labels))
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# Corpus-only conditional effect plot.
# Numeric predictors are fixed at 0, trigger random effects are fixed at 0.

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

corpus_dummy_cols = [col for col in x_cols_model if col.startswith("corpus_")]

if len(corpus_dummy_cols) != 1:
    raise ValueError(
        f"This code assumes binary corpus with one dummy column, got: {corpus_dummy_cols}"
    )

corpus_dummy_col = corpus_dummy_cols[0]
non_reference_corpus = corpus_dummy_col.replace("corpus_", "", 1)

corpus_levels = df_model["corpus"].astype(str).dropna().unique().tolist()
reference_candidates = [level for level in corpus_levels if level != non_reference_corpus]

if len(reference_candidates) != 1:
    raise ValueError(
        f"Could not identify reference corpus. "
        f"levels={corpus_levels}, dummy={corpus_dummy_col}"
    )

reference_corpus = reference_candidates[0]
corpus_labels = [reference_corpus, non_reference_corpus]

# Build two rows: reference corpus and non-reference corpus.
# Because all numeric predictors are fixed at 0, all interactions are also 0.
X_corpus_df = pd.DataFrame(0.0, index=np.arange(2), columns=x_cols_model)
X_corpus_df.loc[1, corpus_dummy_col] = 1.0
X_corpus = X_corpus_df.to_numpy(dtype="float32")

posterior = idata.posterior

beta_samples = (
    posterior["beta"]
    .stack(sample=("chain", "draw"))
    .transpose("sample", "beta_dim_0", "beta_dim_1")
    .values
)

intercept_samples = (
    posterior["intercept"]
    .stack(sample=("chain", "draw"))
    .transpose("sample", "intercept_dim_0")
    .values
)

eta_samples = intercept_samples[:, None, :] + np.einsum(
    "np,spk->snk",
    X_corpus,
    beta_samples,
)

baseline_logits = np.zeros((eta_samples.shape[0], eta_samples.shape[1], 1))
logits_samples = np.concatenate([baseline_logits, eta_samples], axis=2)
logits_samples = logits_samples - logits_samples.max(axis=2, keepdims=True)

prob_samples = np.exp(logits_samples)
prob_samples = prob_samples / prob_samples.sum(axis=2, keepdims=True)

corpus_rows = []
for corpus_idx, corpus_label in enumerate(corpus_labels):
    for class_idx, class_label in enumerate(class_labels):
        draws = prob_samples[:, corpus_idx, class_idx]
        hdi = az.hdi(draws, hdi_prob=0.95)
        corpus_rows.append(
            {
                "corpus": corpus_label,
                "class": class_label,
                "mean": draws.mean(),
                "hdi_2.5%": hdi[0],
                "hdi_97.5%": hdi[1],
            }
        )

corpus_effects_df = pd.DataFrame(corpus_rows)
display(corpus_effects_df) if "IN_COLAB" in globals() and IN_COLAB else print(corpus_effects_df)

fig, ax = plt.subplots(figsize=(8, 5))

x_pos = np.arange(len(corpus_labels))
offsets = np.linspace(-0.18, 0.18, len(class_labels))

for offset, class_label in zip(offsets, class_labels):
    plot_df = corpus_effects_df[corpus_effects_df["class"] == class_label]
    plot_df = plot_df.set_index("corpus").loc[corpus_labels].reset_index()

    means = plot_df["mean"].to_numpy()
    lower = plot_df["hdi_2.5%"].to_numpy()
    upper = plot_df["hdi_97.5%"].to_numpy()

    ax.errorbar(
        x_pos + offset,
        means,
        yerr=[means - lower, upper - means],
        fmt="o",
        capsize=4,
        label=str(class_label),
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(corpus_labels)
ax.set_xlabel("corpus")
ax.set_ylabel("Predicted probability")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.25)
ax.legend(title="marker_type")
fig.tight_layout()
plt.show()
