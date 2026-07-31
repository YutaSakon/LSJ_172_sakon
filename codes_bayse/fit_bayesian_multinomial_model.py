!pip install -q numpyro arviz pandas scikit-learn

import pandas as pd

csv_path = "/content/300k_complement_presence_surprisal.csv"

df = pd.read_csv(csv_path)

y_col = "marker_type"
group_col = "trigger"

numeric_main_effect_cols = [
    "words_before_trigger",
    "sc_wordcount",
    "c_ratio",
    "s_distance",
    "c_surprisal",
    "sc_onset_entropy",
    "sc_onset_surprisal",
]

categorical_fixed_effect_cols = ["corpus"]

interaction_cols = ["c_surprisal", "sc_onset_entropy", "sc_onset_surprisal"]

# Standardize numeric columns after interaction terms are created.
# Categorical dummy columns remain 0/1.
standardize_numeric_predictors = True

# Sampling settings.
run_main_mcmc = True
test_num_warmup = 200
test_num_samples = 200
main_num_warmup = 8000
main_num_samples = 2000
main_num_chains = 1
target_accept_prob = 0.95
random_seed = 0

# Output files. These are written next to the Colab working directory unless changed.
summary_csv_path = "numpyro_multinomial_summary.csv"
idata_netcdf_path = "numpyro_multinomial_idata.nc"

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import arviz as az


# %%
# Prefer GPU when available.
device_platforms = {device.platform for device in jax.devices()}
if "cuda" in device_platforms:
    numpyro.set_platform("cuda")
elif "rocm" in device_platforms:
    numpyro.set_platform("rocm")

print("JAX devices:", jax.devices())

df = pd.read_csv(csv_path)

# Match the R preprocessing:
# names(LSJ)[names(LSJ) == "..."] <- "..."
rename_map = {
    "subordinate_clause_word_count": "sc_wordcount",
    "complement_taking_ratio_z": "c_ratio",
    "trigger_to_bahwa_subject_distance_z": "s_distance",
    "source_file": "corpus",
    "complement_present_surprisal_z": "c_surprisal",
}
df = df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})

# Match:
# LSJ$c_ratio <- as.numeric(scale(LSJ$complement_taking_ratio))
# If the raw column is available, it takes precedence over complement_taking_ratio_z.
if "complement_taking_ratio" in df.columns:
    raw_ratio = pd.to_numeric(df["complement_taking_ratio"], errors="coerce")
    df["c_ratio"] = (raw_ratio - raw_ratio.mean()) / raw_ratio.std(ddof=0)

print("Raw data shape:", df.shape)
display(df.head()) if IN_COLAB else print(df.head())

if len(interaction_cols) != 3:
    raise ValueError("interaction_cols must contain exactly 3 column names.")

required_cols = [
    y_col,
    group_col,
    *numeric_main_effect_cols,
    *categorical_fixed_effect_cols,
]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV after renaming: {missing_cols}")

use_cols = [
    y_col,
    group_col,
    *numeric_main_effect_cols,
    *categorical_fixed_effect_cols,
]
df_model = df[use_cols].copy()

for col in numeric_main_effect_cols:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

df_model = df_model.dropna().copy()


#c_ratioをランダムスロープに
a, b, c = interaction_cols
interaction_effect_cols = [
    f"{a}:{b}",
    f"{a}:{c}",
    f"{b}:{c}",
    f"{a}:{b}:{c}",
]
df_model[interaction_effect_cols[0]] = df_model[a] * df_model[b]
df_model[interaction_effect_cols[1]] = df_model[a] * df_model[c]
df_model[interaction_effect_cols[2]] = df_model[b] * df_model[c]
df_model[interaction_effect_cols[3]] = df_model[a] * df_model[b] * df_model[c]

numeric_design_cols = [*numeric_main_effect_cols, *interaction_effect_cols]

# Random effects:
#   (1 + c_ratio || trigger)
random_slope_cols = ["c_ratio"]
random_effect_cols = ["Intercept", *random_slope_cols]

if standardize_numeric_predictors:
    scaler = StandardScaler()
    X_numeric = pd.DataFrame(
        scaler.fit_transform(df_model[numeric_design_cols]),
        columns=numeric_design_cols,
        index=df_model.index,
    )
else:
    scaler = None
    X_numeric = df_model[numeric_design_cols].copy()

categorical_design = pd.get_dummies(
    df_model[categorical_fixed_effect_cols],
    columns=categorical_fixed_effect_cols,
    drop_first=True,
    dtype="float32",
)

X_df = pd.concat([X_numeric, categorical_design], axis=1)
x_cols_model = list(X_df.columns)
X_np = X_df.to_numpy(dtype="float32")

Z_df = pd.DataFrame(
    {"Intercept": np.ones(len(df_model), dtype="float32")},
    index=df_model.index,
)
Z_df = pd.concat([Z_df, X_numeric[random_slope_cols]], axis=1)
Z_np = Z_df[random_effect_cols].to_numpy(dtype="float32")

df_model[group_col] = df_model[group_col].astype(str).str.strip().str.lower()

y_encoder = LabelEncoder()
y_np = y_encoder.fit_transform(df_model[y_col]).astype("int32")

group_encoder = LabelEncoder()
group_id_np = group_encoder.fit_transform(df_model[group_col]).astype("int32")
group_labels = list(group_encoder.classes_)

n_obs, n_predictors = X_np.shape
n_groups = len(group_encoder.classes_)
n_classes = len(y_encoder.classes_)
n_random_terms = Z_np.shape[1]

print("Model data shape:", df_model.shape)
print("N:", n_obs)
print("P:", n_predictors)
print("Groups:", n_groups)
print("Classes:", n_classes)
print("Random-effect terms:", n_random_terms)
print("Class labels:", list(y_encoder.classes_))
print("Group labels:", group_labels)
print("Predictors:", x_cols_model)
print("Random-effect predictors:", random_effect_cols)

if n_classes != 3:
    raise ValueError(f"This script assumes exactly 3 outcome classes, got {n_classes}.")

X = jnp.asarray(X_np)
y = jnp.asarray(y_np)
group_id = jnp.asarray(group_id_np)

def multinomial_random_intercept_model(X, y, group_id, n_groups):
    """3-class multinomial logistic model with group random intercepts.

    Class 0 is the baseline category. The model estimates two logits:
    class 1 vs class 0, and class 2 vs class 0.
    """
    n, p = X.shape
    k_minus_1 = 2

    intercept = numpyro.sample(
        "intercept",
        dist.StudentT(df=3.0, loc=0.0, scale=2.5).expand([k_minus_1]).to_event(1),
    )

    beta = numpyro.sample(
        "beta",
        dist.Normal(0.0, 2.0).expand([p, k_minus_1]).to_event(2),
    )

    sigma_group = numpyro.sample(
        "sigma_group",
        dist.HalfNormal(1.0).expand([k_minus_1]).to_event(1),
    )

    z_group = numpyro.sample(
        "z_group",
        dist.Normal(0.0, 1.0).expand([n_groups, k_minus_1]).to_event(2),
    )

    group_intercept = z_group * sigma_group
    eta = intercept + X @ beta + group_intercept[group_id]
    logits = jnp.concatenate([jnp.zeros((n, 1)), eta], axis=1)

    with numpyro.plate("obs", n):
        numpyro.sample("y", dist.Categorical(logits=logits), obs=y)

kernel = NUTS(
    multinomial_random_intercept_model,
    target_accept_prob=target_accept_prob,
)

mcmc_test = MCMC(
    kernel,
    num_warmup=test_num_warmup,
    num_samples=test_num_samples,
    num_chains=1,
    progress_bar=True,
)

mcmc_test.run(
    jax.random.PRNGKey(random_seed),
    X=X,
    y=y,
    group_id=group_id,
    n_groups=n_groups,
)

mcmc_test.print_summary()

if run_main_mcmc:
    main_mcmc = MCMC(
        kernel,
        num_warmup=main_num_warmup,
        num_samples=main_num_samples,
        num_chains=main_num_chains,
        chain_method="parallel",
        progress_bar=True,
    )

    main_mcmc.run(
        jax.random.PRNGKey(random_seed + 1),
        X=X,
        y=y,
        group_id=group_id,
        n_groups=n_groups,
    )

    main_mcmc.print_summary()
    idata = az.from_numpyro(main_mcmc)
else:
    idata = az.from_numpyro(mcmc_test)


