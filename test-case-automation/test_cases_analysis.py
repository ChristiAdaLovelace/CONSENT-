# -------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import io

uploader = widgets.FileUpload(
    accept=".csv,.xlsx",
    multiple=False)
display(uploader)
# -------------------------------------
if not uploader.value:
    raise RuntimeError("No file uploaded")
uploaded = list(uploader.value.values())[0]
filename = uploaded.get("name", "uploaded_file")
content = uploaded.get("content", None)
if content is None:
    raise RuntimeError("Could not read uploaded file content")
if filename.lower().endswith(".csv"):
    df = pd.read_csv(io.BytesIO(content))
elif filename.lower().endswith((".xls", ".xlsx")):
    df = pd.read_excel(io.BytesIO(content))
else:
    df = pd.read_csv(io.BytesIO(content))
print(f"File loaded successfully")
df.head()
# -------------------------------------
corr = numeric_df.corr()
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()
# -------------------------------------
numeric_df = df.select_dtypes(include="number")
if numeric_df.empty:
    raise RuntimeError("No numeric columns found")
numeric_df.describe().T[["mean", "min", "max", "std"]]
# -------------------------------------
import pandas as pd
cat_col = "category"
numeric_cols = ["gen_time", "api_latency", "total_time", "output_tokens", "tps"]
summary_df = df.groupby(cat_col)[numeric_cols].mean().reset_index()
summary_df = summary_df.round(2)
summary_df
# -------------------------------------
import matplotlib.pyplot as plt
cat_col = df.select_dtypes(include="object").columns[1]
num_cols = df.select_dtypes(include="number").columns
for num_col in num_cols:
    grouped_data = []
    tick_labels = []
    for cat in sorted(df[cat_col].dropna().unique()):
        values = df.loc[df[cat_col] == cat, num_col].dropna()
        if len(values) > 0:
            grouped_data.append(values)
            tick_labels.append(str(cat))

    print(f"{num_col}: showing {len(tick_labels)} categories")

    if len(grouped_data) == 0:
        continue
    fig_width = max(8, len(tick_labels) * 0.6)
    plt.figure(figsize=(fig_width, 6))
    plt.boxplot(grouped_data, tick_labels=tick_labels)
    plt.title(f"{num_col} by {cat_col}")
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
# -------------------------------------
