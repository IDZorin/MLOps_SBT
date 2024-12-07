import seaborn as sns
import pandas as pd

titanic = sns.load_dataset("titanic")
titanic.to_csv("dvc_data/titanic.csv", index=False)
