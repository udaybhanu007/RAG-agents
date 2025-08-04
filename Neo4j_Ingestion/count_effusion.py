import pandas as pd

df = pd.read_csv("d:/Softwares/Neo4j-poc/Neo4j_Ingestion/source_document/Data_Entry_2017.csv")
effusion_patients = df[df['Finding Labels'].str.contains('Effusion', na=False)]['Patient ID'].unique()
print("Number of unique patients with Effusion:", len(effusion_patients))