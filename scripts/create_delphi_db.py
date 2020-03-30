 #!/usr/bin/python
 
from sqlalchemy import create_engine
import pandas as pd


indicator = pd.read_csv("indicator.csv")
gradableAdjectiveData = pd.read_csv("gradableAdjectiveData.csv")
dssat = pd.read_csv("dssat.csv")
concept_to_indicator_mapping = pd.read_csv("concept_to_indicator_mapping.csv")

engine = create_engine(f'postgresql://postgres:delphi@localhost:5432/{sys.argv[5]}', echo=False, pool_pre_ping=True)

df = pd.DataFrame(indicator)
df.to_sql('indicator', engine, if_exists="replace")
df = pd.DataFrame(concept_to_indicator_mapping)
df.to_sql('concept_to_indicator_mapping', engine, if_exists="replace")
df = pd.DataFrame(gradableAdjectiveData)
df.to_sql('gradableAdjectiveData', engine, if_exists="replace")
df = pd.DataFrame(dssat)
df.to_sql('dssat', engine, if_exists="replace")