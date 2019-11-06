# The Delphi Database

Our ever growing database contains the various tables of data needed to create
and train our delphi models. As mentioned on our
[Installation](https://ml4ai.github.io/delphi/installation.html) page, the
database can be downloaded using the following command:

```
curl -O http://vanga.sista.arizona.edu/delphi_data/delphi.db
```

(Last updated on 8/19/2019)

All data in the database was cleaned and processed by our python scripts found
in our repository in the [scripts/data_processing
directory](https://github.com/ml4ai/delphi/tree/master/scripts/data_processing).

## Indicator data table

This table contains time series data for possible indicators (also known as
observed variables in the models). The main use of the data in this table is to
train and test our inference models.

There are 9 fields for each data entry (exluding the index column). These 
fields include geographical information such as Country, County, and State, 
the data source (e.g, Mitre12, UNHCR, etc), Units, Month and Year, and of 
course the actual values and names of the indicators (listed as Variable).
 
(Month is recorded in the table as an integer value between 1 and 12).

### Data entries with missing fields

Data sets that we enter into the indicator data table occasionally contain
incomplete entries. Leaving fields empty can be problematic when making database queries
using certain programming languages (such as c++). Therefore all empty fields have been 
filled with a surrogate value indicating missing information for that field.

The subsititute values for missing information are as follows:

- Country: "None" (string type)
- County: "None" (string type)
- State: "None" (string type)
- Source: "None" (string type)
- Month: 0 (int type)
- Year: -1 (int type)

## Gradable Adjectives data table

This table contains our gradable adjectives data.

Each entry contains 8 fields (excluding the column index) of 
various statistics and information collected by the CLULAB.

For more details on the collection and usage of this data table, see this
[document](http://vision.cs.arizona.edu/adarsh/Arizona_Text_to_Model_Procedure.pdf) starting on page 10.

## Concept to indicator mappings table

For modeling purposes, we need to connect the participating entities in causal
relations to quantitative data. To do this, we construct a mapping between the
ontology that the entities are grounded to and the list of variables for which
we have obtained and processed quantitative data. The mapping is 
constructed using word embedding similarities (see the
[OntologyMapper](https://github.com/clulab/eidos/blob/master/src/main/scala/org/clulab/wm/eidos/apps/OntologyMapper.scala) tool for
more details).

## DSSAT data table

The DSSAT data table contains monthly historical and forecast monthly rainfall,
along with daily simulated crop yield, for two states in South Sudan - Northern
Bahr el Ghazal and Unity.

## Raw Data

The raw data tables can be downloaded with the following command:

```
curl -O http://vanga.sista.arizona.edu/delphi_data/raw.zip
```

If you prefer to process and clean the data differently than us or would like
to see how the raw data and cleaned data differ.
