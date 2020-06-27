import os, sys
import pandas as pd
from glob import glob
from future.utils import lmap
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as S
from ruamel.yaml import YAML
from pathlib import Path
import zipfile
import subprocess as sp
import contextlib

data_dir = Path("data")

def removeLastEmptyDict(pointer):
    if not pointer: return
    for key, val in pointer.items():
        if val == {}: pointer[key] = None
        else: removeLastEmptyDict(pointer[key])

def clean_FAOSTAT_data(outputFile):
    yaml = YAML()
    open(outputFile, 'w').close()

    faostat_dir = "FAOSTAT"
    domain_name = ['gender, area, socioeconomics', 'CapitalStock', 'ASTI', 'CommodityBalances', 'ConsumerPriceIndices', 'Deflators', 'Development', 'Emissions_Agriculture', 'Emissions_Land_Use', 'Employment_Indicators', 'Environment', 'Exchange_rate', 'Food_Aid_Shipments', 'FoodBalance', 'Food_Security', 'FoodSupply', 'Forestry', 'Household_Surveys', 'Inputs', 'Investment', 'Macro-Statistics', 'Population', 'Price', 'Production', 'Trade']
    domain_name_yaml = ['Food Security', 'Macro-Statistics', 'ASTI R&D Indicators', 'Food Balance', 'Prices', 'Prices',  'Investment', 'Emissions - Agriculture', 'Emissions - Land Use', 'Inputs', 'Agri-Environmental Indicators', 'Prices', 'Emergency Response', 'Food Balance', 'Food Security', 'Food Balance', 'Forestry', 'Food Security', 'Inputs', 'Investment', 'Macro-Statistics', 'Population', 'Prices', 'Production', 'Trade']
    
    sub_domain = ['ASTI_Expenditures',  'ASTI_Researchers',  'CommodityBalances_Crops', 'CommodityBalances_LivestockFish',  'ConsumerPriceIndices', 'Deflators', 'Development_Assistance_to_Agriculture', 'Emissions_Agriculture_Agriculture_total', 'Emissions_Agriculture_Burning_crop_residues', 'Emissions_Agriculture_Burning_Savanna', 'Emissions_Agriculture_Crop_Residues', 'Emissions_Agriculture_Cultivated_Organic_Soils', 'Emissions_Agriculture_Energy', 
    'Emissions_Agriculture_Enteric_Fermentation', 'Emissions_Agriculture_Manure_applied_to_soils', 'Emissions_Agriculture_Manure_left_on_pasture', 'Emissions_Agriculture_Manure_Management', 'Emissions_Agriculture_Rice_Cultivation', 'Emissions_Agriculture_Synthetic_Fertilizers', 'Emissions_Land_Use_Burning_Biomass', 'Emissions_Land_Use_Cropland', 'Emissions_Land_Use_Forest_Land', 'Emissions_Land_Use_Grassland', 'Emissions_Land_Use_Land_Use_Total', 'Employment_Indicators', 'Environment_Emissions_by_Sector', 'Environment_Emissions_intensities', 'Environment_Fertilizers',  'Environment_LandCover', 'Environment_LandUse', 
    'Environment_LivestockManure', 'Environment_LivestockPatterns',  'Environment_Pesticides', 'Environment_Temperature_change', 'Exchange_rate', 'Food_Aid_Shipments_WFP',  'FoodBalanceSheets', 'FoodBalanceSheetsHistoric', 'Food_Security_Data', 'FoodSupply_Crops',  'FoodSupply_LivestockFish', 'Forestry_E', 'Forestry_Trade_Flows', 'Indicators_from_Household_Surveys', 'Inputs_FertilizersArchive', 'Inputs_FertilizersNutrient', 'Inputs_FertilizersProduct', 'Inputs_LandUse', 'Inputs_Pesticides_Trade', 'Inputs_Pesticides_Use', 'Investment_CapitalStock', 'Investment_CountryInvestmentStatisticsProfile', 
    'Investment_CreditAgriculture', 'Investment_ForeignDirectInvestment', 'Investment_GovernmentExpenditure', 'Investment_MachineryArchive', 'Investment_Machinery', 'Macro-Statistics_Key_Indicators', 'Population_E', 'Price_Indices', 'PricesArchive', 'Prices_E_All_Data', 'Prices_Monthly', 'Production_Crops', 'Production_CropsProcessed', 'Production_Indices', 'Production_Livestock', 'Production_LivestockPrimary', 'Production_LivestockProcessed', 'Trade_Crops_Livestock', 'Trade_DetailedTradeMatrix', 'Trade_Indices', 'Trade_LiveAnimals', 'Value_of_Production' ]
    
    sub_domain_yaml = ['ASTI-Expenditures', 'ASTI-Researchers', 'Commodity Balances - Crops Primary Equivalent',  'Commodity Balances - Livestock and Fish Primary Equivalent', 'Consumer Price Indices', 'Deflators', 'Development Flows to Agriculture', 'Agriculture Total', 'Burning - Crop Residues', 'Burning - Savanna', 'Crop Residues', 'Cultivation of Organic Soils', 'Energy Use', 
    'Enteric Fermentation', 'Manure applied to Soils', 'Manure left on Pasture', 'Manure Management', 'Rice Cultivation', 'Synthetic Fertilizers', 'Burning - Biomass', 'Cropland', 'Forest Land', 'Grassland', 'Land Use Total', 'Employment Indicators', 'Emissions shares', 'Emissions intensities', 'Fertilizers indicators', 'Land Cover', 'Land use indicators', 
    'Livestock Manure', 'Livestock Patterns', 'Pesticides indicators', 'Temperature change', 'Exchange rates - Annual', 'Food Aid Shipments (WFP)', 'New Food Balances', 'Food Balances (old methodology and population)', 'Suite of Food Security Indicators', 'Food Supply - Crops Primary Equivalent', 'Food Supply - Livestock and Fish Primary Equivalent', 'Forestry Production and Trade', 'Forestry Trade Flows', 'Indicators from Household Surveys (gender, area, socioeconomics)', 'Fertilizers archive', 'Fertilizers by Nutrient', 'Fertilizers by Product', 'Land Use', 'Pesticides Trade', 'Pesticides Use', 'Capital Stock', 'Country Investment Statistics Profile', 
    'Credit to Agriculture', 'Foreign Direct Investment (FDI)', 'Government Expenditure', 'Machinery Archive', 'Machinery', 'Macro Indicators', 'Annual population', 'Producer Price Indices - Annual', 'Producer Prices (old series)', 'Producer Prices', 'Producer Prices - Monthly', 'Crops', 'Crops processed', 'Production Indices', 'Live Animals', 'Livestock Primary', 'Livestock Processed', 'Crops and livestock products', 'Detailed trade matrix', 'Trade Indices', 'Live animals', 'Value of Agricultural Production']

    required_cols = ("Element", "Purpose", "FAO Source", "Breakdown Variable", "Indicator", "Item", "Measure", "Note")

    dict_file = {}

    for filename in tqdm(
        glob(str(faostat_dir) + "/*.csv"), desc="Processing FAOSTAT files"
    ):
        print(filename)

        if 'E_Flags' in filename: continue

        df = pd.read_csv(
            filename,
            encoding="latin-1",
        )
        df = df.fillna(0)

        pointer2ndlayer = dict_file
        for dn in range(len(domain_name)):
            if domain_name[dn] in filename:
                if domain_name_yaml[dn] not in pointer2ndlayer:
                    pointer2ndlayer[domain_name_yaml[dn]] = {}  
                pointer2ndlayer = pointer2ndlayer[domain_name_yaml[dn]]
                break

        filename_first = filename.split('.')
        filename_first = filename_first[0].split('(')
        filename_first = filename_first[0].split('/')
        filename_first = filename_first[1]
        for sd in range(len(sub_domain)):
            if sub_domain[sd] in filename_first:
                if sub_domain_yaml[sd] not in pointer2ndlayer:
                    pointer2ndlayer[sub_domain_yaml[sd]] = {}
                pointer2ndlayer = pointer2ndlayer[sub_domain_yaml[sd]]
                break

        for index, row in df.iterrows():
            pointer = pointer2ndlayer
            for col in required_cols:
                if col in row and row[col]:
                    if row[col] not in pointer:
                        pointer[row[col]] = {}  
                    pointer = pointer[row[col]]
    removeLastEmptyDict(dict_file)

    with open(outputFile, 'a') as file:
        documents = yaml.dump(dict_file, file)

if __name__ == "__main__":
    clean_FAOSTAT_data('FAOSTAT_hierarchy.yaml')

