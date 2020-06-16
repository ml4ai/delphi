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
    domain_name = ['ASTI', 'CommodityBalances', 'ConsumerPriceIndices', 'Deflators', 'Development', 'Emissions_Agriculture', 'Emissions_Land_Use', 'Employment_Indicators', 'Environment', 'Exchange_rate', 'Food_Aid_Shipments', 'FoodBalance', 'Food_Security', 'FoodSupply', 'Forestry', 'Household_Surveys', 'Inputs', 'Investment', 'Macro-Statistics', 'Population', 'Price', 'Production', 'Trade']
    domain_name_yaml = ['ASTI R&D Indicators', 'Food Balance', 'Prices', 'Prices',  'Investment', 'Emissions - Agriculture', 'Emissions - Land Use', 'Inputs', 'Agri-Environmental Indicators', 'Prices', 'Emergency Response', 'Food Balance', 'Food Security', 'Food Security', 'Forestry', 'Food Security', 'Inputs', 'Investment', 'Macro-Statistics', 'Population', 'Prices', 'Production', 'Trade']
    
    sub_domain = ['ASTI_Expenditures_E_All_Data_',  'ASTI_Researchers_E_All_Data_',  'CommodityBalances_Crops_E_All_Data_', 'CommodityBalances_LivestockFish_E_All_Data_',  'ConsumerPriceIndices_E_All_Data_', 'Deflators_E_All_Data_', 'Development_Assistance_to_Agriculture_E_All_Data_', 'Emissions_Agriculture_Agriculture_total_E_All_Data_', 'Emissions_Agriculture_Burning_crop_residues_E_All_Data_', 'Emissions_Agriculture_Burning_Savanna_E_All_Data_', 'Emissions_Agriculture_Crop_Residues_E_All_Data_', 'Emissions_Agriculture_Cultivated_Organic_Soils_E_All_Data_', 'Emissions_Agriculture_Energy_E_All_Data_', 
    'Emissions_Agriculture_Enteric_Fermentation_E_All_Data_', 'Emissions_Agriculture_Manure_applied_to_soils_E_All_Data_', 'Emissions_Agriculture_Manure_left_on_pasture_E_All_Data_', 'Emissions_Agriculture_Manure_Management_E_All_Data_', 'Emissions_Agriculture_Rice_Cultivation_E_All_Data_', 'Emissions_Agriculture_Synthetic_Fertilizers_E_All_Data_', 'Emissions_Land_Use_Burning_Biomass_E_All_Data_', 'Emissions_Land_Use_Cropland_E_All_Data_', 'Emissions_Land_Use_Forest_Land_E_All_Data_', 'Emissions_Land_Use_Grassland_E_All_Data_', 'Emissions_Land_Use_Land_Use_Total_E_All_Data_', 'Employment_Indicators_E_All_Data_', 'Environment_Emissions_by_Sector_E_All_Data_', 'Environment_Emissions_intensities_E_All_Data_', 'Environment_Fertilizers_E_All_Data_',  'Environment_LandCover_E_All_Data_', 'Environment_LandUse_E_All_Data_', 
    'Environment_LivestockManure_E_All_Data_', 'Environment_LivestockPatterns_E_All_Data_',  'Environment_Pesticides_E_All_Data_', 'Environment_Temperature_change_E_All_Data_', 'Exchange_rate_E_All_Data_', 'Food_Aid_Shipments_WFP_E_All_Data_',  'FoodBalanceSheets_E_All_Data_', 'FoodBalanceSheetsHistoric_E_All_Data_', 'Food_Security_Data_E_All_Data_', 'FoodSupply_Crops_E_All_Data_',  'FoodSupply_LivestockFish_E_All_Data_', 'Forestry_E_All_Data_', 'Forestry_Trade_Flows_E_All_Data_', 'Indicators_from_Household_Surveys_E_All_Data_', 'Inputs_FertilizersArchive_E_All_Data_', 'Inputs_FertilizersNutrient_E_All_Data_', 'Inputs_FertilizersProduct_E_All_Data_', 'Inputs_LandUse_E_All_Data_', 'Inputs_Pesticides_Trade_E_All_Data_', 'Inputs_Pesticides_Use_E_All_Data_', 'Investment_CapitalStock_E_All_Data_', 'Investment_CountryInvestmentStatisticsProfile__E_All_Data_', 
    'Investment_CreditAgriculture_E_All_Data_', 'Investment_ForeignDirectInvestment_E_All_Data_', 'Investment_GovernmentExpenditure_E_All_Data_', 'Investment_MachineryArchive_E_All_Data_', 'Investment_Machinery_E_All_Data_', 'Macro-Statistics_Key_Indicators_E_All_Data_', 'Population_E_All_Data_', 'Price_Indices_E_All_Data_', 'PricesArchive_E_All_Data', 'Prices_E_All_Data_', 'Prices_Monthly_E_All_Data_', 'Production_Crops_E_All_Data_', 'Production_CropsProcessed_E_All_Data_', 'Production_Indices_E_All_Data_', 'Production_Livestock_E_All_Data_', 'Production_LivestockPrimary_E_All_Data_', 'Production_LivestockProcessed_E_All_Data_', 'Trade_Crops_Livestock_E_All_Data_', 'Trade_DetailedTradeMatrix_E_All_Data_', 'Trade_Indices_E_All_Data_', 'Trade_LiveAnimals_E_All_Data_', 'Value_of_Production_E_All_Data_' ]
    
    sub_domain_yaml = ['ASTI-Expenditures', 'ASTI-Researchers', 'Commodity Balances - Crops Primary Equivalent',  'Commodity Balances - Livestock and Fish Primary Equivalent', 'Consumer Price Indices', 'Deflators', 'Development Flows to Agriculture', 'Agriculture Total', 'Burning - Crop Residues', 'Burning - Savanna', 'Crop Residues', 'Cultivation of Organic Soils', 'Energy Use', 
    'Enteric Fermentation', 'Manure applied to Soils', 'Manure left on Pasture', 'Manure Management', 'Rice Cultivation', 'Synthetic Fertilizers', 'Burning - Biomass', 'Cropland', 'Forest Land', 'Grassland', 'Land Use Total', 'Employment Indicators', 'Emissions shares', 'Emissions intensities', 'Fertilizers indicators', 'Land Cover', 'Land use indicators', 
    'Livestock Manure', 'Livestock Patterns', 'Pesticides indicators', 'Temperature change', 'Exchange rates - Annual', 'Food Aid Shipments (WFP)', 'New Food Balances', 'Food Balances (old methodology and population)', 'Suite of Food Security Indicators', 'Food Supply - Crops Primary Equivalent', 'Food Supply - Livestock and Fish Primary Equivalent', 'Forestry Production and Trade', 'Forestry Trade Flows', 'Indicators from Household Surveys (gender, area, socioeconomics)', 'Fertilizers archive', 'Fertilizers by Nutrient', 'Fertilizers by Product', 'Land Use', 'Pesticides Trade', 'Pesticides Use', 'Capital Stock', 'Country Investment Statistics Profile', 
    'Credit to Agriculture', 'Foreign Direct Investment (FDI)', 'Government Expenditure', 'Machinery Archive', 'Machinery', 'Macro Indicators', 'Annual population', 'Producer Price Indices - Annual', 'Producer Prices - Archive', 'Producer Prices - Annual', 'Producer Prices - Monthly', 'Crops', 'Crops processed', 'Production Indices', 'Live Animals', 'Livestock Primary', 'Livestock Processed', 'Crops and livestock products', 'Detailed trade matrix', 'Trade Indices', 'Live animals', 'Value of Agricultural Production']

    required_cols = ("Element", "Purpose", "FAO Source", "Breakdown Variable", "Indicator", "Item", "Measure", "Note")

    dict_file = {}

    for filename in tqdm(
        glob(str(faostat_dir) + "/*.csv"), desc="Processing FAOSTAT files"
    ):
        print(filename)

        if 'Flags' in filename: continue

        df = pd.read_csv(
            filename,
            encoding="latin-1",
        )
        df = df.fillna(0)

        pointer2ndlayer = dict_file
        for dn in range(len(domain_name)):
            if domain_name[dn] in filename:
                pointer2ndlayer[domain_name_yaml[dn]] = {}  
                pointer2ndlayer = pointer2ndlayer[domain_name_yaml[dn]]
                break

        filename_first = filename.split('.')
        filename_first = filename_first[0].split('(')
        filename_first = filename_first[0].split('/')
        filename_first = filename_first[1]
        for sd in range(len(sub_domain)):
            if filename_first == sub_domain[sd]:
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

