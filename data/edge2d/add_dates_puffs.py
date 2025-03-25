import pandas as pd
import numpy as np

def restructure_data(df, X_columns, y_columns):
    """Restructure dataframe
    
    The dataframe is restrucutred such that each row contains the following columns:
    - X_columns
    - y_columns
    - species_columns
    - species_values_columns

    The species_columns are binary columns indicating whether a species is present in the
    main ion puff species or impurity puff species. The species_values_columns are the
    puff values for each species.

    X_columns and y_columns are the columns to include in X and y respectively. The species
    columns are created automatically based on the unique species in the main ion puff species
    and impurity puff species columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to restructure
    X_columns : list
        List of column names to include in X
    y_columns : list
        List of column names to include in y
    """
    # Drop rows where main_ion_puff_species is in impurity_puff_species

    df.loc[:,'main_ion_puff_species'] = [ val[0] for val in df['main_ion_puff_species'].values]

    #df.loc[:,'impurity_puff_species'] = [ val[0] for val in df['impurity_puff_species'].values]
    df = df[~df.apply(lambda row: row["main_ion_puff_species"] in row["impurity_puff_species"], axis=1)]

    # Get unique impurity species and main ion species
    impurities = np.array(df["impurity_puff_species"])


    impurities = np.array([item for sublist in impurities for item in sublist])
    impurities = set([i for i in impurities])
    main_ion = set([s for s in df["main_ion_puff_species"]])
    all_puff_species = impurities.union(main_ion)
    all_puff_species = [l for l in all_puff_species]
    all_puff_species_decoded = [s.decode("utf-8") for s in all_puff_species]
    #print(all_puff_species, main_ion, impurities, df['main_ion_puff_species'].unique())
    #exit(0)
    # Create a new column for each species. Set to 1 if the species is in either
    # main_ion_puff_species or impurity_puff_species
    for species,species_str in zip(all_puff_species, all_puff_species_decoded):
        df[f"{species_str}_puff_present"] = df.apply(lambda row: species in row["main_ion_puff_species"] or species in row["impurity_puff_species"], axis=1)
        df[f"{species_str}_puff_present"] = df[f"{species_str}_puff_present"].astype(int)


    # Get puff values for each species
    def extract_values(row, species):
        if species in row["main_ion_puff_species"]:
            return row["main_ion_puff_values"][0]
        elif species in row["impurity_puff_species"]:
            idx = list(row["impurity_puff_species"]).index(species)
            return row["impurity_puff_values"][idx]
        else:
            return 0
    
    for species,species_str in zip(all_puff_species, all_puff_species_decoded):
        df[f"{species_str}_puff_values"] = df.apply(lambda row: extract_values(row, species), axis=1)

    species_columns = [f"{s}_puff_present" for s in list(all_puff_species_decoded)]
    species_values_columns = [f"{species}_puff_values" for species in all_puff_species_decoded]

    # Sometimes, species is given but puff is 0
    # If this is the case, set the species column to 0
    for species,species_str in zip(all_puff_species, all_puff_species_decoded):
        df[f"{species_str}_puff_present"] = df.apply(lambda row: row[f"{species_str}_puff_present"] if row[f"{species_str}_puff_values"] != 0 else 0, axis=1)

    # Drop rows where all species values columns are 0
    df = df[~df.apply(lambda row: all(row[f"{species_str}_puff_values"] == 0 for species_str in all_puff_species_decoded), axis=1)]
   
    df_new = df[X_columns + species_columns + species_values_columns+Y_columns]
    print(df.shape)
    
    return df_new, all_puff_species


# load data
df = pd.read_pickle('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/UPT/UPT/data_store/edge/preprocessed/data.pkl')
df = df.dropna()

# Extract date
#df = df.loc[:10,:]


df["date"] = df.apply(lambda row: row["path"].split("/")[8] if not isinstance(row["path"], float) else np.nan, axis=1)
#[d.split("/")[8] for d in df["path"].values if not isinstance(d,float) ]

df = df[df["date"].str.len() == 7]
df["date"] = pd.to_datetime(df["date"], format="%b%d%y")
# rejig data

X_columns = ['psep','pumped_neutral_flux','inner_avg_albedo','outer_avg_albedo','particle_flux_omp','connection_length','flux_expansion','strike_point_poloidal_angle','path','date','sim_index']
Y_columns = ['peak_target_outer_power_temperature','peak_target_outer_power_density']
df, all_puff_species = restructure_data(df, X_columns, y_columns=Y_columns)

df.to_pickle('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/UPT/UPT/data_store/edge/preprocessed/data_dates_puffs.pkl')
