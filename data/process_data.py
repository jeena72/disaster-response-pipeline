import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets
    
    Parameters
    -----------
    messages_filepath : str
        Messages csv file path
    categories_filepath : str
        Categories csv file path
    
    Returns
    ----------
    df : DataFrame
        Dataframe containing merged data from both messages and categories datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=["id"], how="inner")
    return df


def clean_data(df):
    """
    Extract category information and concatenate it as new category columns
    
    Parameters
    ----------
    df : DataFrame
        dataframe containing category info in a single column
    
    Returns
    ----------
    df : DataFrame
        dataframe with category info as indicator data in separate columns
    """
    # create dataframe of individual category columns
    categories = df["categories"].str.split(";", expand=True)
    # get list of categories 
    row = categories.loc[0, :]
    categories_list = row.apply(lambda x: x.split("-")[0]).values.tolist()
    # rename columns of `categories` dataframe
    categories.columns = categories_list
    # extract indicator data
    for category in categories:
        categories[category] = categories[category].apply(lambda x: x.split("-")[1])
        # convert column from string to numeric
        categories[category] = categories[category].astype(int)
        # replace values > 1 with mode of the category
        category_mode = categories[category].mode()[0]
        categories[category] = categories[category].map(lambda val: category_mode if val > 1 else val)
        # check if no information present in the category
        if((categories[category][0] == categories[category]).all()):
            categories.drop([category], axis=1, inplace=True)
    # drop the original categories column from `df`
    df.drop(["categories"], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Export dataframe to a SQLite database
    
    Parameters
    ----------
    df : DataFrame
        Dataframe to be exported
    database_filename : str
        File path for the generated database

    Returns
    ----------
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponseData', engine, index=False, if_exists="replace")  


def main():
    """
    Runner function
    
    This function:
        1) Loads and merge datasets
        2) Cleans and pre-process data
        3) Save data in SQLite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()