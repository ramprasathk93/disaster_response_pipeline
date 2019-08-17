import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    - Loads messages and their categories from the given two csv files
    - Merge two dataframes based on the primary key 'id'

    :param messages_filepath: CSV file for disaster messsages
    :param categories_filepath: CSV file for response category
    :return: merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories, messages, on='id', how='outer')
    return df


def clean_data(df):
    """
    Clean and transforms the data in the following manner.
    - Splits the category column based on the ';' delimiter
    - Have only the values 0 and 1 in the records and make the remaining string as a column header
    - Concatenate all the split columns with the original dataframe

    :param df: dataframe to be cleaned and transformed
    :return: cleaned and transformed dataframe
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0][:]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [str(x).strip()[-1] for x in categories[column]]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataframe as a table in a SQLite database

    :param df: dataframe to be converted to a MySQL database table
    :param database_filename: SQLite database name
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_categories', engine, index=False)


def main():
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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
