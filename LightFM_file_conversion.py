#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:59:12 2023

@author: hoahduong & ridhikaagrawal

Usage:
    $ spark-submit --deploy-mode client LightFM_file_conversion.py 
"""

# Import command line arguments and helper functions(if necessary)
import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import row_number, count, rand, sum, collect_list, array, sort_array, slice, struct, lit, explode
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.feature import StringIndexer, IndexToString
import pandas as pd

    
def main(spark, userID):
    # file_names = ['interactions_train_small', 'interactions_train', 'interactions_val_small', 'interactions_val']
    
    # file_names = ['interactions_train']
    # file_paths = [f'{x}_split_row.parquet' for x in file_names]
    # # file_paths.append('interactions_test_indexed_row.parquet')
    
    # interactions_train = spark.read.parquet(file_paths[0])
    # # interactions_val = spark.read.parquet(file_paths[1])
    # print("READ FILE")
    
    # interactions_train_pandas_df = interactions_train.toPandas()
    # print("CONVERTED")

    # interactions_train_pandas_df.to_parquet('interactions_train_split_row_pandas.parquet')
    # print("SAVED")
    file_names = ['interactions_train', 'interactions_val']
    file_paths = [f'{x}_split_row.parquet' for x in file_names]
    file_paths.append('interactions_test_indexed_row.parquet') 

    interactions_train = spark.read.parquet(file_paths[0])
    interactions_val = spark.read.parquet(file_paths[1])
    interactions_test = spark.read.parquet(file_paths[2])
    
    # Give the dataframe a temporary view so we can run SQL queries
    interactions_train.createOrReplaceTempView('interactions_train')
    interactions_val.createOrReplaceTempView('interactions_val')
    interactions_test.createOrReplaceTempView('interactions_test')
    
    utility_train = interactions_train.groupby('user_id', 'song_id').count() \
                                 .withColumnRenamed('count', 'total_count') 
    utility_val = interactions_val.groupby('user_id', 'song_id').count() \
                                 .withColumnRenamed('count', 'total_count')
    utility_test = interactions_test.groupby('user_id', 'song_id').count() \
                                 .withColumnRenamed('count', 'total_count')
    print("FILES CREATED")
    
    # utility_train.write.mode('overwrite').parquet('utility_train.parquet')
    # print('WRITTEN')
    # utility_val.write.mode('overwrite').parquet('utility_val.parquet')
    # print('WRITTEN')
    # utility_test.write.mode('overwrite').parquet('utility_test.parquet')
    # print('WRITTEN')
    

    query_count_utility_train = 'SELECT COUNT(*) FROM utility_train'
    query_count_utility_val = 'SELECT COUNT(*) FROM utility_val'
    query_count_utility_test = 'SELECT COUNT(*) FROM utility_test'

    count_utility_train = spark.sql(query_count_utility_train)
    count_utility_val = spark.sql(query_count_utility_val)
    count_utility_test = spark.sql(query_count_utility_test)

    print("Utility train len")
    count_utility_train.show()
    print("Utility val len")
    count_utility_val.show()
    print("Utility test len")
    count_utility_test.show()

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('LightFM_file_conversion').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    
    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    # Call our train routine
    main(spark, userID)

