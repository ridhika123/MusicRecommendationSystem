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
from pyspark.sql.functions import row_number, count, rand, sum, collect_list, array, sort_array, slice, struct, lit, explode

    
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
    
    query_train = "SELECT COUNT(*) FROM interactions_train"
    query_val = "SELECT COUNT(*) FROM interactions_val"
    query_test = "SELECT COUNT(*) FROM interactions_test"
    
    count_train = spark.sql(query_train)
    print('INTERACTIONS_TRAIN')
    count_train.show()
    
    count_val = spark.sql(query_val)
    print('INTERACTIONS_VAL')
    count_val.show()
    
    count_test = spark.sql(query_test)
    print('INTERACTIONS_TEST')
    count_test.show()
    
    utility_train = interactions_train.groupby('user_id', 'song_id').count() \
                                 .withColumnRenamed('count', 'total_count') 
    utility_val = interactions_val.groupby('user_id', 'song_id').count() \
                                 .withColumnRenamed('count', 'total_count')
    utility_test = interactions_test.groupby('user_id', 'song_id').count() \
                                 .withColumnRenamed('count', 'total_count')
    print("FILES CREATED")
    
    # Give the dataframe a temporary view so we can run SQL queries
    utility_train.createOrReplaceTempView('utility_train')
    utility_val.createOrReplaceTempView('utility_val')
    utility_test.createOrReplaceTempView('utility_test')
    
    query_train = "SELECT COUNT(*) FROM utility_train"
    query_val = "SELECT COUNT(*) FROM utility_val"
    query_test = "SELECT COUNT(*) FROM utility_test"
    
    count_train = spark.sql(query_train)
    print('UTILITY_TRAIN')
    count_train.show()
    
    count_val = spark.sql(query_val)
    print('UTILITY_VAL')
    count_val.show()
    
    count_test = spark.sql(query_test)
    print('UTILITY_TEST')
    count_test.show()
    
    # Actual songs
    grouped_val = utility_val.groupBy('user_id')\
                            .agg(slice(sort_array(collect_list(struct('total_count', 'song_id')), asc = False), 1, 100).alias('songs'))
    selected_val = grouped_val.select('user_id', grouped_val.songs.song_id)\
                        .withColumnRenamed('songs.song_id', 'actual_songs')      
    print('FINISHED ACTUAL SONGS')
    
    utility_train.write.mode('overwrite').parquet('utility_train.parquet')
    print('WRITTEN TRAIN')
    utility_val.write.mode('overwrite').parquet('utility_val.parquet')
    print('WRITTEN VAL')
    utility_test.write.mode('overwrite').parquet('utility_test.parquet')
    print('WRITTEN TEST')
    selected_val.write.mode('overwrite').parquet('selected_val.parquet')
    print('WRITTEN ACTUAL SONGS')
    
    # utility_train_rep = utility_train.repartition('user_id')
    # utility_val_rep = utility_val.repartition('user_id')
    # utility_test_rep = utility_test.repartition('user_id')
    
    # utility_train_rep.write.mode('overwrite').parquet('utility_train_rep.parquet')
    # print('WRITTEN TRAIN REP')
    # utility_val_rep.write.mode('overwrite').parquet('utility_val_rep.parquet')
    # print('WRITTEN VAL REP')
    # utility_test_rep.write.mode('overwrite').parquet('utility_test_rep.parquet')
    # print('WRITTEN TEST REP')

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('utility_matrix').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    
    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    # Call our train routine
    main(spark, userID)

