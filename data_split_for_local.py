#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 19:16:18 2023

@author: hoahduong

Usage:
    $ spark-submit --deploy-mode client data_split_for_local.py 
"""

# Import command line arguments and helper functions(if necessary)
import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession

"""
/user/bm106_nyu_edu/1004-project-2023/

interactions_test.parquet   
interactions_train_small.parquet  
interactions_train.parquet
"""

seed = 11455685

def main(spark, userID):
    
    file_path_val = 'utility_val_filtered30.parquet'
    file_path_train = 'utility_train_filtered30.parquet'
    
    utility_val = spark.read.parquet(file_path_val)
    utility_train = spark.read.parquet(file_path_train)
    
    # Give the dataframe a temporary view so we can run SQL queries
    utility_val.createOrReplaceTempView('utility_val')
    utility_train.createOrReplaceTempView('utility_train')
    # tracks.createOrReplaceTempView('tracks')

    ######### SQL
    query_val = "SELECT * FROM utility_val LIMIT 55000000"
    query_train = "SELECT * FROM utility_train LIMIT 55000000"

    query_count_val = "SELECT COUNT(*) FROM utility_val"
    query_count_train = "SELECT COUNT(*) FROM utility_train"

    ######### count
    output_count = spark.sql(query_count_val)
    print('COUNT VAL')
    output_count.show()

    output_count = spark.sql(query_count_train)
    print('COUNT TRAIN')
    output_count.show()
        
    ######### exporting data for local
    output_val = spark.sql(query_val)
    output_val.write.mode('overwrite').parquet('utility_val_local_all_filtered30.parquet')
    print('FINISHED WRITING lOCAL VAL')

    output_train = spark.sql(query_train)
    output_train.write.mode('overwrite').parquet('utility_train_local_all_filtered30.parquet')
    print('FINISHED WRITING lOCAL TRAIN')
    
    ######### Transformation
    # next_interactions = interactions.limit(25190069).offset(30000000)
    # next_interactions.write.mode("overwrite").parquet('utility_train_local_2.parquet')
    # print('FINISHED WRITING lOCAL')

    #########
    # from pyspark.sql.functions import row_number
    # from pyspark.sql.window import Window

    # # Assuming you have a DataFrame named `interactions`
    # # Add an index column to each row
    # df_with_index = interactions.withColumn("index", row_number().over(Window.orderBy("column_name")))

    # # Filter the rows based on their index
    # next_10_million = df_with_index.filter("index > 30000000 and index <= 20000000")

    # # Remove the index column
    # next_10_million = next_10_million.drop("index")

    # # Write the next 10 million rows to a Parquet file
    # next_10_million.write.mode("overwrite").parquet("path/to/next_10_million.parquet")

    ######### 
    # output2 = spark.sql(query2)
    # print('COUNT')
    # output2.show()
    
    # file_path = 'interactions_train_small_indexed_row.parquet'
    
    # interactions = spark.read.parquet(file_path)
    # # tracks = spark.read.parquet(file_path)
    
    # # Give the dataframe a temporary view so we can run SQL queries
    # interactions.createOrReplaceTempView('interactions')
    # # tracks.createOrReplaceTempView('tracks')

    # query = "SELECT * FROM interactions LIMIT 200000"
    # # query = "SELECT * FROM tracks LIMIT 200000"
        
    # output = spark.sql(query)
 
    # output.write.parquet('interactions_row_small.parquet')
    
    # print('FINISHED WRITING ROW')
    
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('data_partition').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)