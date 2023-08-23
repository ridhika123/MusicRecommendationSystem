#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:38:06 2023

@author: hoahduong

Usage:
    $ spark-submit --deploy-mode client preprocessing.py 
"""

# Import command line arguments and helper functions(if necessary)
import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, count

"""
/user/bm106_nyu_edu/1004-project-2023/

interactions_test.parquet   
interactions_train_small.parquet  
interactions_train.parquet
"""

def main(spark, userID):
    
    file_names = ['interactions_train_small', 'interactions_train']
    file_paths = [f'hdfs:/user/bm106_nyu_edu/1004-project-2023/{x}.parquet' for x in file_names]
    
    for i in range(len(file_names)):
        name = file_names[i]
        path = file_paths[i]
        
        interactions = spark.read.parquet(path)
        
        # Give the dataframe a temporary view so we can run SQL queries
        interactions.createOrReplaceTempView('interactions')
        
        # Remove NULLs values
        query_count_before = 'SELECT COUNT(*) FROM interactions'
        count_before = spark.sql(query_count_before)
    
        print("Count before removing NULLs:")
        count_before.show()
         
        query_null = 'SELECT * FROM interactions WHERE user_id IS NOT NULL OR recording_msid IS NOT NULL'
        
        noNull = spark.sql(query_null)
        noNull.createOrReplaceTempView('noNull') 
        
        query_count_noNull = 'SELECT COUNT(*) FROM noNull'
        count_noNull = spark.sql(query_count_noNull)
        
        print("Count after removing NULLs:")
        count_noNull.show()
        
        # Remove duplicate rows
        query_dupe = 'SELECT DISTINCT(*) FROM noNull'
        noDupe = spark.sql(query_dupe)
        noDupe.createOrReplaceTempView('noDupe') 
        
        query_count_noDupe = 'SELECT COUNT(*) FROM noDupe'
        count_noDupe = spark.sql(query_count_noDupe)
        
        print("Count after removing duplicates:")
        count_noDupe.show()
        
        noDupe.write.mode('overwrite').parquet(f'{name}_cleaned.parquet')
        
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('preprocess').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    
    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    # Call our main routine
    main(spark, userID)