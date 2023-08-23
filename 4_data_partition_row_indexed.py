#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 19:16:18 2023

@author: hoahduong & ridhikaagrawal

Usage:
    $ spark-submit --deploy-mode client data_partition.py 
"""

# Import command line arguments and helper functions(if necessary)
import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, count, rand

"""
/user/bm106_nyu_edu/1004-project-2023/

interactions_test.parquet   
interactions_train_small.parquet  
interactions_train.parquet

f'{name}_processed.parquet'
"""

seed = 11455685

def main(spark, userID):
    
    file_names = ['interactions_train_small', 'interactions_train']
    file_paths = [f'{x}_indexed_row.parquet' for x in file_names]
    
    for i in range(len(file_names)):
        name = file_names[i]
        path = file_paths[i]
        
        interactions = spark.read.parquet(path)
        
        # Give the dataframe a temporary view so we can run SQL queries
        interactions.createOrReplaceTempView('interactions')
        
        interactions = interactions.select('recording_msid', 'user_id', 'timestamp', 'song_id')
         
        train_size = 0.8
        
        # Define windows
        window_id = Window.partitionBy('user_id')
        window_ordered = Window.partitionBy('user_id').orderBy(rand(seed = seed))
        
        # Count total number of interaction for each user
        interactions = interactions.withColumn('total_count', count('*').over(window_id))
        
        # Assign a row number to each interaction within each user_id partition
        interactions = interactions.withColumn("row_num", row_number().over(window_ordered))
    
        # For each user, split the first rows into train and the last rows into validation 
        # Note that we add +1 to force those user with only one interaction into the training set
        train_interact = interactions.filter(interactions.row_num <= (interactions.total_count + 1) * train_size)
        val_interact = interactions.filter(interactions.row_num > (interactions.total_count + 1) * train_size)
        
        train_interact = train_interact.drop('row_num', 'total_count')
        val_interact = val_interact.drop('row_num', 'total_count')
        
        print('FINISHED SPLITTING')
    
        train_interact.write.mode('overwrite').parquet(f'{name}_split_row.parquet')
        print('FINISHED WRITING TRAINING')
        
        val_name = name.replace('train', 'val')
        val_interact.write.mode('overwrite').parquet(f'{val_name}_split_row.parquet')
        print('FINISHED WRITING VALIDATION')
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('data_partition').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    
    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    # Call our main routine
    main(spark, userID)