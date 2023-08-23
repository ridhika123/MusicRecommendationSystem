#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:38:06 2023

@author: hoahduong & ridhikaagrawal

Usage:
    $ spark-submit --deploy-mode client baseline_v3.py 
"""

# Import command line arguments and helper functions(if necessary)
import sys
import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, count, rand, sum, collect_list, array, sort_array, slice, struct, lit
from pyspark.mllib.evaluation import RankingMetrics

"""
/user/bm106_nyu_edu/1004-project-2023/

interactions_test.parquet   
interactions_train_small.parquet  
interactions_train.parquet
"""

   
def baseline_popularity(spark, userID, path, beta):
    interactions = spark.read.parquet(path)
    
    # Give the dataframe a temporary view so we can run SQL queries
    interactions.createOrReplaceTempView('interactions')
    
    utility_matrix = interactions.groupby('user_id', 'recording_msid').count()\
                .withColumnRenamed('count', 'total_count')  
    
    popularity_matrix = utility_matrix.groupby('recording_msid')\
                            .agg((sum('total_count') * (count('total_count') + beta)).alias('popularity'))

    popularity_matrix = popularity_matrix.orderBy(popularity_matrix.popularity.desc()).limit(100)
    
    return popularity_matrix
   
def evaluate(spark, userID, val_path, popularity_matrix):
    interactions_val = spark.read.parquet(val_path)
    interactions_val.createOrReplaceTempView('interactions_val')
    
    filled_val_utility = interactions_val.groupBy('user_id', 'recording_msid').count()\
                                        .withColumnRenamed('count', 'total_count') 
    
    
    grouped_val = filled_val_utility.groupBy('user_id')\
                .agg(slice(sort_array(collect_list(struct('total_count', 'recording_msid')), asc = False), 1, 100).alias('songs'))
    
                    
    popular_songs = popularity_matrix.agg(collect_list('recording_msid')).collect()[0][0]
    
    
    selected_val = grouped_val.select(grouped_val.songs.recording_msid)\
                        .withColumnRenamed('songs.recording_msid', 'actual_songs')\
                        .withColumn('popular_songs', array([lit(i) for i in popular_songs]))
                
    pred_and_actual_rdd = selected_val.select('popular_songs', 'actual_songs').rdd
    
    metrics = RankingMetrics(pred_and_actual_rdd)
    
    map_at_100 = metrics.meanAveragePrecisionAt(100)
    
    return map_at_100
            
    
def main(spark, userID):
    # file_names = ['interactions_train_small', 'interactions_train', 'interactions_val_small', 'interactions_val']
    
    file_names = ['interactions_train_small', 'interactions_val_small']
    file_paths = [f'{x}_split.parquet' for x in file_names]
    
    betas = [0]
    # betas = [500, 750, 1000, 2000, 3000, 4000, 5000, 10000]
    # betas = [10000, 12000, 14000, 16000, 18000, 20000, 35000, 50000]
    maps_val = []
    maps_train = []
    
    for beta in betas:
        pop_matrix = baseline_popularity(spark, userID, file_paths[0], beta)
        # print('finished pop_matrix!!!!!!!!!!!!!!!!!!!!')
        
        curr_map_train = evaluate(spark, userID, file_paths[0], pop_matrix)
        print('MeanAP on training set for beta =', beta, ':', curr_map_train)
        maps_train.append(curr_map_train)
        
        curr_map_val = evaluate(spark, userID, file_paths[1], pop_matrix)
        print('MeanAP on evaluation set for beta =', beta, ':', curr_map_val)
        maps_val.append(curr_map_val)
   
    print('MeanAP @ 100', maps_val)
        
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline_v3').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']
    
    # Get file_path for dataset to analyze
    # file_path = sys.argv[1]

    # Call our train routine
    main(spark, userID)