# ProcessMining
This is the repository for the Data Challenge 2 project.
Includes changes to:
  Preprocessing.ipynb
    preprocess, split and feature_process functions: added an if statement to stop the functions from 
                                              running time-related preprocessing functions, needed for the final random forest.
    
    encoder function: added the encoding of additional variables 
    
    trace_data and cleaner functions: added new independent variables for RFC.                                          
  
  
  RandomForest.ipynb 
    sampler function: randomly sample train data
    
