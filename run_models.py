#### This file will run many different iterations of the model, append the results to unique columns of the shapefile and validate the results.

import numpy as np
import copy
import arcpy
import pandas as pd

###########################################################################################
################################## EDIT THESE PARAMETERS ##################################
###########################################################################################

# data information - all files should be in the working directory
working_dir = r"C:\Users\aoheg\Desktop\School\GIS 5572\model_dir"

ctu_shp = "ctus_with_id.shp"
unique_fld = "UNIQUE_ID"
pop_fld = "POPULATION"

presence_csv = "cumulative_bmsb_data.csv"

model_start_date = "12/2022"
end_evaluation_date = "07/2024"

attraction_csv = "zonalhist_attractiveness.csv"
attract_fld = "attractiveness"

val_threshold = 0.4

# inverse distance models to run - list of tuples (alpha, p_max)
invd_models = [(1, 0.5), (1.5, 0.5), (2, 0.5), (2.5, 0.5)]

# gravity models to run - list of tuples (alpha, p_max)
grav_models = [(0.4, 0.5), (0.5, 0.5), (0.6, 0.5), (0.8, 0.5)]

# huff models to run - list of tuples (alpha, beta, p_max)
huff_models = [(4, 2, 0.5), (4, 1.5, 0.5), (3, 2, 0.5), (3, 1.5, 0.5), (4, 2, 0.5), (4, 1.5, 0.5)]

# output files
output_shp = "model_outputs.shp"
metrics_csv = "model_metrics.csv"

###########################################################################################
###########################################################################################
###########################################################################################

##### Functions ###########################################################################

## single cycle of probability aggregation
#   param: current - n*1 array of current probabilities, where n_i in [0,1], will be updated
#   param: prob - n*n array of movement probabilities where n_ij in [0,1]
def prob_agg2(current, prob):
   x = len(current)
   transmit = np.multiply(prob, np.tile(np.atleast_2d(current).T, (1,x)))
   current[:] = 1 - np.prod(1 - transmit, axis = 0) # P(A or B) = 1 - P(!A and !B)

## function to run the model, repeatedly makes calls to prob_agg2
#   param: presence - n*1 array of current presence, where n_i in {0,1}
#   param: probabilities - n*n array of movement probabilities from i to j where n_ij in [0,1]
#                    or  - n*n*k array of movement probabilities from i to j in time step k where n_ijk in [0,1]
#   param: num_cycles - number of time steps to run the array for 
#   param: prob_start - start index for use with probabilities vary by time step
#   param: print_flag - T/F print the model's progress
#   param: return_all - T/F return probabilities at all time steps, if false will just return final
#   return: - n*t array of the models probabilities through time or n*1 array of final probabilities.
def model(presence, probabilities, num_cycles, prob_start=0, print_flag=False, return_all= False):
    # convert to numpy
    presence = np.array(presence)
    probabilities = np.array(probabilities)

    # get number of prob arrays
    num_dims = len(probabilities.shape)
    if num_dims == 3:
        num_probs = probabilities.shape[2]

    # Check inputs
    if num_dims < 2 or num_dims > 3:
        print("ERROR: incorrect number of dimensions in probability array")
        return
    if probabilities.shape[0] != probabilities.shape[1]:
        print("ERROR: Probabilities Matrix not square")
        return
    if presence.shape[0] != probabilities.shape[0]:
        print("ERROR: Presence and Probabilities are not the same size")
        return
    if np.max(probabilities) > 1 or np.max(probabilities) < 0:
        print("ERROR: Invalid Probability value")
        return
    if len(set(presence) - set([0,1])) > 0:
        print("ERROR: Presence is not binary")
        return
    if num_cycles < 1:
        print("ERROR: Number of cycles cannot be less than 1")
        return
    if num_dims == 3 and prob_start >= num_probs:
        print("ERROR: Probability start index is greater than the number of probabilities")
        return

    # start state
    current_state = copy.deepcopy(presence).astype(np.float64)
    if print_flag:
        print(presence) # intial
    prog_track = presence

    # run model - same probabilities
    if num_dims == 2:
        # fill diagonal
        np.fill_diagonal(probabilities, 1)

        for i in range(num_cycles):
            prob_agg2(current_state, probabilities)
            if print_flag:
                print(current_state) # print progress
            prog_track = np.vstack((prog_track, current_state))

    # run model - different probabilities
    if num_dims == 3:
        # fill diagonal
        for i in range(num_probs):
            np.fill_diagonal(probabilities[:,:,i], 1)

        curr_prob = prob_start
        for i in range(num_cycles):
            prob_agg2(current_state, probabilities[:,:,curr_prob])
            if print_flag:
                print(current_state) # print progress
            prog_track = np.vstack((prog_track, current_state))
            curr_prob += 1
            if curr_prob == num_probs:
                curr_prob = 0

    # returns:
    if return_all:
        return prog_track
    return current_state

# function for inverse distance, alpha modifies the stretch, and values returned range from 0 to p_max
#   param: distances - n*n array of distances between all pairs (i,j) or cities d_ij > 0
#   param: alpha - exponential for the distances, probably should be positive
#   param: p_max - maximum probability. Output probabilities wil l be linearly scaled between 0 and this values. p_max in [0,1]
#   return: n*n array of probabilities
def inverse_distance_probabilities(distances, alpha=1, p_max=0.5):
    # Convert to numpy
    np_distances = np.array(distances)

    # Check inputs
    if np_distances.shape[0] != np_distances.shape[1]:
        print("ERROR: Distance Matrix not square")
        return
    if np.min(np_distances < 0):
        print("ERROR: Negative Distance")
        return
    if p_max < 0 or p_max > 1:
        print("ERROR: Invalid value of p_max")
        return

    # Function
    new_distances = (np_distances < 1).astype(int) + np.multiply((np_distances >= 1).astype(int), np_distances) # replace everything less than 1 with 1
    result = np.power(np.power(new_distances, alpha), -1)

    # Rescale array
    arr_min = np.min(result)
    arr_max = np.max(result)
    result = ((result - arr_min)/(arr_max - arr_min)) * p_max

    return result


# function for gravity model, alpha modifies the stretch, and values returned range from 0 to p_max
#   param: distances - n*n array of distances between all pairs (i,j) or cities d_ij > 0
#   param: populations - n*1 array of populations for each CTU unit
#   param: alpha - exponential for the distances, probably should be positive
#   param: p_max - maximum probability. Output probabilities wil l be linearly scaled between 0 and this values. p_max in [0,1]
#   return: n*n array of probabilities
def gravity_model(distances, populations, alpha=0.1, p_max=0.5):
    # Convert to numpy
    np_distances = np.array(distances)
    np_populations = np.array(populations)

    # check inputs
    if np_distances.shape[0] != np_distances.shape[1]:
        print("ERROR: Distance Matrix not square")
        return
    if np.min(np_distances < 0):
        print("ERROR: Negative Distance")
        return
    if np_distances.shape[0] != np_populations.shape[0]:
        print("ERROR: Distance and Populations are not the same size")
        return
    if np.min(np_populations) < 0:
        print("ERROR: Negative Population")
        return
    if p_max < 0 or p_max > 1:
        print("ERROR: Invalid value of p_max")
        return

    # Function
    new_populations = np_populations / 1000 # prevent integer overflow
    new_distances = (np_distances < 1).astype(int) + np.multiply((np_distances >= 1).astype(int), np_distances) # replace everything less than 1 with 1
    weight = np.multiply(np.tile(np.atleast_2d(new_populations).T, (1, len(new_populations))), np.tile(new_populations, (len(new_populations), 1)))
    result = np.power(np.divide(weight, new_distances), alpha)

    # Rescale array
    arr_min = np.min(result)
    arr_max = np.max(result)
    result = ((result - arr_min)/(arr_max - arr_min)) * p_max
    
    return result

# function for huff model, alpha and beta modify the stretch, and values returned range from 0 to p_max
#   param: distances - n*n array of distances between all pairs (i,j) or cities d_ij > 0
#   param: attractiveness - n*1 array of attractivess for each CTU unit a in [0,1]
#   param: alpha - exponential for the attractiveness, probably should be positive
#   param: beta - exponential for the distances, probably should be positive
#   param: p_max - maximum probability. Output probabilities wil l be linearly scaled between 0 and this values. p_max in [0,1]
#   return: n*n array of probabilities
def huff_model(distances, attractiveness, alpha=0.1, beta=0.1, p_max=0.5):
    # Convert to numpy
    np_distances = np.array(distances)
    np_attractiveness = np.array(attractiveness)

    # Error handling
    if np_distances.shape[0] != np_distances.shape[1]:
        print("ERROR: Distance Matrix not square")
        return
    if np.min(np_distances < 0):
        print("ERROR: Negative Distance")
        return
    if np_distances.shape[0] != np_attractiveness.shape[0]:
        print("ERROR: Distance and Populations are not the same size")
        return
    if np.min(np_attractiveness) < 0 or np.max(np_attractiveness) > 1:
        print("ERROR: Attractivness not on scale [0,1]")
        return
    if p_max < 0 or p_max > 1:
        print("ERROR: Invalid value of p_max")
        return

    # Function
    new_distances = (np_distances < 1).astype(int) + np.multiply((np_distances >= 1).astype(int), np_distances) # replace everything less than 1 with 1
    A_j_alpha = np.power(np.tile(np.atleast_2d(np_attractiveness).T, (1,len(np_attractiveness))), alpha)
    D_i_j_beta = np.power(new_distances,-1*beta)
    denominator = np.sum(new_distances, axis=1) * np.sum(np_attractiveness)
    result = np.divide(np.multiply(A_j_alpha, D_i_j_beta), denominator)

    # Rescale array
    arr_min = np.min(result)
    arr_max = np.max(result)
    result = ((result - arr_min)/(arr_max - arr_min)) * p_max

    return result

# function for model validation,
#   param: predictions - a n*t array of prediction probabilities for n ctu units and t timesteps
#   param: truth - actual value of the presence, binary array of ones and zeros
#   param: threshold - prediction threhold
#   return: tuple of (precision, accuracy, recall, TP, FP, FN, TN)
def validate(predictions, truth, thres=0.5):
    # Convert to numpy
    np_predictions = np.array(predictions)
    np_truth = np.array(truth)

    # Error handling
    if np_predictions.shape != np_truth.shape:
        print("ERROR: Predictions and truth arrays do not have the same dimensions")
        return
    if np.min(np_predictions) < 0 or np.max(np_predictions) > 1:
        print("ERROR: Predictions not on scale [0,1]")
    if len(set(np_truth.flatten()) - set([0,1])) > 0:
        print("ERROR: Presence is not binary")
        return
    
    # Make predictions
    np_predictions = (np_predictions > thres).astype(int)

    # Find confusion matrix
    TP = np.sum(np.multiply(np_predictions, np_truth))
    FP = np.sum(np.multiply(np_predictions, 1-np_truth))
    FN = np.sum(np.multiply(1-np_predictions, np_truth))
    TN = np.sum(np.multiply(1-np_predictions, 1-np_truth))

    # get metric values
    precision = TP / (TP+FP)
    accuracy = (TP+TN) / (TP+FP+FN+TN)
    recall = TP / (TP+FN)
    f1_score = 2* (precision*recall) / (precision + recall)

    return((precision, accuracy, recall, f1_score, TP, FP, FN, TN))  


###########################################################################################
##### Main Code ###########################################################################
###########################################################################################

print("Extracting information from source files...")

# Calculate number of timesteps
number_timesteps = int(end_evaluation_date[-4:]) * 12 + int(end_evaluation_date[:2]) - int(model_start_date[-4:]) * 12 - int(model_start_date[:2])

# Make list of validation months
val_months = []
for i in range(1,number_timesteps+1):
    month = i + int(model_start_date[:2])
    year = int(model_start_date[-4:])
    year += month // 12
    month = month % 12
    if month == 0:
        month = 12
    val_months.append(str(month).zfill(2) + "/" + str(year).zfill(4))

# Copy ctus to memory
arcpy.conversion.ExportFeatures(working_dir + "/" + ctu_shp, "memory/ctus", sort_field = unique_fld + " ASCENDING")

# Get populations and distances from ctu file
populations = arcpy.da.FeatureClassToNumPyArray("memory/ctus", pop_fld).astype(int)

# Make OID/UNIQUE_ID dictionary
codes_dict = {}
with arcpy.da.SearchCursor("memory/ctus", ["OBJECTID", unique_fld]) as cursor:
    for row in cursor:
        codes_dict[row[0]] = row[1]

arcpy.management.FeatureToPoint("memory/ctus", "memory/ctu_points")
arcpy.analysis.GenerateNearTable("memory/ctu_points", "memory/ctu_points", "memory/ctu_distances", closest="ALL", distance_unit="Miles")

# put distances into array
distances = np.zeros([2743, 2743])
with arcpy.da.SearchCursor("memory/ctu_distances", ["IN_FID","NEAR_FID","NEAR_DIST"]) as cursor:
    for row in cursor:
        distances[codes_dict[row[0]],codes_dict[row[1]]] = row[2]

# get presence information
bmsb_data = pd.read_csv(working_dir + "/" + presence_csv).sort_values(unique_fld)
start_presence = bmsb_data[model_start_date].to_numpy().astype(int)

# get attractive information
attract_data = pd.read_csv(working_dir + "/" + attraction_csv).sort_values(unique_fld)
attract = attract_data[attract_fld].to_numpy().astype(float)

# store all results
all_results = []

print("Run inverse distance models...")

## INVERSE DISTANCE MODEL
for mod in invd_models:
    if len(mod) != 2:
        print("ERROR: Inverse distance model tuple does not have length 2")
    p = inverse_distance_probabilities(distances, alpha=mod[0], p_max=mod[1])
    res = model(start_presence, p, num_cycles=number_timesteps, return_all=True)
    all_results.append(res)

print("Run gravity models...")

## GRAVITY MODEL
for mod in grav_models:
    if len(mod) != 2:
        print("ERROR: Gravity model tuple does not have length 2")
    p = gravity_model(distances, populations, alpha=mod[0], p_max=mod[1])
    res = model(start_presence, p, num_cycles=number_timesteps, return_all=True)
    all_results.append(res)

print("Run huff models...")

## HUFF MODEL
for mod in huff_models:
    if len(mod) != 3:
        print("ERROR: Huff model tuple does not have length 3")
    p = huff_model(distances, attract, alpha=mod[0], beta=mod[1], p_max=mod[2])
    res = model(start_presence, p, num_cycles=number_timesteps, return_all=True)
    all_results.append(res)

print("Writing data to shapefile...")

## Get list of fields to add
new_fields = []
count = 1
for i in invd_models:
    new_fields.append("INVD" + str(count))
    count += 1
count = 1
for i in grav_models:
    new_fields.append("GRAV" + str(count))
    count += 1
count = 1
for i in huff_models:
    new_fields.append("HUFF" + str(count))
    count += 1

# Add new fields
for fld in new_fields:
    arcpy.management.AddField("memory/ctus", fld, "FLOAT")
arcpy.management.AddField("memory/ctus", "START_PRES", "LONG")

# Populate new fields
with arcpy.da.UpdateCursor("memory/ctus", [unique_fld] + new_fields + ["START_PRES"]) as cursor:
    for row in cursor:
        for fld in range(0,len(new_fields)):
            row[fld+1] = all_results[fld][-1][row[0]]
        row[-1] = start_presence[row[0]]
        cursor.updateRow(row)

# Create output shapefile
arcpy.management.Delete(working_dir + "/" + output_shp)
arcpy.conversion.ExportFeatures("memory/ctus", working_dir + "/" + output_shp)

print("Validate models...")

# Validate each model
metrics = []
for mod in all_results:
    # build truth array
    truth = bmsb_data[val_months].to_numpy().astype(int).T
    metrics.append(validate(mod[1:,:], truth, thres=val_threshold))

# Write model metrics to csv
column_names = ['model_name', 'alpha', 'beta', 'p_max',  'val_threshold', 'start_date', 'precision', 'accuracy', 'recall', 'f1_score', 'TP', 'FP', 'FN', 'TN']
df = pd.DataFrame(columns=column_names)
all_params = invd_models + grav_models + huff_models

for i in range(len(new_fields)):
    to_add = [new_fields[i]]
    if len(all_params[i]) == 2:
        to_add += [all_params[i][0]] + [None] + [all_params[i][1]] + [val_threshold, model_start_date]
    if len(all_params[i]) == 3:
        to_add += list(all_params[i]) + [val_threshold, model_start_date]
    to_add += metrics[i]
    df.loc[len(df)] = to_add

df.to_csv(working_dir + "/" + metrics_csv) 
