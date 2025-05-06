# BMSB Project - GIS 5572 - Spring 2025
## Andrew Arlt, Emily Bender, Arlan Hegenbarth, Tim Johnson, Tayrn Reitsma, Lucy Sageser

Folder and file structure:
- **app** - Code for cloud run
- **QA** - Folder for Data Extracting and QA
    - **bmsb_combined_presence_matrix.csv** - csv required to run run_models.py
    - **BMSB_Data.ipynb** - Extract inaturalist bmsb data and combines in with the mn dnr data
    - **BMSBMANAGEMENT.ipynb** - Extracts mn dnr bmsb data from the mn geocommons
    - **CDD_BMSB.ipynb** - Notebook for extract and preping Cooling Degree Day data, which was not used in the final project, but was a planned addition
    - **CDD.ipynb** - Notebook for extract and preping Cooling Degree Day data, which was not used in the final project, but was a planned addition
    - **ctu.ipynb** - Data extract and prepration for the CTU units
    - **landcover_bmsb_final.ipynb** - Notebook for extract and calculating land cover information for each of the CTU units.
    - **zonalhist_attractiveness.csv** - csv required to run run_models.py
- **BMSB_PostGIS.ipynb** - Notebook for pushing model results from local shapefile to cloud database
- **run_models.py** - This is the main code for running and validating the models. It expects the presence of three files specfied in the description and output a probabilities shapefile and evaulation metrics for each model that it runs.
