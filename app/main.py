import os
import psycopg2
from flask import Flask


# Create Flask app
app = Flask(__name__)

# Create index route
@app.route("/")
def index():
    return "The API is running!"

# General DB to GeoJSON function
def database_to_geojson(table_name):
    # Connect to the database
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD")
        )

    # Retrieve data from the database
    with conn.cursor() as cur:
        query = f"""
        SELECT JSON_BUILD_OBJECT(
            'type', 'FeatureCollection',
            'features', JSON_AGG(
                JSON_BUILD_OBJECT(
                    ST_AsGeoJSON({table_name}.*)::json
                )
            )
            FROM {table_name};
        """

        cur.execute(query)

        data = cur.fetchall()

    # Close the connection
    conn.close()

    # Returning the data
    return data[0][0]

# Create data route
@app.route("/BMSB_sites", methods=["GET"])
def bmsb_sites():
    # Call general function
    bmsb = database_to_geojson("BMSB_sites")

    return bmsb

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))