import os
import psycopg2
import json
from flask import Response
from flask import Flask


# Create Flask app
app = Flask(__name__)

# Create index route
@app.route("/")
def index():
    return "The API is running!"

# General DB to GeoJSON function
def database_to_geojson(table_name, geom_column='geometry'):
    # Connect to the database
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        pot = os.environ.get("DB_PORT")
    )

    # Retrieve data from the database
    with conn.cursor() as cur:
        query = f"""
        SELECT JSON_BUILD_OBJECT(
                'type', 'FeatureCollection',
                'features', JSON_AGG(
                    JSON_BUILD_OBJECT(
                        'type', 'Feature',
                        'geometry', ST_AsGeoJSON(t.{geom_column})::json,
                        'properties', TO_JSONB(t) - '{geom_column}'
                    )
                )
            )
        FROM (SELECT * FROM {table_name}) AS t;
        """

        cur.execute(query)

        data = cur.fetchone()

    # Close the connection
    conn.close()

    # Returning the data
    return data[0][0]

# Create data route
@app.route("/bmsb", methods=["GET"])
def bmsb_sites():
    # Call general function
    bmsb = database_to_geojson("public.bmsb")

    return Response(json.dumps(bmsb), mimetype="application/json")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
