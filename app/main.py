import os
import psycopg2
from flask import Flask


# Create Flask app
app = Flask(__name__)

# Create index route
@app.route("/")
def index():
    return "The API is running!"

# Create data route
@app.route("/BMSB_sites", methods=["GET"])
def bmsb_sites():
    # Connect to the database
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD")
    )
    cursor = conn.cursor()

    # Retrieve data from the database
    with conn.cursor() as cur:
        query = """
        SELECT JSON_BUILD_OBJECT(
            'type', 'FeatureCollection',
            'features', JSON_AGG(
                JSON_BUILD_OBJECT(
                    ST_AsGeoJSON(table.*)::json
                )
            )
            FROM table;
        """

        cursor.execute(query)

        data = cur.fetchall()

    # Close the connection
    conn.close()

    # Returning the data
    return data[0][0]

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))