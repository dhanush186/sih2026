import os
import json
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "voice_security"),
}


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def insert_analysis(
    filename,
    label,
    confidence,
    fake_probability,
    risk_score,
    risk_level,
    audio_duration=None,
    sample_rate=None,
    alerts=None,
    recommendation=None,
):
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO analysis_results
        (
            filename,
            label,
            confidence,
            fake_probability,
            risk_score,
            risk_level,
            audio_duration,
            sample_rate,
            alerts,
            recommendation
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    alerts_json = json.dumps(alerts) if alerts is not None else None

    values = (
        filename,
        label,
        confidence,
        fake_probability,
        risk_score,
        risk_level,
        audio_duration,
        sample_rate,
        alerts_json,
        recommendation,
    )

    cursor.execute(query, values)
    connection.commit()

    analysis_id = cursor.lastrowid

    cursor.close()
    connection.close()

    return analysis_id


def get_analysis(analysis_id):
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            id,
            filename,
            timestamp,
            label,
            confidence,
            fake_probability,
            risk_score,
            risk_level,
            audio_duration,
            sample_rate,
            alerts,
            recommendation
        FROM analysis_results
        WHERE id = %s
    """

    cursor.execute(query, (analysis_id,))
    result = cursor.fetchone()

    cursor.close()
    connection.close()

    if result and result["alerts"]:
        result["alerts"] = json.loads(result["alerts"])

    return result


def get_all_analyses():
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            id,
            filename,
            timestamp,
            label,
            confidence,
            fake_probability,
            risk_score,
            risk_level,
            audio_duration,
            sample_rate,
            alerts,
            recommendation
        FROM analysis_results
        ORDER BY timestamp DESC
    """

    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    for result in results:
        if result["alerts"]:
            result["alerts"] = json.loads(result["alerts"])

    return results
