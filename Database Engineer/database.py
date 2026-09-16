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
            audio_path,
            sample_rate,
            duration_seconds,
            mfcc_shape,
            spectrogram_shape,
            pitch_mean,
            pitch_std,
            pitch_min,
            pitch_max,
            energy_mean,
            energy_std,
            speaking_rate,
            pause_ratio,
            speaker_verification,
            prediction,
            label,
            real_probability,
            fake_probability,
            risk_score,
            risk_level,
            confidence,
            risk_factors,
            recommendation,
            alert,
            security_decision
        FROM analysis_results
        WHERE id = %s
    """

    cursor.execute(query, (analysis_id,))
    result = cursor.fetchone()

    cursor.close()
    connection.close()

    if result:
        for field in [
            "mfcc_shape",
            "spectrogram_shape",
            "speaker_verification",
            "risk_factors",
            "recommendation",
        ]:
            if result[field]:
                result[field] = json.loads(result[field])

    return result


def get_all_analyses():
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT
            id,
            filename,
            timestamp,
            audio_path,
            sample_rate,
            duration_seconds,
            mfcc_shape,
            spectrogram_shape,
            pitch_mean,
            pitch_std,
            pitch_min,
            pitch_max,
            energy_mean,
            energy_std,
            speaking_rate,
            pause_ratio,
            speaker_verification,
            prediction,
            label,
            real_probability,
            fake_probability,
            risk_score,
            risk_level,
            confidence,
            risk_factors,
            recommendation,
            alert,
            security_decision
        FROM analysis_results
        ORDER BY timestamp DESC
    """

    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    for result in results:
        for field in [
            "mfcc_shape",
            "spectrogram_shape",
            "speaker_verification",
            "risk_factors",
            "recommendation",
        ]:
            if result[field]:
                result[field] = json.loads(result[field])

    return results


def insert_analysis_result(
    filename,
    audio_analysis,
    ml_prediction,
    risk_assessment,
):
    connection = get_connection()
    cursor = connection.cursor()

    prosody = audio_analysis.get("prosody") or {}

    query = """
        INSERT INTO analysis_results (
            filename,
            audio_path,
            sample_rate,
            duration_seconds,
            mfcc_shape,
            spectrogram_shape,
            pitch_mean,
            pitch_std,
            pitch_min,
            pitch_max,
            energy_mean,
            energy_std,
            speaking_rate,
            pause_ratio,
            speaker_verification,
            prediction,
            label,
            real_probability,
            fake_probability,
            risk_score,
            risk_level,
            confidence,
            risk_factors,
            recommendation,
            alert,
            security_decision
        )
        VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s
        )
    """

    values = (
        filename,
        audio_analysis.get("audio_path"),
        audio_analysis.get("sample_rate"),
        audio_analysis.get("duration_seconds"),

        json.dumps(audio_analysis.get("mfcc_shape"))
        if audio_analysis.get("mfcc_shape") is not None
        else None,

        json.dumps(audio_analysis.get("spectrogram_shape"))
        if audio_analysis.get("spectrogram_shape") is not None
        else None,

        prosody.get("pitch_mean"),
        prosody.get("pitch_std"),
        prosody.get("pitch_min"),
        prosody.get("pitch_max"),
        prosody.get("energy_mean"),
        prosody.get("energy_std"),
        prosody.get("speaking_rate"),
        prosody.get("pause_ratio"),

        json.dumps(audio_analysis.get("speaker_verification"))
        if audio_analysis.get("speaker_verification") is not None
        else None,

        ml_prediction.get("prediction"),
        ml_prediction.get("label"),
        ml_prediction.get("real_probability"),
        ml_prediction.get("fake_probability"),

        risk_assessment.get("risk_score"),
        risk_assessment.get("risk_level"),
        risk_assessment.get("confidence"),

        json.dumps(risk_assessment.get("risk_factors"))
        if risk_assessment.get("risk_factors") is not None
        else None,

        json.dumps(risk_assessment.get("recommendation"))
        if risk_assessment.get("recommendation") is not None
        else None,

        risk_assessment.get("alert"),
        risk_assessment.get("security_decision"),
    )

    cursor.execute(query, values)
    connection.commit()

    analysis_id = cursor.lastrowid

    cursor.close()
    connection.close()

    return analysis_id