CREATE DATABASE IF NOT EXISTS voice_security;

USE voice_security;

CREATE TABLE IF NOT EXISTS analysis_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    label VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    fake_probability FLOAT NOT NULL,
    risk_score FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    audio_duration FLOAT,
    sample_rate INT,
    alerts JSON,
    recommendation TEXT
);
