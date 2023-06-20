CREATE TABLE IF NOT EXISTS energyplus_simulation (
    id INTEGER NOT NULL,
    metadata_id INTEGER NOT NULL,
    reference INTEGER NOT NULL,
    ecobee_building_id INTEGER,
    PRIMARY KEY (id),
    FOREIGN KEY (metadata_id) REFERENCES metadata (id)
        ON DELETE NO ACTION
        ON UPDATE CASCADE,
    FOREIGN KEY (ecobee_building_id) REFERENCES ecobee_building (id)
        ON DELETE NO ACTION
        ON UPDATE CASCADE,
    UNIQUE (metadata_id, reference)
);
CREATE TABLE IF NOT EXISTS lstm_train_data (
    simulation_id INTEGER NOT NULL,
    timestep INTEGER NOT NULL,
    month INTEGER NOT NULL,
    day INTEGER NOT NULL,
    day_name TEXT NOT NULL,
    day_of_week INTEGER NOT NULL,
    hour INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    direct_solar_radiation REAL NOT NULL,
    diffuse_solar_radiation REAL NOT NULL,
    outdoor_air_temperature REAL NOT NULL,
    average_indoor_air_temperature REAL NOT NULL,
    occupant_count INTEGER NOT NULL,
    cooling_load REAL NOT NULL,
    heating_load REAL NOT NULL,
    setpoint REAL NOT NULL,
    PRIMARY KEY (simulation_id, timestep),
    FOREIGN KEY (simulation_id) REFERENCES energyplus_simulation (id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS lstm_train_data_simulation_id ON lstm_train_data(simulation_id);
CREATE TABLE IF NOT EXISTS energyplus_simulation_error_description (
    id INTEGER NOT NULL,
    description TEXT NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (description)
);
INSERT OR IGNORE INTO energyplus_simulation_error_description (id, description)
VALUES 
    (1, 'forward translate error: air loop not found in osm and thermostat not found in idf')
;
CREATE TABLE IF NOT EXISTS energyplus_simulation_error (
    id INTEGER NOT NULL,
    metadata_id INTEGER NOT NULL,
    description_id INTEGER NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (metadata_id) REFERENCES metadata (id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (description_id) REFERENCES energyplus_simulation_error_description (id)
        ON DELETE NO ACTION
        ON UPDATE CASCADE,
    UNIQUE (metadata_id, description_id)
);
DROP VIEW IF EXISTS energyplus_simulation_monthly_unmet_hour_summary;
CREATE VIEW energyplus_simulation_monthly_unmet_hour_summary AS
    WITH t AS (
        SELECT
            m.id AS metadata_id,
            m.bldg_id,
            m.in_resstock_county_id,
            l.month,
            MIN(l.timestep) AS min_timestep,
            MAX(l.timestep) AS max_timestep,
            COUNT(l.timestep) AS count_timestep,
            TOTAL(CASE WHEN ABS(l.average_indoor_air_temperature - l.setpoint) > 2.0 THEN 1 END) AS count_unmet_hour,
            TOTAL(CASE WHEN l.average_indoor_air_temperature - l.setpoint < -2.0 THEN 1 END) AS count_cold_hour,
            TOTAL(CASE WHEN l.average_indoor_air_temperature - l.setpoint > 2.0 THEN 1 END) AS count_hot_hour,
            MIN(l.average_indoor_air_temperature - l.setpoint) AS min_delta,
            MAX(l.average_indoor_air_temperature - l.setpoint) AS max_delta,
            AVG(l.average_indoor_air_temperature - l.setpoint) AS avg_delta
        FROM lstm_train_data l
        LEFT JOIN energyplus_simulation s ON s.id = l.simulation_id
        LEFT JOIN metadata m ON m.id = s.metadata_id
        WHERE l.simulation_id IN (
            SELECT id FROM energyplus_simulation WHERE 
                reference = 0 
                AND metadata_id IN (SELECT metadata_id FROM thermal_zone_count WHERE value = 1)
        )
        GROUP BY
            m.id,
            m.bldg_id,
            m.in_resstock_county_id,
            l.month
    )

    SELECT
        t.metadata_id,
        t.bldg_id,
        t.in_resstock_county_id,
        t.month,
        l.label AS cluster_label,
        t.min_timestep,
        t.max_timestep,
        t.count_timestep,
        t.count_unmet_hour,
        t.count_cold_hour,
        t.count_hot_hour,
        t.count_unmet_hour*100.0/t.count_timestep AS percent_unmet_hour,
        t.count_cold_hour*100.0/t.count_timestep AS percent_cold_hour,
        t.count_hot_hour*100.0/t.count_timestep AS percent_hot_hour,
        t.min_delta,
        t.max_delta,
        t.avg_delta
    FROM t
    LEFT JOIN (
        SELECT
            metadata_id,
            label
        FROM metadata_clustering_label
        WHERE clustering_id IN (SELECT clustering_id FROM optimal_metadata_clustering)
    ) l ON l.metadata_id = t.metadata_id
;