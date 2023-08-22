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

CREATE TABLE IF NOT EXISTS static_lstm_train_data (
    metadata_id INTEGER NOT NULL,
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
    occupant_count INTEGER NOT NULL,
    setpoint REAL NOT NULL,
    PRIMARY KEY (metadata_id, timestep),
    FOREIGN KEY (metadata_id) REFERENCES metadata (id)
        ON DELETE NO ACTION
        ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS static_lstm_train_data_metadata_id ON static_lstm_train_data(metadata_id);

CREATE TABLE IF NOT EXISTS dynamic_lstm_train_data (
    simulation_id INTEGER NOT NULL,
    timestep INTEGER NOT NULL,
    average_indoor_air_temperature REAL NOT NULL,
    cooling_load REAL NOT NULL,
    heating_load REAL NOT NULL,
    PRIMARY KEY (simulation_id, timestep),
    FOREIGN KEY (simulation_id) REFERENCES energyplus_simulation (id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS dynamic_lstm_train_data_simulation_id ON dynamic_lstm_train_data(simulation_id);

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

DROP VIEW IF EXISTS dynamic_lstm_train_data_hourly_summary;
CREATE VIEW IF NOT EXISTS dynamic_lstm_train_data_hourly_summary AS
    WITH m AS (
        SELECT
            e.metadata_id,
            t.*
        FROM dynamic_lstm_train_data t
        LEFT JOIN energyplus_simulation e ON e.id = t.simulation_id
        WHERE t.simulation_id IN (SELECT id FROM energyplus_simulation WHERE reference = 0)
    ), i AS (
        SELECT
            e.metadata_id,
            t.*
        FROM dynamic_lstm_train_data t
        LEFT JOIN energyplus_simulation e ON e.id = t.simulation_id
        WHERE t.simulation_id IN (SELECT id FROM energyplus_simulation WHERE reference = 1)
    ), p AS (
        SELECT
            e.metadata_id,
            t.*
        FROM dynamic_lstm_train_data t
        LEFT JOIN energyplus_simulation e ON e.id = t.simulation_id
        WHERE t.simulation_id IN (SELECT id FROM energyplus_simulation WHERE reference = 2)
    )

    SELECT
        m.metadata_id,
        e.bldg_id,
        e.in_resstock_county_id,
        m.timestep,
        CAST((m.timestep - 1)/24 AS INTEGER) AS day_index,
        CAST((m.timestep - 1)/(24*30) AS INTEGER) AS month_index,
        s.setpoint,
        m.average_indoor_air_temperature AS mechanical_temperature,
        i.average_indoor_air_temperature AS ideal_temperature,
        p.average_indoor_air_temperature AS partial_temperature,
        m.heating_load AS mechanical_heating_load,
        i.heating_load AS ideal_heating_load,
        p.heating_load AS partial_heating_load,
        m.cooling_load AS mechanical_cooling_load,
        i.cooling_load AS ideal_cooling_load,
        p.cooling_load AS partial_cooling_load,
        m.average_indoor_air_temperature - s.setpoint AS mechanical_setpoint_difference,
        i.average_indoor_air_temperature - s.setpoint AS ideal_setpoint_difference,
        p.average_indoor_air_temperature - s.setpoint AS partial_setpoint_difference,
        i.average_indoor_air_temperature - m.average_indoor_air_temperature AS ideal_and_mechanical_temperature_difference,
        p.average_indoor_air_temperature - i.average_indoor_air_temperature AS partial_and_ideal_temperature_difference,
        i.heating_load - m.heating_load AS ideal_and_mechanical_heating_load_difference,
        p.heating_load - i.heating_load AS partial_and_ideal_heating_load_difference,
        i.cooling_load - m.cooling_load AS ideal_and_mechanical_cooling_load_difference,
        p.cooling_load - i.cooling_load AS partial_and_ideal_cooling_load_difference,
        (m.average_indoor_air_temperature - s.setpoint)*100.0/s.setpoint AS mechanical_setpoint_percent_difference,
        (i.average_indoor_air_temperature - s.setpoint)*100.0/s.setpoint AS ideal_setpoint_percent_difference,
        (p.average_indoor_air_temperature - s.setpoint)*100.0/s.setpoint AS partial_setpoint_percent_difference,
        (i.average_indoor_air_temperature - m.average_indoor_air_temperature)*100.0/m.average_indoor_air_temperature AS ideal_and_mechanical_temperature_percent_difference,
        (p.average_indoor_air_temperature - i.average_indoor_air_temperature)*100.0/i.average_indoor_air_temperature AS partial_and_ideal_temperature_percent_difference,
        (i.heating_load - m.heating_load)*100.0/m.heating_load AS ideal_and_mechanical_heating_load_percent_difference,
        (p.heating_load - i.heating_load)*100.0/i.heating_load AS partial_and_ideal_heating_load_percent_difference,
        (i.cooling_load - m.cooling_load)*100.0/m.cooling_load AS ideal_and_mechanical_cooling_load_percent_difference,
        (p.cooling_load - i.cooling_load)*100.0/i.cooling_load AS partial_and_ideal_cooling_load_percent_difference
    FROM m
    LEFT JOIN i ON i.metadata_id = m.metadata_id AND i.timestep = m.timestep
    LEFT JOIN p ON p.metadata_id = m.metadata_id AND p.timestep = m.timestep
    LEFT JOIN static_lstm_train_data s ON s.metadata_id = m.metadata_id AND s.timestep = m.timestep
    LEFT JOIN metadata e ON e.id = m.metadata_id
;

DROP VIEW IF EXISTS dynamic_lstm_train_data_multi_resolution_summary;
CREATE VIEW IF NOT EXISTS dynamic_lstm_train_data_multi_resolution_summary AS
    SELECT
        t.metadata_id,
        t.bldg_id,
        t.in_resstock_county_id,
        t.day_index AS timestep,
        'daily' AS timestep_resolution,
        MIN(t.setpoint) AS minimum_setpoint,
        MIN(t.mechanical_temperature) AS minimum_mechanical_temperature,
        MIN(t.ideal_temperature) AS minimum_ideal_temperature,
        MIN(t.partial_temperature) AS minimum_partial_temperature,
        MIN(t.mechanical_heating_load) AS minimum_mechanical_heating_load,
        MIN(t.ideal_heating_load) AS minimum_ideal_heating_load,
        MIN(t.partial_heating_load) AS minimum_partial_heating_load,
        MIN(t.mechanical_cooling_load) AS minimum_mechanical_cooling_load,
        MIN(t.ideal_cooling_load) AS minimum_ideal_cooling_load,
        MIN(t.partial_cooling_load) AS minimum_partial_cooling_load,
        MIN(t.mechanical_setpoint_difference) AS minimum_mechanical_setpoint_difference,
        MIN(t.ideal_setpoint_difference) AS minimum_ideal_setpoint_difference,
        MIN(t.partial_setpoint_difference) AS minimum_partial_setpoint_difference,
        MIN(t.ideal_and_mechanical_temperature_difference) AS minimum_ideal_and_mechanical_temperature_difference,
        MIN(t.partial_and_ideal_temperature_difference) AS minimum_partial_and_ideal_temperature_difference,
        MIN(t.ideal_and_mechanical_heating_load_difference) AS minimum_ideal_and_mechanical_heating_load_difference,
        MIN(t.partial_and_ideal_heating_load_difference) AS minimum_partial_and_ideal_heating_load_difference,
        MIN(t.ideal_and_mechanical_cooling_load_difference) AS minimum_ideal_and_mechanical_cooling_load_difference,
        MIN(t.partial_and_ideal_cooling_load_difference) AS minimum_partial_and_ideal_cooling_load_difference,
        MIN(t.mechanical_setpoint_percent_difference) AS minimum_mechanical_setpoint_percent_difference,
        MIN(t.ideal_setpoint_percent_difference) AS minimum_ideal_setpoint_percent_difference,
        MIN(t.partial_setpoint_percent_difference) AS minimum_partial_setpoint_percent_difference,
        MIN(t.ideal_and_mechanical_temperature_percent_difference) AS minimum_ideal_and_mechanical_temperature_percent_difference,
        MIN(t.partial_and_ideal_temperature_percent_difference) AS minimum_partial_and_ideal_temperature_percent_difference,
        MIN(t.ideal_and_mechanical_heating_load_percent_difference) AS minimum_ideal_and_mechanical_heating_load_percent_difference,
        MIN(t.partial_and_ideal_heating_load_percent_difference) AS minimum_partial_and_ideal_heating_load_percent_difference,
        MIN(t.ideal_and_mechanical_cooling_load_percent_difference) AS minimum_ideal_and_mechanical_cooling_load_percent_difference,
        MIN(t.partial_and_ideal_cooling_load_percent_difference) AS minimum_partial_and_ideal_cooling_load_percent_difference,
        MAX(t.setpoint) AS maximum_setpoint,
        MAX(t.mechanical_temperature) AS maximum_mechanical_temperature,
        MAX(t.ideal_temperature) AS maximum_ideal_temperature,
        MAX(t.partial_temperature) AS maximum_partial_temperature,
        MAX(t.mechanical_heating_load) AS maximum_mechanical_heating_load,
        MAX(t.ideal_heating_load) AS maximum_ideal_heating_load,
        MAX(t.partial_heating_load) AS maximum_partial_heating_load,
        MAX(t.mechanical_cooling_load) AS maximum_mechanical_cooling_load,
        MAX(t.ideal_cooling_load) AS maximum_ideal_cooling_load,
        MAX(t.partial_cooling_load) AS maximum_partial_cooling_load,
        MAX(t.mechanical_setpoint_difference) AS maximum_mechanical_setpoint_difference,
        MAX(t.ideal_setpoint_difference) AS maximum_ideal_setpoint_difference,
        MAX(t.partial_setpoint_difference) AS maximum_partial_setpoint_difference,
        MAX(t.ideal_and_mechanical_temperature_difference) AS maximum_ideal_and_mechanical_temperature_difference,
        MAX(t.partial_and_ideal_temperature_difference) AS maximum_partial_and_ideal_temperature_difference,
        MAX(t.ideal_and_mechanical_heating_load_difference) AS maximum_ideal_and_mechanical_heating_load_difference,
        MAX(t.partial_and_ideal_heating_load_difference) AS maximum_partial_and_ideal_heating_load_difference,
        MAX(t.ideal_and_mechanical_cooling_load_difference) AS maximum_ideal_and_mechanical_cooling_load_difference,
        MAX(t.partial_and_ideal_cooling_load_difference) AS maximum_partial_and_ideal_cooling_load_difference,
        MAX(t.mechanical_setpoint_difference) AS maximum_mechanical_setpoint_percent_difference,
        MAX(t.ideal_setpoint_difference) AS maximum_ideal_setpoint_percent_difference,
        MAX(t.partial_setpoint_difference) AS maximum_partial_setpoint_percent_difference,
        MAX(t.ideal_and_mechanical_temperature_percent_difference) AS maximum_ideal_and_mechanical_temperature_percent_difference,
        MAX(t.partial_and_ideal_temperature_percent_difference) AS maximum_partial_and_ideal_temperature_percent_difference,
        MAX(t.ideal_and_mechanical_heating_load_percent_difference) AS maximum_ideal_and_mechanical_heating_load_percent_difference,
        MAX(t.partial_and_ideal_heating_load_percent_difference) AS maximum_partial_and_ideal_heating_load_percent_difference,
        MAX(t.ideal_and_mechanical_cooling_load_percent_difference) AS maximum_ideal_and_mechanical_cooling_load_percent_difference,
        MAX(t.partial_and_ideal_cooling_load_percent_difference) AS maximum_partial_and_ideal_cooling_load_percent_difference,
        AVG(t.setpoint) AS average_setpoint,
        AVG(t.mechanical_temperature) AS average_mechanical_temperature,
        AVG(t.ideal_temperature) AS average_ideal_temperature,
        AVG(t.partial_temperature) AS average_partial_temperature,
        AVG(t.mechanical_heating_load) AS average_mechanical_heating_load,
        AVG(t.ideal_heating_load) AS average_ideal_heating_load,
        AVG(t.partial_heating_load) AS average_partial_heating_load,
        AVG(t.mechanical_cooling_load) AS average_mechanical_cooling_load,
        AVG(t.ideal_cooling_load) AS average_ideal_cooling_load,
        AVG(t.partial_cooling_load) AS average_partial_cooling_load,
        AVG(t.mechanical_setpoint_difference) AS average_mechanical_setpoint_difference,
        AVG(t.ideal_setpoint_difference) AS average_ideal_setpoint_difference,
        AVG(t.partial_setpoint_difference) AS average_partial_setpoint_difference,
        AVG(t.ideal_and_mechanical_temperature_difference) AS average_ideal_and_mechanical_temperature_difference,
        AVG(t.partial_and_ideal_temperature_difference) AS average_partial_and_ideal_temperature_difference,
        AVG(t.ideal_and_mechanical_heating_load_difference) AS average_ideal_and_mechanical_heating_load_difference,
        AVG(t.partial_and_ideal_heating_load_difference) AS average_partial_and_ideal_heating_load_difference,
        AVG(t.ideal_and_mechanical_cooling_load_difference) AS average_ideal_and_mechanical_cooling_load_difference,
        AVG(t.partial_and_ideal_cooling_load_difference) AS average_partial_and_ideal_cooling_load_difference,
        AVG(t.mechanical_setpoint_difference) AS average_mechanical_setpoint_percent_difference,
        AVG(t.ideal_setpoint_difference) AS average_ideal_setpoint_percent_difference,
        AVG(t.partial_setpoint_difference) AS average_partial_setpoint_percent_difference,
        AVG(t.ideal_and_mechanical_temperature_percent_difference) AS average_ideal_and_mechanical_temperature_percent_difference,
        AVG(t.partial_and_ideal_temperature_percent_difference) AS average_partial_and_ideal_temperature_percent_difference,
        AVG(t.ideal_and_mechanical_heating_load_percent_difference) AS average_ideal_and_mechanical_heating_load_percent_difference,
        AVG(t.partial_and_ideal_heating_load_percent_difference) AS average_partial_and_ideal_heating_load_percent_difference,
        AVG(t.ideal_and_mechanical_cooling_load_percent_difference) AS average_ideal_and_mechanical_cooling_load_percent_difference,
        AVG(t.partial_and_ideal_cooling_load_percent_difference) AS average_partial_and_ideal_cooling_load_percent_difference
    FROM dynamic_lstm_train_data_hourly_summary t
    GROUP BY
        t.metadata_id,
        t.bldg_id,
        t.in_resstock_county_id,
        t.day_index

    UNION ALL

    SELECT
        t.metadata_id,
        t.bldg_id,
        t.in_resstock_county_id,
        t.month_index AS timestep,
        'monthly' AS timestep_resolution,
        MIN(t.setpoint) AS minimum_setpoint,
        MIN(t.mechanical_temperature) AS minimum_mechanical_temperature,
        MIN(t.ideal_temperature) AS minimum_ideal_temperature,
        MIN(t.partial_temperature) AS minimum_partial_temperature,
        MIN(t.mechanical_heating_load) AS minimum_mechanical_heating_load,
        MIN(t.ideal_heating_load) AS minimum_ideal_heating_load,
        MIN(t.partial_heating_load) AS minimum_partial_heating_load,
        MIN(t.mechanical_cooling_load) AS minimum_mechanical_cooling_load,
        MIN(t.ideal_cooling_load) AS minimum_ideal_cooling_load,
        MIN(t.partial_cooling_load) AS minimum_partial_cooling_load,
        MIN(t.mechanical_setpoint_difference) AS minimum_mechanical_setpoint_difference,
        MIN(t.ideal_setpoint_difference) AS minimum_ideal_setpoint_difference,
        MIN(t.partial_setpoint_difference) AS minimum_partial_setpoint_difference,
        MIN(t.ideal_and_mechanical_temperature_difference) AS minimum_ideal_and_mechanical_temperature_difference,
        MIN(t.partial_and_ideal_temperature_difference) AS minimum_partial_and_ideal_temperature_difference,
        MIN(t.ideal_and_mechanical_heating_load_difference) AS minimum_ideal_and_mechanical_heating_load_difference,
        MIN(t.partial_and_ideal_heating_load_difference) AS minimum_partial_and_ideal_heating_load_difference,
        MIN(t.ideal_and_mechanical_cooling_load_difference) AS minimum_ideal_and_mechanical_cooling_load_difference,
        MIN(t.partial_and_ideal_cooling_load_difference) AS minimum_partial_and_ideal_cooling_load_difference,
        MIN(t.mechanical_setpoint_percent_difference) AS minimum_mechanical_setpoint_percent_difference,
        MIN(t.ideal_setpoint_percent_difference) AS minimum_ideal_setpoint_percent_difference,
        MIN(t.partial_setpoint_percent_difference) AS minimum_partial_setpoint_percent_difference,
        MIN(t.ideal_and_mechanical_temperature_percent_difference) AS minimum_ideal_and_mechanical_temperature_percent_difference,
        MIN(t.partial_and_ideal_temperature_percent_difference) AS minimum_partial_and_ideal_temperature_percent_difference,
        MIN(t.ideal_and_mechanical_heating_load_percent_difference) AS minimum_ideal_and_mechanical_heating_load_percent_difference,
        MIN(t.partial_and_ideal_heating_load_percent_difference) AS minimum_partial_and_ideal_heating_load_percent_difference,
        MIN(t.ideal_and_mechanical_cooling_load_percent_difference) AS minimum_ideal_and_mechanical_cooling_load_percent_difference,
        MIN(t.partial_and_ideal_cooling_load_percent_difference) AS minimum_partial_and_ideal_cooling_load_percent_difference,
        MAX(t.setpoint) AS maximum_setpoint,
        MAX(t.mechanical_temperature) AS maximum_mechanical_temperature,
        MAX(t.ideal_temperature) AS maximum_ideal_temperature,
        MAX(t.partial_temperature) AS maximum_partial_temperature,
        MAX(t.mechanical_heating_load) AS maximum_mechanical_heating_load,
        MAX(t.ideal_heating_load) AS maximum_ideal_heating_load,
        MAX(t.partial_heating_load) AS maximum_partial_heating_load,
        MAX(t.mechanical_cooling_load) AS maximum_mechanical_cooling_load,
        MAX(t.ideal_cooling_load) AS maximum_ideal_cooling_load,
        MAX(t.partial_cooling_load) AS maximum_partial_cooling_load,
        MAX(t.mechanical_setpoint_difference) AS maximum_mechanical_setpoint_difference,
        MAX(t.ideal_setpoint_difference) AS maximum_ideal_setpoint_difference,
        MAX(t.partial_setpoint_difference) AS maximum_partial_setpoint_difference,
        MAX(t.ideal_and_mechanical_temperature_difference) AS maximum_ideal_and_mechanical_temperature_difference,
        MAX(t.partial_and_ideal_temperature_difference) AS maximum_partial_and_ideal_temperature_difference,
        MAX(t.ideal_and_mechanical_heating_load_difference) AS maximum_ideal_and_mechanical_heating_load_difference,
        MAX(t.partial_and_ideal_heating_load_difference) AS maximum_partial_and_ideal_heating_load_difference,
        MAX(t.ideal_and_mechanical_cooling_load_difference) AS maximum_ideal_and_mechanical_cooling_load_difference,
        MAX(t.partial_and_ideal_cooling_load_difference) AS maximum_partial_and_ideal_cooling_load_difference,
        MAX(t.mechanical_setpoint_difference) AS maximum_mechanical_setpoint_percent_difference,
        MAX(t.ideal_setpoint_difference) AS maximum_ideal_setpoint_percent_difference,
        MAX(t.partial_setpoint_difference) AS maximum_partial_setpoint_percent_difference,
        MAX(t.ideal_and_mechanical_temperature_percent_difference) AS maximum_ideal_and_mechanical_temperature_percent_difference,
        MAX(t.partial_and_ideal_temperature_percent_difference) AS maximum_partial_and_ideal_temperature_percent_difference,
        MAX(t.ideal_and_mechanical_heating_load_percent_difference) AS maximum_ideal_and_mechanical_heating_load_percent_difference,
        MAX(t.partial_and_ideal_heating_load_percent_difference) AS maximum_partial_and_ideal_heating_load_percent_difference,
        MAX(t.ideal_and_mechanical_cooling_load_percent_difference) AS maximum_ideal_and_mechanical_cooling_load_percent_difference,
        MAX(t.partial_and_ideal_cooling_load_percent_difference) AS maximum_partial_and_ideal_cooling_load_percent_difference,
        AVG(t.setpoint) AS average_setpoint,
        AVG(t.mechanical_temperature) AS average_mechanical_temperature,
        AVG(t.ideal_temperature) AS average_ideal_temperature,
        AVG(t.partial_temperature) AS average_partial_temperature,
        AVG(t.mechanical_heating_load) AS average_mechanical_heating_load,
        AVG(t.ideal_heating_load) AS average_ideal_heating_load,
        AVG(t.partial_heating_load) AS average_partial_heating_load,
        AVG(t.mechanical_cooling_load) AS average_mechanical_cooling_load,
        AVG(t.ideal_cooling_load) AS average_ideal_cooling_load,
        AVG(t.partial_cooling_load) AS average_partial_cooling_load,
        AVG(t.mechanical_setpoint_difference) AS average_mechanical_setpoint_difference,
        AVG(t.ideal_setpoint_difference) AS average_ideal_setpoint_difference,
        AVG(t.partial_setpoint_difference) AS average_partial_setpoint_difference,
        AVG(t.ideal_and_mechanical_temperature_difference) AS average_ideal_and_mechanical_temperature_difference,
        AVG(t.partial_and_ideal_temperature_difference) AS average_partial_and_ideal_temperature_difference,
        AVG(t.ideal_and_mechanical_heating_load_difference) AS average_ideal_and_mechanical_heating_load_difference,
        AVG(t.partial_and_ideal_heating_load_difference) AS average_partial_and_ideal_heating_load_difference,
        AVG(t.ideal_and_mechanical_cooling_load_difference) AS average_ideal_and_mechanical_cooling_load_difference,
        AVG(t.partial_and_ideal_cooling_load_difference) AS average_partial_and_ideal_cooling_load_difference,
        AVG(t.mechanical_setpoint_difference) AS average_mechanical_setpoint_percent_difference,
        AVG(t.ideal_setpoint_difference) AS average_ideal_setpoint_percent_difference,
        AVG(t.partial_setpoint_difference) AS average_partial_setpoint_percent_difference,
        AVG(t.ideal_and_mechanical_temperature_percent_difference) AS average_ideal_and_mechanical_temperature_percent_difference,
        AVG(t.partial_and_ideal_temperature_percent_difference) AS average_partial_and_ideal_temperature_percent_difference,
        AVG(t.ideal_and_mechanical_heating_load_percent_difference) AS average_ideal_and_mechanical_heating_load_percent_difference,
        AVG(t.partial_and_ideal_heating_load_percent_difference) AS average_partial_and_ideal_heating_load_percent_difference,
        AVG(t.ideal_and_mechanical_cooling_load_percent_difference) AS average_ideal_and_mechanical_cooling_load_percent_difference,
        AVG(t.partial_and_ideal_cooling_load_percent_difference) AS average_partial_and_ideal_cooling_load_percent_difference
    FROM dynamic_lstm_train_data_hourly_summary t
    GROUP BY
        t.metadata_id,
        t.bldg_id,
        t.in_resstock_county_id,
        t.month_index

    UNION ALL

    SELECT
        t.metadata_id,
        t.bldg_id,
        t.in_resstock_county_id,
        MIN(t.month_index) AS timestep,
        'yearly' AS timestep_resolution,
        MIN(t.setpoint) AS minimum_setpoint,
        MIN(t.mechanical_temperature) AS minimum_mechanical_temperature,
        MIN(t.ideal_temperature) AS minimum_ideal_temperature,
        MIN(t.partial_temperature) AS minimum_partial_temperature,
        MIN(t.mechanical_heating_load) AS minimum_mechanical_heating_load,
        MIN(t.ideal_heating_load) AS minimum_ideal_heating_load,
        MIN(t.partial_heating_load) AS minimum_partial_heating_load,
        MIN(t.mechanical_cooling_load) AS minimum_mechanical_cooling_load,
        MIN(t.ideal_cooling_load) AS minimum_ideal_cooling_load,
        MIN(t.partial_cooling_load) AS minimum_partial_cooling_load,
        MIN(t.mechanical_setpoint_difference) AS minimum_mechanical_setpoint_difference,
        MIN(t.ideal_setpoint_difference) AS minimum_ideal_setpoint_difference,
        MIN(t.partial_setpoint_difference) AS minimum_partial_setpoint_difference,
        MIN(t.ideal_and_mechanical_temperature_difference) AS minimum_ideal_and_mechanical_temperature_difference,
        MIN(t.partial_and_ideal_temperature_difference) AS minimum_partial_and_ideal_temperature_difference,
        MIN(t.ideal_and_mechanical_heating_load_difference) AS minimum_ideal_and_mechanical_heating_load_difference,
        MIN(t.partial_and_ideal_heating_load_difference) AS minimum_partial_and_ideal_heating_load_difference,
        MIN(t.ideal_and_mechanical_cooling_load_difference) AS minimum_ideal_and_mechanical_cooling_load_difference,
        MIN(t.partial_and_ideal_cooling_load_difference) AS minimum_partial_and_ideal_cooling_load_difference,
        MIN(t.mechanical_setpoint_percent_difference) AS minimum_mechanical_setpoint_percent_difference,
        MIN(t.ideal_setpoint_percent_difference) AS minimum_ideal_setpoint_percent_difference,
        MIN(t.partial_setpoint_percent_difference) AS minimum_partial_setpoint_percent_difference,
        MIN(t.ideal_and_mechanical_temperature_percent_difference) AS minimum_ideal_and_mechanical_temperature_percent_difference,
        MIN(t.partial_and_ideal_temperature_percent_difference) AS minimum_partial_and_ideal_temperature_percent_difference,
        MIN(t.ideal_and_mechanical_heating_load_percent_difference) AS minimum_ideal_and_mechanical_heating_load_percent_difference,
        MIN(t.partial_and_ideal_heating_load_percent_difference) AS minimum_partial_and_ideal_heating_load_percent_difference,
        MIN(t.ideal_and_mechanical_cooling_load_percent_difference) AS minimum_ideal_and_mechanical_cooling_load_percent_difference,
        MIN(t.partial_and_ideal_cooling_load_percent_difference) AS minimum_partial_and_ideal_cooling_load_percent_difference,
        MAX(t.setpoint) AS maximum_setpoint,
        MAX(t.mechanical_temperature) AS maximum_mechanical_temperature,
        MAX(t.ideal_temperature) AS maximum_ideal_temperature,
        MAX(t.partial_temperature) AS maximum_partial_temperature,
        MAX(t.mechanical_heating_load) AS maximum_mechanical_heating_load,
        MAX(t.ideal_heating_load) AS maximum_ideal_heating_load,
        MAX(t.partial_heating_load) AS maximum_partial_heating_load,
        MAX(t.mechanical_cooling_load) AS maximum_mechanical_cooling_load,
        MAX(t.ideal_cooling_load) AS maximum_ideal_cooling_load,
        MAX(t.partial_cooling_load) AS maximum_partial_cooling_load,
        MAX(t.mechanical_setpoint_difference) AS maximum_mechanical_setpoint_difference,
        MAX(t.ideal_setpoint_difference) AS maximum_ideal_setpoint_difference,
        MAX(t.partial_setpoint_difference) AS maximum_partial_setpoint_difference,
        MAX(t.ideal_and_mechanical_temperature_difference) AS maximum_ideal_and_mechanical_temperature_difference,
        MAX(t.partial_and_ideal_temperature_difference) AS maximum_partial_and_ideal_temperature_difference,
        MAX(t.ideal_and_mechanical_heating_load_difference) AS maximum_ideal_and_mechanical_heating_load_difference,
        MAX(t.partial_and_ideal_heating_load_difference) AS maximum_partial_and_ideal_heating_load_difference,
        MAX(t.ideal_and_mechanical_cooling_load_difference) AS maximum_ideal_and_mechanical_cooling_load_difference,
        MAX(t.partial_and_ideal_cooling_load_difference) AS maximum_partial_and_ideal_cooling_load_difference,
        MAX(t.mechanical_setpoint_difference) AS maximum_mechanical_setpoint_percent_difference,
        MAX(t.ideal_setpoint_difference) AS maximum_ideal_setpoint_percent_difference,
        MAX(t.partial_setpoint_difference) AS maximum_partial_setpoint_percent_difference,
        MAX(t.ideal_and_mechanical_temperature_percent_difference) AS maximum_ideal_and_mechanical_temperature_percent_difference,
        MAX(t.partial_and_ideal_temperature_percent_difference) AS maximum_partial_and_ideal_temperature_percent_difference,
        MAX(t.ideal_and_mechanical_heating_load_percent_difference) AS maximum_ideal_and_mechanical_heating_load_percent_difference,
        MAX(t.partial_and_ideal_heating_load_percent_difference) AS maximum_partial_and_ideal_heating_load_percent_difference,
        MAX(t.ideal_and_mechanical_cooling_load_percent_difference) AS maximum_ideal_and_mechanical_cooling_load_percent_difference,
        MAX(t.partial_and_ideal_cooling_load_percent_difference) AS maximum_partial_and_ideal_cooling_load_percent_difference,
        AVG(t.setpoint) AS average_setpoint,
        AVG(t.mechanical_temperature) AS average_mechanical_temperature,
        AVG(t.ideal_temperature) AS average_ideal_temperature,
        AVG(t.partial_temperature) AS average_partial_temperature,
        AVG(t.mechanical_heating_load) AS average_mechanical_heating_load,
        AVG(t.ideal_heating_load) AS average_ideal_heating_load,
        AVG(t.partial_heating_load) AS average_partial_heating_load,
        AVG(t.mechanical_cooling_load) AS average_mechanical_cooling_load,
        AVG(t.ideal_cooling_load) AS average_ideal_cooling_load,
        AVG(t.partial_cooling_load) AS average_partial_cooling_load,
        AVG(t.mechanical_setpoint_difference) AS average_mechanical_setpoint_difference,
        AVG(t.ideal_setpoint_difference) AS average_ideal_setpoint_difference,
        AVG(t.partial_setpoint_difference) AS average_partial_setpoint_difference,
        AVG(t.ideal_and_mechanical_temperature_difference) AS average_ideal_and_mechanical_temperature_difference,
        AVG(t.partial_and_ideal_temperature_difference) AS average_partial_and_ideal_temperature_difference,
        AVG(t.ideal_and_mechanical_heating_load_difference) AS average_ideal_and_mechanical_heating_load_difference,
        AVG(t.partial_and_ideal_heating_load_difference) AS average_partial_and_ideal_heating_load_difference,
        AVG(t.ideal_and_mechanical_cooling_load_difference) AS average_ideal_and_mechanical_cooling_load_difference,
        AVG(t.partial_and_ideal_cooling_load_difference) AS average_partial_and_ideal_cooling_load_difference,
        AVG(t.mechanical_setpoint_difference) AS average_mechanical_setpoint_percent_difference,
        AVG(t.ideal_setpoint_difference) AS average_ideal_setpoint_percent_difference,
        AVG(t.partial_setpoint_difference) AS average_partial_setpoint_percent_difference,
        AVG(t.ideal_and_mechanical_temperature_percent_difference) AS average_ideal_and_mechanical_temperature_percent_difference,
        AVG(t.partial_and_ideal_temperature_percent_difference) AS average_partial_and_ideal_temperature_percent_difference,
        AVG(t.ideal_and_mechanical_heating_load_percent_difference) AS average_ideal_and_mechanical_heating_load_percent_difference,
        AVG(t.partial_and_ideal_heating_load_percent_difference) AS average_partial_and_ideal_heating_load_percent_difference,
        AVG(t.ideal_and_mechanical_cooling_load_percent_difference) AS average_ideal_and_mechanical_cooling_load_percent_difference,
        AVG(t.partial_and_ideal_cooling_load_percent_difference) AS average_partial_and_ideal_cooling_load_percent_difference
    FROM dynamic_lstm_train_data_hourly_summary t
    GROUP BY
        t.metadata_id,
        t.bldg_id,
        t.in_resstock_county_id
;

-- DROP VIEW IF EXISTS energyplus_simulation_monthly_unmet_hour_summary;
-- CREATE VIEW energyplus_simulation_monthly_unmet_hour_summary AS
--     WITH t AS (
--         SELECT
--             m.id AS metadata_id,
--             m.bldg_id,
--             m.in_resstock_county_id,
--             l.month,
--             MIN(l.timestep) AS min_timestep,
--             MAX(l.timestep) AS max_timestep,
--             COUNT(l.timestep) AS count_timestep,
--             TOTAL(CASE WHEN ABS(l.average_indoor_air_temperature - l.setpoint) > 2.0 THEN 1 END) AS count_unmet_hour,
--             TOTAL(CASE WHEN l.average_indoor_air_temperature - l.setpoint < -2.0 THEN 1 END) AS count_cold_hour,
--             TOTAL(CASE WHEN l.average_indoor_air_temperature - l.setpoint > 2.0 THEN 1 END) AS count_hot_hour,
--             MIN(l.average_indoor_air_temperature - l.setpoint) AS min_delta,
--             MAX(l.average_indoor_air_temperature - l.setpoint) AS max_delta,
--             AVG(l.average_indoor_air_temperature - l.setpoint) AS avg_delta
--         FROM lstm_train_data l
--         LEFT JOIN energyplus_simulation s ON s.id = l.simulation_id
--         LEFT JOIN metadata m ON m.id = s.metadata_id
--         WHERE l.simulation_id IN (
--             SELECT id FROM energyplus_simulation WHERE 
--                 reference = 0 
--                 AND metadata_id IN (SELECT metadata_id FROM thermal_zone_count WHERE value = 1)
--         )
--         GROUP BY
--             m.id,
--             m.bldg_id,
--             m.in_resstock_county_id,
--             l.month
--     )

--     SELECT
--         t.metadata_id,
--         t.bldg_id,
--         t.in_resstock_county_id,
--         t.month,
--         l.label AS cluster_label,
--         t.min_timestep,
--         t.max_timestep,
--         t.count_timestep,
--         t.count_unmet_hour,
--         t.count_cold_hour,
--         t.count_hot_hour,
--         t.count_unmet_hour*100.0/t.count_timestep AS percent_unmet_hour,
--         t.count_cold_hour*100.0/t.count_timestep AS percent_cold_hour,
--         t.count_hot_hour*100.0/t.count_timestep AS percent_hot_hour,
--         t.min_delta,
--         t.max_delta,
--         t.avg_delta
--     FROM t
--     LEFT JOIN (
--         SELECT
--             metadata_id,
--             label
--         FROM metadata_clustering_label
--         WHERE clustering_id IN (SELECT clustering_id FROM optimal_metadata_clustering)
--     ) l ON l.metadata_id = t.metadata_id
-- ;