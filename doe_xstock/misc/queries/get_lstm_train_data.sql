SELECT
    m.in_resstock_county_id AS "location",
    m.bldg_id AS resstock_building_id,
    b.name AS ecobee_building_id,
    i.reference AS simulation_reference,
    s.timestep,
    s.month,
    d.day,
    d.day_of_week,
    s.hour,
    s.minute,
    s.direct_solar_radiation,
    s.diffuse_solar_radiation,
    s.outdoor_air_temperature,
    d.average_indoor_air_temperature,
    s.occupant_count,
    d.cooling_load,
    d.heating_load,
    s.setpoint
FROM dynamic_lstm_train_data d
LEFT JOIN (
    SELECT 
        * 
    FROM static_lstm_train_data 
    WHERE metadat_id = <metadata_id>
) s ON s.timestep = d.timestep
LEFT JOIN energyplus_simulation i ON i.id = d.simulation_id
LEFT JOIN ecobee_building b ON b.id = i.ecobee_building_id
LEFT JOIN metadata m ON m.id = i.metadata_id
WHERE d.simulation_id IN (
    SELECT 
        id 
    FROM energyplus_simulation 
    WHERE metadata_id = <metadata_id> AND reference > 1
)
;