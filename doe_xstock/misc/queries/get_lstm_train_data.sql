SELECT
    m.in_resstock_county_id AS "location",
    m.bldg_id AS resstock_building_id,
    b.name AS ecobee_building_id,
    s.reference AS simulation_reference,
    l.timestep,
    l.month,
    l.day,
    l.day_of_week,
    l.hour,
    l.minute,
    l.direct_solar_radiation,
    l.diffuse_solar_radiation,
    l.outdoor_air_temperature,
    l.average_indoor_air_temperature,
    l.occupant_count,
    l.cooling_load,
    l.heating_load,
    l.setpoint
FROM lstm_train_data l
LEFT JOIN energyplus_simulation s ON
    s.id = l.simulation_id
LEFT JOIN ecobee_building b ON b.id = s.ecobee_building_id
LEFT JOIN metadata m ON m.id = s.metadata_id
WHERE l.simulation_id IN (
    SELECT id FROM energyplus_simulation WHERE metadata_id = <metadata_id>
)
;