-- PRAGMA foreign_keys = ON;
-- DELETE FROM energyplus_simulation;
-- DELETE FROM energyplus_simulation_error;
-- DELETE FROM static_lstm_train_data;
-- DELETE FROM dynamic_lstm_train_data;
-- SELECT * FROM energyplus_simulation_error;
-- SELECT osm FROM model WHERE metadata_id = 494;

-- SELECT * FROM metadata_clustering where id IN (SELECT clustering_id FROM optimal_metadata_clustering);

-- SELECT 
--     bldg_id,
--     in_resstock_county_id,
--     one_degree_mechanical_unmet,
--     one_degree_ideal_unmet,
--     one_degree_partial_unmet
-- FROM dynamic_lstm_train_data_thermal_comfort_summary 
-- WHERE 
--     timestep_resolution = 'monthly'
-- ORDER BY one_degree_partial_unmet DESC;

-- SELECT * FROM dynamic_lstm_train_data_multi_resolution_summary WHERE timestep_resolution = 'yearly' AND in_resstock_county_id = 'TX, Travis County' ORDER BY maximum_partial_setpoint_absolute_difference DESC;

-- SELECT * FROM dynamic_lstm_train_data_hourly_summary WHERE metadata_id = 990;

-- AND metadata_id IN (
-- SELECT metadata_id FROM thermal_zone_count WHERE "value" = 1
-- ) 

-- SELECT
--     m.in_resstock_county_id,
--     c.value,
--     COUNT(c.metadata_id)
-- FROM thermal_zone_count c
-- INNER JOIN (
-- SELECT DISTINCT
-- metadata_id
-- FROM energyplus_simulation
-- ) s ON s.metadata_id = c.metadata_id
-- LEFT JOIN metadata m ON m.id = c.metadata_id
-- GROUP BY
-- m.in_resstock_county_id,
-- c.value
-- ;

