SELECT
    r.TimeIndex AS timestep,
    'cooling_load' AS load,
    z.ZoneIndex AS zone_index,
    z.ZoneName AS zone_name,
    ABS(CASE WHEN r.Value > 0 THEN 0 ELSE r.Value END)/1000.0 AS value
FROM ReportData r
INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
LEFT JOIN Zones z ON z.ZoneName = d.KeyValue
WHERE d.Name = 'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate' AND z.ZoneName IN (<cooled_zone_names>)

UNION ALL

SELECT
    r.TimeIndex AS timestep,
    'heating_load' AS load,
    z.ZoneIndex AS zone_index,
    z.ZoneName AS zone_name,
    ABS(CASE WHEN r.Value < 0 THEN 0 ELSE r.Value END)/1000.0 AS value
FROM ReportData r
INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
LEFT JOIN Zones z ON z.ZoneName = d.KeyValue
WHERE d.Name = 'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate' AND z.ZoneName IN (<heated_zone_names>)