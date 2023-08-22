SELECT
    r.TimeIndex AS timestep,
    'cooling_load' AS load,
    z.ZoneIndex AS zone_index,
    z.ZoneName AS zone_name,
    ABS(r.Value)/1000.0 AS value
FROM ReportData r
INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
LEFT JOIN Zones z ON z.ZoneName = REPLACE(d.KeyValue, ' IDEAL LOADS AIR SYSTEM', '')
WHERE d.Name = 'Zone Ideal Loads Zone Sensible Cooling Rate' AND z.ZoneName IN (<cooled_zone_names>)

UNION ALL

SELECT
    r.TimeIndex AS timestep,
    'heating_load' AS load,
    z.ZoneIndex AS zone_index,
    z.ZoneName AS zone_name,
    ABS(r.Value)/1000.0 AS value
FROM ReportData r
INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
LEFT JOIN Zones z ON z.ZoneName = REPLACE(d.KeyValue, ' IDEAL LOADS AIR SYSTEM', '')
WHERE d.Name = 'Zone Ideal Loads Zone Sensible Heating Rate' AND z.ZoneName IN (<heated_zone_names>)