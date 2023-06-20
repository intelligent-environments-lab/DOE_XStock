SELECT
    MAX(CASE WHEN d.Name = 'Site Outdoor Air Drybulb Temperature' THEN r.value END) AS "Outdoor Drybulb Temperature (C)",
    MAX(CASE WHEN d.Name = 'Site Outdoor Air Relative Humidity' THEN r.value END) AS "Outdoor Relative Humidity (%)",
    MAX(CASE WHEN d.Name = 'Site Diffuse Solar Radiation Rate per Area' THEN r.value END) AS "Diffuse Solar Radiation (W/m2)",
    MAX(CASE WHEN d.Name = 'Site Direct Solar Radiation Rate per Area' THEN r.value END) AS "Direct Solar Radiation (W/m2)"
FROM ReportData r
LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
LEFT JOIN Time t ON t.TimeIndex = r.TimeIndex
WHERE t.DayType NOT IN ('SummerDesignDay', 'WinterDesignDay')
GROUP BY t.TimeIndex
ORDER BY t.TimeIndex
;