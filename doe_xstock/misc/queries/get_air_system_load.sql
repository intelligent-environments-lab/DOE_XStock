SELECT
    d.Name as name,
    SUM(r.Value) AS value
FROM ReportData r
INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
WHERE d.Name IN ('Zone Air System Sensible Cooling Rate', 'Zone Air System Sensible Heating Rate')
GROUP BY d.Name