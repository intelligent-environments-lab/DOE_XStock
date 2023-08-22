SELECT
    r.TimeIndex AS timestep,
    SUM(r.Value) AS value
FROM weighted_variable r
LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
WHERE d.Name IN ('Zone Air Temperature')
GROUP BY r.TimeIndex