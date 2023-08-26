WITH u AS (
    -- weighted conditioned zone variables
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'weighted_variable' AS label,
        r.Value
    FROM weighted_variable r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE d.Name IN ('Zone Air Temperature', 'Zone Air Relative Humidity')

    UNION ALL

    -- setpoint variables
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'setpoint' AS label,
        r.Value
    FROM ReportData r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    LEFT JOIN Zones z ON z.ZoneName = d.KeyValue
    WHERE 
        d.Name IN ('Zone Thermostat Cooling Setpoint Temperature')

    UNION ALL

    --  thermal load
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        CASE WHEN r.Value > 0 THEN 'heating_load' ELSE 'cooling_load' END AS label,
        ABS(r.Value) AS Value
    FROM ReportData r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE 
        d.Name = 'Other Equipment Convective Heating Rate' AND
        (d.KeyValue LIKE '%HEATING_LOAD' OR d.KeyValue LIKE '%COOLING_LOAD')

    UNION ALL

    -- other variables
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'other' AS label,
        r.Value
    FROM ReportData r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE 
        d.Name IN (
            'Water Use Equipment Heating Rate',
            'Zone Lights Electricity Rate',
            'Zone Electric Equipment Electricity Rate',
            'Zone People Occupant Count'
        )
), p AS (
    SELECT
        u.TimeIndex,
        SUM(CASE WHEN d.Name = 'Zone Air Temperature' THEN Value END) AS "Indoor Temperature (C)",
        SUM(CASE WHEN d.Name = 'Zone Air Relative Humidity' THEN Value END) AS "Indoor Relative Humidity (%)",
        SUM(CASE WHEN d.Name IN ('Zone Lights Electricity Rate', 'Zone Electric Equipment Electricity Rate') THEN Value/(1000.0) END) AS "Equipment Electric Power (kWh)",
        SUM(CASE WHEN d.Name = 'Water Use Equipment Heating Rate' THEN ABS(Value)/(1000.0) END) AS "DHW Heating (kWh)",
        SUM(CASE WHEN u.label = 'cooling_load' THEN Value/1000.0 END) AS cooling_load,
        SUM(CASE WHEN u.label = 'heating_load' THEN Value/1000.0 END) AS heating_load,
        SUM(CASE WHEN d.Name = 'Zone People Occupant Count' THEN Value END) AS "Occupancy",
        MAX(CASE WHEN d.Name = 'Zone Thermostat Cooling Setpoint Temperature' THEN Value END) AS "Temperature Set Point (C)"
    FROM u
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = u.ReportDataDictionaryIndex
    GROUP BY u.TimeIndex
)

SELECT
    t.Month AS "Month",
    t.Hour AS "Hour",
    CASE
        WHEN t.DayType = 'Monday' THEN 1
            WHEN t.DayType = 'Tuesday' THEN 2
                WHEN t.DayType = 'Wednesday' THEN 3
                    WHEN t.DayType = 'Thursday' THEN 4
                        WHEN t.DayType = 'Friday' THEN 5
                            WHEN t.DayType = 'Saturday' THEN 6
                                WHEN t.DayType = 'Sunday' THEN 7
                                    WHEN t.DayType = 'Holiday' THEN 8
                                        ELSE NULL
                                            END AS "Day Type",
    t.Dst AS "Daylight Savings Status",
    p."Indoor Temperature (C)",
    NULL AS "Average Unmet Cooling Setpoint Difference (C)",
    p."Indoor Relative Humidity (%)",
    p."Equipment Electric Power (kWh)",
    p."DHW Heating (kWh)",
    COALESCE(p."Cooling Load (kWh)", 0.0) AS "Cooling Load (kWh)",
    COALESCE(p."Heating Load (kWh)", 0.0) AS "Heating Load (kWh)",
    NULL AS "Solar Generation (W/kW)",
    p."Occupancy" AS "Occupant Count (people)",
    p."Temperature Set Point (C)",
    CASE 
        WHEN COALESCE(p."Cooling Load (kWh)", 0.0) 
            >= COALESCE(p."Heating Load (kWh)", 0.0) THEN 1 
                ELSE 2 
                    END AS "HVAC Mode (Off/Cooling/Heating)"
FROM p
LEFT JOIN Time t ON t.TimeIndex = p.TimeIndex
WHERE t.DayType NOT IN ('SummerDesignDay', 'WinterDesignDay')
ORDER BY t.TimeIndex
;