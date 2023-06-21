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

    -- thermal load variables
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'thermal_load' AS label,
        r.Value
    FROM ReportData r
    LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    LEFT JOIN Zones z ON z.ZoneName = d.KeyValue
    WHERE 
        d.Name IN (
            'Zone Air System Sensible Cooling Rate', 
            'Zone Air System Sensible Heating Rate', 
            'Zone Thermostat Cooling Setpoint Temperature'
        )

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
            'Water Heater Use Side Heat Transfer Energy',
            'Exterior Lights Electricity Energy',
            'Lights Electricity Energy',
            'Electric Equipment Electricity Energy',
            'Zone People Occupant Count'
        )
), p AS (
    SELECT
        u.TimeIndex,
        SUM(CASE WHEN d.Name = 'Zone Air Temperature' THEN Value END) AS "Indoor Temperature (C)",
        SUM(CASE WHEN d.Name = 'Zone Air Relative Humidity' THEN Value END) AS "Indoor Relative Humidity (%)",
        SUM(CASE WHEN d.Name IN ('Exterior Lights Electricity Energy', 'Lights Electricity Energy', 'Electric Equipment Electricity Energy') THEN Value/(3600.0*1000.0) END) AS "Equipment Electric Power (kWh)",
        SUM(CASE WHEN d.Name = 'Water Heater Use Side Heat Transfer Energy' THEN ABS(Value)/(3600.0*1000.0) END) AS "DHW Heating (kWh)",
        SUM(CASE WHEN d.Name = 'Zone Air System Sensible Cooling Rate' THEN ABS(Value)/(1000.0) END) AS "Cooling Load (kWh)",
        SUM(CASE WHEN d.Name = 'Zone Air System Sensible Heating Rate' THEN ABS(Value)/(1000.0) END) AS "Heating Load (kWh)",
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
    CASE 
        WHEN COALESCE(p."Cooling Load (kWh)", 0.0) >= COALESCE(p."Heating Load (kWh)", 0.0)
            THEN COALESCE(p."Cooling Load (kWh)", 0.0) - COALESCE(p."Heating Load (kWh)", 0.0)
                ELSE 0.0
                    END AS "Cooling Load (kWh)",
    CASE 
        WHEN COALESCE(p."Heating Load (kWh)", 0.0) > COALESCE(p."Cooling Load (kWh)", 0.0)
            THEN COALESCE(p."Heating Load (kWh)", 0.0) - COALESCE(p."Cooling Load (kWh)", 0.0)
                ELSE 0.0
                    END AS "Heating Load (kWh)",
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