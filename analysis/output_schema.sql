WITH zone_conditioning AS (
    SELECT
        KeyValue AS "zone",
        SUM(CASE WHEN "Name" = 'Zone Thermostat Cooling Setpoint Temperature' AND average_setpoint > 0 THEN 1 ELSE 0 END) AS is_cooled,
        SUM(CASE WHEN "Name" = 'Zone Thermostat Heating Setpoint Temperature' AND average_setpoint > 0 THEN 1 ELSE 0 END) AS is_heated,
        SUM(CASE WHEN "Name" = 'Zone Thermostat Cooling Setpoint Temperature' AND average_setpoint > 0 THEN average_setpoint END) AS average_cooling_setpoint,
        SUM(CASE WHEN "Name" = 'Zone Thermostat Heating Setpoint Temperature' AND average_setpoint > 0 THEN average_setpoint END) AS average_heating_setpoint
    FROM (
        SELECT
            d.KeyValue,
            d.Name,
            AVG(r."value") AS average_setpoint
        FROM ReportData r
        INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        WHERE d.Name IN ('Zone Thermostat Cooling Setpoint Temperature', 'Zone Thermostat Heating Setpoint Temperature')
        GROUP BY d.KeyValue, d.Name
    )
    GROUP BY KeyValue

), zone AS (
    -- get zone floor area proportion of total zone floor area
    SELECT
        z.ZoneName,
        z.Multiplier,
        z.Volume,
        z.FloorArea,
        (z.FloorArea*z.Multiplier)/t.total_floor_area AS total_floor_area_proportion,
        CASE WHEN c.is_cooled != 0 OR c.is_heated != 0 THEN (z.FloorArea*z.Multiplier)/t.conditioned_floor_area ELSE 0 END AS conditioned_floor_area_proportion,
        c.is_cooled,
        c.is_heated
    FROM Zones z
    CROSS JOIN (
        SELECT
            SUM(z.FloorArea*z.Multiplier) AS total_floor_area,
            SUM(CASE WHEN c.is_cooled != 0 OR c.is_heated != 0 THEN z.FloorArea*z.Multiplier ELSE 0 END) AS conditioned_floor_area
        FROM Zones z
        LEFT JOIN zone_conditioning c ON c.zone = z.ZoneName
    ) t
    LEFT JOIN zone_conditioning c ON c.zone = z.ZoneName

), unioned_variables AS (
    -- weighted_cooling/heating_setpoint_difference
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'setpoint_difference' AS label,
        ABS(s.value - r.Value)*z.conditioned_floor_area_proportion AS "value"
    FROM ReportData r
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    INNER JOIN "zone" z ON z.ZoneName = d.KeyValue
    INNER JOIN (
        SELECT
            r.TimeIndex,
            d.KeyValue,
            r.Value AS "value"
        FROM ReportData r
        INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        WHERE
            d.Name IN ('Zone Air Temperature')
    ) s ON 
        s.TimeIndex = r.TimeIndex
        AND s.KeyValue = d.KeyValue
    WHERE
        (d.Name = 'Zone Thermostat Cooling Setpoint Temperature' AND z.is_cooled != 0)
        OR (d.Name = 'Zone Thermostat Heating Setpoint Temperature' AND z.is_heated != 0)

    UNION

        -- other_weighted_average_variable
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'indoor_environment' AS label,
        r.Value*z.conditioned_floor_area_proportion AS "value"
    FROM ReportData r
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    INNER JOIN "zone" z ON z.ZoneName = d.KeyValue
    WHERE
        (d.Name = 'Zone Air Temperature' AND (z.is_cooled != 0 OR z.is_heated != 0))
        OR (d.Name = 'Zone Air Relative Humidity' AND (z.is_cooled != 0 OR z.is_heated != 0))
        OR (d.Name = 'Zone Thermostat Cooling Setpoint Temperature' AND (z.is_cooled != 0 OR z.is_heated != 0))
        OR (d.Name = 'Zone Thermostat Heating Setpoint Temperature' AND (z.is_cooled != 0 OR z.is_heated != 0))

    UNION

    -- domestic hot water and plug loads
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'electrical_load' AS label,
        r.Value AS "value"
    FROM ReportData r
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE d.Name IN (
        'Water Heater Heating Energy',
        'Exterior Lights Electricity Energy',
        'Lights Electricity Energy',
        'Electric Equipment Electricity Energy'
    )

    UNION

    -- cooling and heating loads
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'predicted_sensible_cooling_load' AS label,
        CASE
            WHEN r.Value > 0 THEN 0
            ELSE ABS(r.Value)
        END  AS "value"
    FROM ReportData r
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE d.Name = 'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate'
    UNION
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'predicted_sensible_heating_load' AS label,
        CASE
            WHEN r.Value < 0 THEN 0
            ELSE ABS(r.Value)
        END  AS "value"
    FROM ReportData r
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE d.Name = 'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate'
    UNION
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'ideal_sensible_cooling_load' AS label,
        r.Value AS "value"
    FROM ReportData r
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE d.Name = 'Zone Ideal Loads Zone Sensible Cooling Rate'
    UNION
    SELECT
        r.TimeIndex,
        r.ReportDataDictionaryIndex,
        'ideal_sensible_heating_load' AS label,
        r.Value AS "value"
    FROM ReportData r
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
    WHERE d.Name = 'Zone Ideal Loads Zone Sensible Heating Rate'

), "aggregate" AS (
    -- sum the variables per timestamp
    SELECT
        u.TimeIndex,
        d.Name,
        u.label,
        (u.value) AS value
    FROM unioned_variables u
    INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = u.ReportDataDictionaryIndex
    GROUP BY
        u.TimeIndex,
        label,
        d.Name
), aggregate_pivot AS (
    -- pivot table to match CityLearn input format
    SELECT
        a.TimeIndex,
        SUM(CASE WHEN a.Name = 'Zone Air Temperature' THEN a.value END) AS "Indoor Temperature (C)",
        SUM(CASE WHEN a.Name = 'Zone Thermostat Cooling Setpoint Temperature'
            AND a.label = 'setpoint_difference' THEN a.value END
        ) AS "Average Unmet Cooling Setpoint Difference (C)",
        SUM(CASE WHEN a.Name = 'Zone Thermostat Heating Setpoint Temperature'
            AND a.label = 'setpoint_difference' THEN a.value END
        ) AS "Average Unmet Heating Setpoint Difference (C)",
        SUM(CASE WHEN a.Name = 'Zone Air Relative Humidity' THEN a.value END) AS "Indoor Relative Humidity (%)",
        SUM(CASE WHEN a.Name = 'Water Heater Heating Energy' THEN (a.value/3600)/1000 END) AS "DHW Heating (kW)",
        SUM(CASE WHEN a.label = 'predicted_sensible_cooling_load' THEN a.value/1000 END) AS "Predicted Sensible Cooling Load (kW)",
        SUM(CASE WHEN a.label = 'predicted_sensible_heating_load' THEN a.value/1000 END) AS "Predicted Sensible Heating Load (kW)",
        SUM(CASE WHEN a.label = 'ideal_sensible_cooling_load' THEN a.value/1000 END) AS "Ideal Sensible Cooling Load (kW)",
        SUM(CASE WHEN a.label = 'ideal_sensible_heating_load' THEN a.value/1000 END) AS "Ideal Sensible Heating Load (kW)",
        SUM(CASE WHEN a.Name = 'Zone Thermostat Cooling Setpoint Temperature' THEN a.value END) AS "Cooling Setpoint (C)",
        SUM(CASE WHEN a.Name = 'Zone Thermostat Heating Setpoint Temperature' THEN a.value END) AS "Heating Setpoint (C)",
        SUM(CASE WHEN a.Name IN (
            'Exterior Lights Electricity Energy', 'Lights Electricity Energy', 'Electric Equipment Electricity Energy') THEN (a.value/3600)/1000 END
        ) AS "Equipment Electric Power (kW)"
    FROM aggregate a
    GROUP BY
        a.TimeIndex
)

-- define time-related columns
SELECT
    t.TimeIndex AS 'Timestep',
    t.Month,
    t.Day,
    t.Hour,
    t.Minute,
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
    a."Indoor Temperature (C)",
    a."Cooling Setpoint (C)",
    a."Heating Setpoint (C)",
    a."Average Unmet Cooling Setpoint Difference (C)",
    a."Average Unmet Heating Setpoint Difference (C)",
    a."Indoor Relative Humidity (%)",
    a."Equipment Electric Power (kW)",
    a."DHW Heating (kW)",
    a."Predicted Sensible Cooling Load (kW)",
    a."Predicted Sensible Heating Load (kW)",
    a."Ideal Sensible Cooling Load (kW)",
    a."Ideal Sensible Heating Load (kW)"
FROM aggregate_pivot a
INNER JOIN Time t ON t.TimeIndex = a.TimeIndex
WHERE
    t.DayType NOT IN ('SummerDesignDay', 'WinterDesignDay')
ORDER BY
    t.TimeIndex