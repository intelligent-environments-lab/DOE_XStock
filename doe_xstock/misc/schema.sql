-- tables
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY NOT NULL,
    dataset_type TEXT NOT NULL,
    weather_data TEXT NOT NULL,
    year_of_publication INTEGER NOT NULL,
    release INTEGER NOT NULL,
    UNIQUE (dataset_type, weather_data, year_of_publication, release)
);

CREATE TABLE IF NOT EXISTS weather (
    id INTEGER PRIMARY KEY NOT NULL,
    dataset_id INTEGER NOT NULL REFERENCES dataset(id),
    weather_file_tmy3 TEXT NOT NULL,
    weather_file_latitude TEXT NOT NULL,
    weather_file_longitude TEXT NOT NULL,
    energyplus_title TEXT NOT NULL,
    epw_url TEXT NOT NULL,
    ddy_url TEXT NOT NULL,
    epw TEXT NOT NULL,
    ddy TEXT NOT NULL,
    UNIQUE (dataset_id, weather_file_tmy3, weather_file_latitude, weather_file_longitude)
);

CREATE TABLE IF NOT EXISTS data_dictionary (
    id INTEGER PRIMARY KEY NOT NULL,
    dataset_id INTEGER NOT NULL REFERENCES dataset(id),
    field_location TEXT NOT NULL,
    field_name TEXT NOT NULL,
    data_type TEXT NOT NULL,
    units TEXT,
    field_description TEXT,
    allowable_enumerations TEXT,
    UNIQUE (dataset_id, field_location, field_name)
);

CREATE TABLE IF NOT EXISTS upgrade_dictionary (
    id INTEGER PRIMARY KEY NOT NULL,
    dataset_id INTEGER NOT NULL REFERENCES dataset(id),
    upgrade_id INTEGER NOT NULL,
    upgrade_name TEXT NOT NULL,
    upgrade_description TEXT NOT NULL,
    UNIQUE (dataset_id, upgrade_id)
);

CREATE TABLE IF NOT EXISTS spatial_tract (
    id INTEGER PRIMARY KEY NOT NULL,
    dataset_id INTEGER NOT NULL REFERENCES dataset(id),
    nhgis_tract_gisjoin TEXT NOT NULL,
    nhgis_county_gisjoin TEXT NOT NULL,
    nhgis_puma_gisjoin TEXT NOT NULL,
    state_name TEXT NOT NULL,
    state_abbreviation TEXT NOT NULL,
    census_division_name TEXT NOT NULL,
    census_region_name TEXT NOT NULL,
    census_division_name_recs TEXT NOT NULL,
    american_housing_survey_region TEXT NOT NULL,
    weather_file_2018 TEXT,
    weather_file_tmy3 TEXT,
    climate_zone_building_america TEXT NOT NULL,
    climate_zone_ashrae_2004 TEXT NOT NULL,
    iso_region TEXT NOT NULL,
    reeds_balancing_area REAL,
    resstock_county_id TEXT NOT NULL,
    resstock_puma_id TEXT NOT NULL,
    resstock_custom_region TEXT NOT NULL,
    UNIQUE (dataset_id, nhgis_tract_gisjoin)
);

CREATE TABLE IF NOT EXISTS metadata (
    id INTEGER PRIMARY KEY NOT NULL,
    bldg_id INTEGER NOT NULL,
    dataset_id INTEGER NOT NULL REFERENCES dataset(id),
    weather_id INTEGER NOT NULL REFERENCES weather(id),
    in_county TEXT,
    in_puma TEXT,
    in_ashrae_iecc_climate_zone_2004 TEXT,
    in_building_america_climate_zone TEXT,
    in_iso_rto_region TEXT,
    applicability INTEGER,
    "weight" REAL,
    in_sqft REAL,
    in_ahs_region TEXT,
    in_applicable INTEGER,
    in_bathroom_spot_vent_hour TEXT,
    in_bedrooms TEXT,
    in_cec_climate_zone TEXT,
    in_ceiling_fan TEXT,
    in_census_division TEXT,
    in_census_division_recs TEXT,
    in_census_region TEXT,
    in_clothes_dryer TEXT,
    in_clothes_washer TEXT,
    in_clothes_washer_presence TEXT,
    in_cooking_range TEXT,
    in_cooling_setpoint TEXT,
    in_cooling_setpoint_has_offset TEXT,
    in_cooling_setpoint_offset_magnitude TEXT,
    in_cooling_setpoint_offset_period TEXT,
    in_corridor TEXT,
    in_dehumidifier TEXT,
    in_dishwasher TEXT,
    in_door_area TEXT,
    in_doors TEXT,
    in_ducts TEXT,
    in_eaves TEXT,
    in_electric_vehicle TEXT,
    in_geometry_attic_type TEXT,
    in_geometry_building_horizontal_location_mf TEXT,
    in_geometry_building_horizontal_location_sfa TEXT,
    in_geometry_building_level_mf TEXT,
    in_geometry_building_number_units_mf TEXT,
    in_geometry_building_number_units_sfa TEXT,
    in_geometry_building_type_acs TEXT,
    in_geometry_building_type_height TEXT,
    in_geometry_building_type_recs TEXT,
    in_geometry_floor_area TEXT,
    in_geometry_floor_area_bin TEXT,
    in_geometry_foundation_type TEXT,
    in_geometry_garage TEXT,
    in_geometry_stories TEXT,
    in_geometry_stories_low_rise TEXT,
    in_geometry_wall_exterior_finish TEXT,
    in_geometry_wall_type TEXT,
    in_geometry_wall_type_and_exterior_finish TEXT,
    in_has_pv TEXT,
    in_heating_fuel TEXT,
    in_heating_setpoint TEXT,
    in_heating_setpoint_has_offset TEXT,
    in_heating_setpoint_offset_magnitude TEXT,
    in_heating_setpoint_offset_period TEXT,
    in_holiday_lighting TEXT,
    in_hot_water_distribution TEXT,
    in_hot_water_fixtures TEXT,
    in_hvac_cooling_efficiency TEXT,
    in_hvac_cooling_type TEXT,
    in_hvac_has_ducts TEXT,
    in_hvac_has_shared_system TEXT,
    in_hvac_has_zonal_electric_heating TEXT,
    in_hvac_heating_efficiency TEXT,
    in_hvac_heating_type TEXT,
    in_hvac_heating_type_and_fuel TEXT,
    in_hvac_secondary_heating_efficiency TEXT,
    in_hvac_secondary_heating_type_and_fuel TEXT,
    in_hvac_shared_efficiencies TEXT,
    in_hvac_system_is_faulted TEXT,
    in_hvac_system_single_speed_ac_airflow TEXT,
    in_hvac_system_single_speed_ac_charge TEXT,
    in_hvac_system_single_speed_ashp_airflow TEXT,
    in_hvac_system_single_speed_ashp_charge TEXT,
    in_infiltration TEXT,
    in_insulation_ceiling TEXT,
    in_insulation_floor TEXT,
    in_insulation_foundation_wall TEXT,
    in_insulation_roof TEXT,
    in_insulation_slab TEXT,
    in_insulation_wall TEXT,
    in_interior_shading TEXT,
    in_lighting TEXT,
    in_lighting_interior_use TEXT,
    in_lighting_other_use TEXT,
    in_location_region TEXT,
    in_mechanical_ventilation TEXT,
    in_misc_extra_refrigerator TEXT,
    in_misc_freezer TEXT,
    in_misc_gas_fireplace TEXT,
    in_misc_gas_grill TEXT,
    in_misc_gas_lighting TEXT,
    in_misc_hot_tub_spa TEXT,
    in_misc_pool TEXT,
    in_misc_pool_heater TEXT,
    in_misc_pool_pump TEXT,
    in_misc_well_pump TEXT,
    in_natural_ventilation TEXT,
    in_neighbors TEXT,
    in_occupants TEXT,
    in_orientation TEXT,
    in_overhangs TEXT,
    in_plug_load_diversity TEXT,
    in_plug_loads TEXT,
    in_pv_orientation TEXT,
    in_pv_system_size TEXT,
    in_radiant_barrier TEXT,
    in_range_spot_vent_hour TEXT,
    in_reeds_balancing_area TEXT,
    in_refrigerator TEXT,
    in_roof_material TEXT,
    in_schedules TEXT,
    in_setpoint_demand_response TEXT,
    in_solar_hot_water TEXT,
    in_state TEXT,
    in_units_represented REAL,
    in_usage_level TEXT,
    in_vacancy_status TEXT,
    in_vintage TEXT,
    in_vintage_acs TEXT,
    in_water_heater_efficiency TEXT,
    in_water_heater_fuel TEXT,
    in_water_heater_in_unit TEXT,
    in_weather_file_city TEXT,
    in_weather_file_latitude TEXT,
    in_weather_file_longitude TEXT,
    in_window_areas TEXT,
    in_windows TEXT,
    in_nhgis_county_gisjoin TEXT,
    in_nhgis_puma_gisjoin TEXT,
    in_state_name TEXT,
    in_american_housing_survey_region TEXT,
    in_weather_file_2018 TEXT,
    in_weather_file_tmy3 TEXT,
    in_resstock_county_id TEXT,
    in_resstock_puma_id TEXT,
    out_electricity_bath_fan_energy_consumption REAL,
    out_electricity_bath_fan_energy_consumption_intensity REAL,
    out_electricity_ceiling_fan_energy_consumption REAL,
    out_electricity_ceiling_fan_energy_consumption_intensity REAL,
    out_electricity_clothes_dryer_energy_consumption REAL,
    out_electricity_clothes_dryer_energy_consumption_intensity REAL,
    out_electricity_clothes_washer_energy_consumption REAL,
    out_electricity_clothes_washer_energy_consumption_intensity REAL,
    out_electricity_cooking_range_energy_consumption REAL,
    out_electricity_cooking_range_energy_consumption_intensity REAL,
    out_electricity_cooling_energy_consumption REAL,
    out_electricity_cooling_energy_consumption_intensity REAL,
    out_electricity_dishwasher_energy_consumption REAL,
    out_electricity_dishwasher_energy_consumption_intensity REAL,
    out_electricity_ext_holiday_light_energy_consumption REAL,
    out_electricity_ext_holiday_light_energy_consumption_intensity REAL,
    out_electricity_exterior_lighting_energy_consumption REAL,
    out_electricity_exterior_lighting_energy_consumption_intensity REAL,
    out_electricity_extra_refrigerator_energy_consumption REAL,
    out_electricity_extra_refrigerator_energy_consumption_intensity REAL,
    out_electricity_fans_cooling_energy_consumption REAL,
    out_electricity_fans_cooling_energy_consumption_intensity REAL,
    out_electricity_fans_heating_energy_consumption REAL,
    out_electricity_fans_heating_energy_consumption_intensity REAL,
    out_electricity_freezer_energy_consumption REAL,
    out_electricity_freezer_energy_consumption_intensity REAL,
    out_electricity_garage_lighting_energy_consumption REAL,
    out_electricity_garage_lighting_energy_consumption_intensity REAL,
    out_electricity_heating_energy_consumption REAL,
    out_electricity_heating_energy_consumption_intensity REAL,
    out_electricity_heating_supplement_energy_consumption REAL,
    out_electricity_heating_supplement_energy_consumption_intensity REAL,
    out_electricity_hot_tub_heater_energy_consumption REAL,
    out_electricity_hot_tub_heater_energy_consumption_intensity REAL,
    out_electricity_hot_tub_pump_energy_consumption REAL,
    out_electricity_hot_tub_pump_energy_consumption_intensity REAL,
    out_electricity_house_fan_energy_consumption REAL,
    out_electricity_house_fan_energy_consumption_intensity REAL,
    out_electricity_interior_lighting_energy_consumption REAL,
    out_electricity_interior_lighting_energy_consumption_intensity REAL,
    out_electricity_plug_loads_energy_consumption REAL,
    out_electricity_plug_loads_energy_consumption_intensity REAL,
    out_electricity_pool_heater_energy_consumption REAL,
    out_electricity_pool_heater_energy_consumption_intensity REAL,
    out_electricity_pool_pump_energy_consumption REAL,
    out_electricity_pool_pump_energy_consumption_intensity REAL,
    out_electricity_pumps_cooling_energy_consumption REAL,
    out_electricity_pumps_cooling_energy_consumption_intensity REAL,
    out_electricity_pumps_heating_energy_consumption REAL,
    out_electricity_pumps_heating_energy_consumption_intensity REAL,
    out_electricity_pv_energy_consumption REAL,
    out_electricity_pv_energy_consumption_intensity REAL,
    out_electricity_range_fan_energy_consumption REAL,
    out_electricity_range_fan_energy_consumption_intensity REAL,
    out_electricity_recirc_pump_energy_consumption REAL,
    out_electricity_recirc_pump_energy_consumption_intensity REAL,
    out_electricity_refrigerator_energy_consumption REAL,
    out_electricity_refrigerator_energy_consumption_intensity REAL,
    out_electricity_vehicle_energy_consumption REAL,
    out_electricity_vehicle_energy_consumption_intensity REAL,
    out_electricity_water_systems_energy_consumption REAL,
    out_electricity_water_systems_energy_consumption_intensity REAL,
    out_electricity_well_pump_energy_consumption REAL,
    out_electricity_well_pump_energy_consumption_intensity REAL,
    out_fuel_oil_heating_energy_consumption REAL,
    out_fuel_oil_heating_energy_consumption_intensity REAL,
    out_fuel_oil_water_systems_energy_consumption REAL,
    out_fuel_oil_water_systems_energy_consumption_intensity REAL,
    out_natural_gas_clothes_dryer_energy_consumption REAL,
    out_natural_gas_clothes_dryer_energy_consumption_intensity REAL,
    out_natural_gas_cooking_range_energy_consumption REAL,
    out_natural_gas_cooking_range_energy_consumption_intensity REAL,
    out_natural_gas_fireplace_energy_consumption REAL,
    out_natural_gas_fireplace_energy_consumption_intensity REAL,
    out_natural_gas_grill_energy_consumption REAL,
    out_natural_gas_grill_energy_consumption_intensity REAL,
    out_natural_gas_heating_energy_consumption REAL,
    out_natural_gas_heating_energy_consumption_intensity REAL,
    out_natural_gas_hot_tub_heater_energy_consumption REAL,
    out_natural_gas_hot_tub_heater_energy_consumption_intensity REAL,
    out_natural_gas_lighting_energy_consumption REAL,
    out_natural_gas_lighting_energy_consumption_intensity REAL,
    out_natural_gas_pool_heater_energy_consumption REAL,
    out_natural_gas_pool_heater_energy_consumption_intensity REAL,
    out_natural_gas_water_systems_energy_consumption REAL,
    out_natural_gas_water_systems_energy_consumption_intensity REAL,
    out_propane_clothes_dryer_energy_consumption REAL,
    out_propane_clothes_dryer_energy_consumption_intensity REAL,
    out_propane_cooking_range_energy_consumption REAL,
    out_propane_cooking_range_energy_consumption_intensity REAL,
    out_propane_heating_energy_consumption REAL,
    out_propane_heating_energy_consumption_intensity REAL,
    out_propane_water_systems_energy_consumption REAL,
    out_propane_water_systems_energy_consumption_intensity REAL,
    out_electricity_total_energy_consumption REAL,
    out_electricity_total_energy_consumption_intensity REAL,
    out_fuel_oil_total_energy_consumption REAL,
    out_fuel_oil_total_energy_consumption_intensity REAL,
    out_natural_gas_total_energy_consumption REAL,
    out_natural_gas_total_energy_consumption_intensity REAL,
    out_propane_total_energy_consumption REAL,
    out_propane_total_energy_consumption_intensity REAL,
    out_wood_total_energy_consumption REAL,
    out_wood_total_energy_consumption_intensity REAL,
    out_wood_heating_energy_consumption REAL,
    out_wood_heating_energy_consumption_intensity REAL,
    out_site_energy_total_energy_consumption REAL,
    out_site_energy_total_energy_consumption_intensity REAL,
    qoi_report_average_maximum_daily_timing_cooling_hour REAL,
    qoi_report_average_maximum_daily_timing_heating_hour REAL,
    qoi_report_average_maximum_daily_timing_overlap_hour REAL,
    qoi_report_average_maximum_daily_use_cooling_kw REAL,
    qoi_report_average_maximum_daily_use_heating_kw REAL,
    qoi_report_average_maximum_daily_use_overlap_kw REAL,
    qoi_report_average_minimum_daily_use_cooling_kw REAL,
    qoi_report_average_minimum_daily_use_heating_kw REAL,
    qoi_report_average_minimum_daily_use_overlap_kw REAL,
    qoi_report_average_of_top_ten_highest_peaks_timing_cooling_hour REAL,
    qoi_report_average_of_top_ten_highest_peaks_timing_heating_hour REAL,
    qoi_report_average_of_top_ten_highest_peaks_use_cooling_kw REAL,
    qoi_report_average_of_top_ten_highest_peaks_use_heating_kw REAL,
    qoi_report_peak_magnitude_timing_hour REAL,
    qoi_report_peak_magnitude_use_kw REAL,
    in_door_area_ft_2 REAL,
    in_duct_unconditioned_surface_area_ft_2 REAL,
    in_floor_area_attic_ft_2 REAL,
    in_floor_area_conditioned_ft_2 REAL,
    in_floor_area_lighting_ft_2 REAL,
    in_roof_area_ft_2 REAL,
    in_wall_area_above_grade_conditioned_ft_2 REAL,
    in_wall_area_above_grade_exterior_ft_2 REAL,
    in_wall_area_below_grade_ft_2 REAL,
    in_window_area_ft_2 REAL,
    upgrade INTEGER NOT NULL,
    metadata_index INTEGER,
    UNIQUE (bldg_id, dataset_id, upgrade)
);

CREATE TABLE IF NOT EXISTS timeseries (
    metadata_id INTEGER NOT NULL REFERENCES metadata(id),
    "timestamp" TEXT,
    bldg_id INTEGER NOT NULL,
    out_electricity_bath_fan_energy_consumption REAL,
    out_electricity_bath_fan_energy_consumption_intensity REAL,
    out_electricity_ceiling_fan_energy_consumption REAL,
    out_electricity_ceiling_fan_energy_consumption_intensity REAL,
    out_electricity_clothes_dryer_energy_consumption REAL,
    out_electricity_clothes_dryer_energy_consumption_intensity REAL,
    out_electricity_clothes_washer_energy_consumption REAL,
    out_electricity_clothes_washer_energy_consumption_intensity REAL,
    out_electricity_cooking_range_energy_consumption REAL,
    out_electricity_cooking_range_energy_consumption_intensity REAL,
    out_electricity_cooling_energy_consumption REAL,
    out_electricity_cooling_energy_consumption_intensity REAL,
    out_electricity_dishwasher_energy_consumption REAL,
    out_electricity_dishwasher_energy_consumption_intensity REAL,
    out_electricity_ext_holiday_light_energy_consumption REAL,
    out_electricity_ext_holiday_light_energy_consumption_intensity REAL,
    out_electricity_exterior_lighting_energy_consumption REAL,
    out_electricity_exterior_lighting_energy_consumption_intensity REAL,
    out_electricity_extra_refrigerator_energy_consumption REAL,
    out_electricity_extra_refrigerator_energy_consumption_intensity REAL,
    out_electricity_fans_cooling_energy_consumption REAL,
    out_electricity_fans_cooling_energy_consumption_intensity REAL,
    out_electricity_fans_heating_energy_consumption REAL,
    out_electricity_fans_heating_energy_consumption_intensity REAL,
    out_electricity_freezer_energy_consumption REAL,
    out_electricity_freezer_energy_consumption_intensity REAL,
    out_electricity_garage_lighting_energy_consumption REAL,
    out_electricity_garage_lighting_energy_consumption_intensity REAL,
    out_electricity_heating_energy_consumption REAL,
    out_electricity_heating_energy_consumption_intensity REAL,
    out_electricity_heating_supplement_energy_consumption REAL,
    out_electricity_heating_supplement_energy_consumption_intensity REAL,
    out_electricity_hot_tub_heater_energy_consumption REAL,
    out_electricity_hot_tub_heater_energy_consumption_intensity REAL,
    out_electricity_hot_tub_pump_energy_consumption REAL,
    out_electricity_hot_tub_pump_energy_consumption_intensity REAL,
    out_electricity_house_fan_energy_consumption REAL,
    out_electricity_house_fan_energy_consumption_intensity REAL,
    out_electricity_interior_lighting_energy_consumption REAL,
    out_electricity_interior_lighting_energy_consumption_intensity REAL,
    out_electricity_plug_loads_energy_consumption REAL,
    out_electricity_plug_loads_energy_consumption_intensity REAL,
    out_electricity_pool_heater_energy_consumption REAL,
    out_electricity_pool_heater_energy_consumption_intensity REAL,
    out_electricity_pool_pump_energy_consumption REAL,
    out_electricity_pool_pump_energy_consumption_intensity REAL,
    out_electricity_pumps_cooling_energy_consumption REAL,
    out_electricity_pumps_cooling_energy_consumption_intensity REAL,
    out_electricity_pumps_heating_energy_consumption REAL,
    out_electricity_pumps_heating_energy_consumption_intensity REAL,
    out_electricity_pv_energy_consumption REAL,
    out_electricity_pv_energy_consumption_intensity REAL,
    out_electricity_range_fan_energy_consumption REAL,
    out_electricity_range_fan_energy_consumption_intensity REAL,
    out_electricity_recirc_pump_energy_consumption REAL,
    out_electricity_recirc_pump_energy_consumption_intensity REAL,
    out_electricity_refrigerator_energy_consumption REAL,
    out_electricity_refrigerator_energy_consumption_intensity REAL,
    out_electricity_vehicle_energy_consumption REAL,
    out_electricity_vehicle_energy_consumption_intensity REAL,
    out_electricity_water_systems_energy_consumption REAL,
    out_electricity_water_systems_energy_consumption_intensity REAL,
    out_electricity_well_pump_energy_consumption REAL,
    out_electricity_well_pump_energy_consumption_intensity REAL,
    out_fuel_oil_heating_energy_consumption REAL,
    out_fuel_oil_heating_energy_consumption_intensity REAL,
    out_fuel_oil_water_systems_energy_consumption REAL,
    out_fuel_oil_water_systems_energy_consumption_intensity REAL,
    out_natural_gas_clothes_dryer_energy_consumption REAL,
    out_natural_gas_clothes_dryer_energy_consumption_intensity REAL,
    out_natural_gas_cooking_range_energy_consumption REAL,
    out_natural_gas_cooking_range_energy_consumption_intensity REAL,
    out_natural_gas_fireplace_energy_consumption REAL,
    out_natural_gas_fireplace_energy_consumption_intensity REAL,
    out_natural_gas_grill_energy_consumption REAL,
    out_natural_gas_grill_energy_consumption_intensity REAL,
    out_natural_gas_heating_energy_consumption REAL,
    out_natural_gas_heating_energy_consumption_intensity REAL,
    out_natural_gas_hot_tub_heater_energy_consumption REAL,
    out_natural_gas_hot_tub_heater_energy_consumption_intensity REAL,
    out_natural_gas_lighting_energy_consumption REAL,
    out_natural_gas_lighting_energy_consumption_intensity REAL,
    out_natural_gas_pool_heater_energy_consumption REAL,
    out_natural_gas_pool_heater_energy_consumption_intensity REAL,
    out_natural_gas_water_systems_energy_consumption REAL,
    out_natural_gas_water_systems_energy_consumption_intensity REAL,
    out_propane_clothes_dryer_energy_consumption REAL,
    out_propane_clothes_dryer_energy_consumption_intensity REAL,
    out_propane_cooking_range_energy_consumption REAL,
    out_propane_cooking_range_energy_consumption_intensity REAL,
    out_propane_heating_energy_consumption REAL,
    out_propane_heating_energy_consumption_intensity REAL,
    out_propane_water_systems_energy_consumption REAL,
    out_propane_water_systems_energy_consumption_intensity REAL,
    out_electricity_total_energy_consumption REAL,
    out_electricity_total_energy_consumption_intensity REAL,
    out_fuel_oil_total_energy_consumption REAL,
    out_fuel_oil_total_energy_consumption_intensity REAL,
    out_natural_gas_total_energy_consumption REAL,
    out_natural_gas_total_energy_consumption_intensity REAL,
    out_propane_total_energy_consumption REAL,
    out_propane_total_energy_consumption_intensity REAL,
    out_wood_total_energy_consumption REAL,
    out_wood_total_energy_consumption_intensity REAL,
    out_wood_heating_energy_consumption REAL,
    out_wood_heating_energy_consumption_intensity REAL,
    out_site_energy_total_energy_consumption REAL,
    out_site_energy_total_energy_consumption_intensity REAL,
    PRIMARY KEY (metadata_id, "timestamp")
);

CREATE TABLE IF NOT EXISTS model (
    id INTEGER PRIMARY KEY NOT NULL,
    metadata_id INTEGER NOT NULL UNIQUE REFERENCES metadata(id),
    osm TEXT
);

CREATE TABLE IF NOT EXISTS schedule (
    metadata_id INTEGER NOT NULL REFERENCES metadata(id),
    timestep INTEGER NOT NULL,
    occupants REAL NOT NULL,
    lighting_interior REAL NOT NULL,
    lighting_exterior REAL NOT NULL,
    lighting_garage REAL NOT NULL,
    lighting_exterior_holiday REAL NOT NULL,
    cooking_range REAL NOT NULL,
    dishwasher REAL NOT NULL,
    dishwasher_power REAL NOT NULL,
    clothes_washer REAL NOT NULL,
    clothes_washer_power REAL NOT NULL,
    clothes_dryer REAL NOT NULL,
    clothes_dryer_exhaust REAL NOT NULL,
    baths REAL NOT NULL,
    showers REAL NOT NULL,
    sinks REAL NOT NULL,
    ceiling_fan REAL NOT NULL,
    plug_loads REAL NOT NULL,
    plug_loads_vehicle REAL NOT NULL,
    plug_loads_well_pump REAL NOT NULL,
    fuel_loads_grill REAL NOT NULL,
    fuel_loads_lighting REAL NOT NULL,
    fuel_loads_fireplace REAL NOT NULL,
    sleep REAL NOT NULL,
    vacancy REAL NOT NULL,
    PRIMARY KEY (metadata_id, timestep)
);

CREATE TABLE IF NOT EXISTS ecobee_location (
    id INTEGER PRIMARY KEY NOT NULL,
    "name" TEXT NOT NULL,
    UNIQUE ("name")
);

CREATE TABLE IF NOT EXISTS ecobee_building (
    id INTEGER PRIMARY KEY NOT NULL,
    location_id INTEGER NOT NULL,
    "name" TEXT NOT NULL,
    FOREIGN KEY (location_id) REFERENCES ecobee_location (id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    UNIQUE ("name")
);

CREATE TABLE IF NOT EXISTS ecobee_timeseries (
    building_id INTEGER NOT NULL,
    timestep INTEGER NOT NULL,
    setpoint REAL NOT NULL,
    PRIMARY KEY (building_id, timestep)
    FOREIGN KEY (building_id) REFERENCES ecobee_building (id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- views
DROP VIEW IF EXISTS xstock_ecobee_pair;
CREATE VIEW xstock_ecobee_pair AS
    WITH x AS (
        SELECT
            ((e.row_number - 1) % e.building_count) + 1 AS ecobee_row_number,
            e.metadata_id,
            e.location
        FROM (
            SELECT
                ROW_NUMBER() OVER(PARTITION BY n.name ORDER BY s.metadata_id ASC) AS row_number,
                e.building_count,
                s.metadata_id,
                n.name AS location
            FROM building_to_simulate_in_energyplus s
            INNER JOIN metadata_clustering_name n ON n.id = s.name_id
            INNER JOIN (
                SELECT
                    l.name,
                    COUNT(b.name) AS building_count
                FROM ecobee_building b
                INNER JOIN ecobee_location l ON l.id = b.location_id
                GROUP BY l.name
            ) e ON e.name = n.name
        ) e
    ), e AS (
        SELECT
            ROW_NUMBER() OVER(PARTITION BY l.name ORDER BY b.name ASC) AS row_number,
            b.id AS building_id,
            l.name AS location
        FROM ecobee_building b
        INNER JOIN ecobee_location l ON l.id = b.location_id
    )

    SELECT
        x.location,
        x.metadata_id AS xstock_metadata_id,
        e.building_id AS ecobee_building_id
    FROM x
    INNER JOIN e ON 
        e.location = x.location 
        AND e.row_number = x.ecobee_row_number
;

DROP VIEW IF EXISTS energyplus_simulation_input;
CREATE VIEW energyplus_simulation_input AS
    SELECT
        m.id AS metadata_id,
        d.dataset_type,
        d.weather_data AS dataset_weather_data,
        d.year_of_publication AS dataset_year_of_publication,
        d.release AS dataset_release,
        m.bldg_id,
        c.ecobee_building_id AS ecobee_bldg_id,
        m.upgrade AS bldg_upgrade,
        e.osm AS bldg_osm,
        w.epw AS bldg_epw
    FROM metadata m
    INNER JOIN dataset d ON d.id = m.dataset_id
    INNER JOIN model e ON e.metadata_id = m.id
    INNER JOIN weather w ON w.id = m.weather_id
    LEFT JOIN xstock_ecobee_pair c ON c.xstock_metadata_id = m.id
;

CREATE VIEW thermal_zone_count AS
    SELECT
        metadata_id,
        (
            LENGTH(osm) 
            - LENGTH(REPLACE(osm, 'OS:ThermostatSetpoint:DualSetpoint', ''))
        )/LENGTH('OS:ThermostatSetpoint:DualSetpoint') AS value
    from model
;