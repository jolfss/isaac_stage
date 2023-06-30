#-----------#
#   Skies   #
#-----------#

class Skies:
    # Omniverse's default dynamic skyboxes.
    DEFAULT_DYNAMIC_SKIES= {sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Dynamic/{sky_name}.usd" \
                                    for sky_name in ["Cirrus", "ClearSky", "CloudySky", "CumulusHeavy", "CumulusLight", "NightSky", "Overcast"]}

    # Omniverse's default HDR-imaged, static skyboxes.
    __DEFAULT_HDRI_CLEAR_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Clear/{sky_name}.hdr" \
                                    for sky_name in ["evening_road_01_4k", "kloppenheim_02_4k", "mealie_road_4k", "noon_grass_4k", 
                                                        "qwantani_4k", "signal_hill_sunrise_4k", "sunflowers_4k","syferfontein_18d_clear_4k",
                                                        "venice_sunset_4k","white_cliff_top_4k"]}
    __DEFAULT_HDRI_CLOUDY_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Cloudy/{sky_name}.hdr" \
                                    for sky_name in ["abandoned_parking_4k", "champagne_castle_1_4k", "evening_road_01_4k", 
                                                        "kloofendal_48d_partly_cloudy_4k", "lakeside_4k", "sunflowers_4k", 
                                                        "table_mountain_1_4k"]}
    __DEFAULT_HDRI_EVENING_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Evening/{sky_name}.hdr" \
                                    for sky_name in ["evening_road_01_4k"]}
    __DEFAULT_HDRI_INDOOR_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Indoor/{sky_name}.hdr" \
                                    for sky_name in ["adams_place_bridge_4k","autoshop_01_4k","bathroom_4k","carpentry_shop_01_4k",
                                                        "en_suite_4k","entrance_hall_4k","hospital_room_4k","hotel_room_4k","lebombo_4k",
                                                        "old_bus_depot_4k","small_empty_house_4k","studio_small_04_4k","surgery_4k",
                                                        "vulture_hide_4k","wooden_lounge_4k","ZetoCG_com_WarehouseInterior2b",
                                                        "ZetoCGcom_Exhibition_Hall_Interior1"]}
    __DEFAULT_HDRI_NIGHT_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Night/{sky_name}.hdr" \
                                    for sky_name in ["kloppenheim_02_4k","moonlit_golf_4k"]}
    __DEFAULT_HDRI_STORM_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Storm/{sky_name}.hdr" \
                                    for sky_name in ["approaching_storm_4k"]}
    __DEFAULT_HDRI_STUDIO_SKIES={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Studio/{sky_name}.hdr" \
                                    for sky_name in ["photo_studio_01_4k","studio_small_05_4k","studio_small_07_4k"]}

    # Primary dictionary containing the above sub-dictionaries.
    DEFAULT_HDRI_SKIES = { "Clear" : __DEFAULT_HDRI_CLEAR_SKIES,"Cloudy" : __DEFAULT_HDRI_CLOUDY_SKIES,"Evening" : __DEFAULT_HDRI_EVENING_SKIES,
                            "Indoor" : __DEFAULT_HDRI_INDOOR_SKIES,"Night" : __DEFAULT_HDRI_NIGHT_SKIES, "Storm" : __DEFAULT_HDRI_STORM_SKIES,
                            "Studio" : __DEFAULT_HDRI_STUDIO_SKIES,}
    
#-----------#
#   Skies   #
#-----------#

class Textures:
    pass