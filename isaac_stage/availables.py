import omni

#-----------#
#   Skies   #
#-----------#

class Skies:
    # Omniverse's default dynamic skyboxes.
    default_dyanmic_skies= {sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Dynamic/{sky_name}.usd" \
                                    for sky_name in ["Cirrus", "ClearSky", "CloudySky", "CumulusHeavy", "CumulusLight", "NightSky", "Overcast"]}

    # Omniverse's default HDR-imaged, static skyboxes.
    _default_hdri_clear_skies={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Clear/{sky_name}.hdr" \
                                    for sky_name in ["evening_road_01_4k", "kloppenheim_02_4k", "mealie_road_4k", "noon_grass_4k", 
                                                        "qwantani_4k", "signal_hill_sunrise_4k", "sunflowers_4k","syferfontein_18d_clear_4k",
                                                        "venice_sunset_4k","white_cliff_top_4k"]}
    _default_hdri_cloudy_skies={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Cloudy/{sky_name}.hdr" \
                                    for sky_name in ["abandoned_parking_4k", "champagne_castle_1_4k", "evening_road_01_4k", 
                                                        "kloofendal_48d_partly_cloudy_4k", "lakeside_4k", "sunflowers_4k", 
                                                        "table_mountain_1_4k"]}
    _default_hdri_evening_skies={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Evening/{sky_name}.hdr" \
                                    for sky_name in ["evening_road_01_4k"]}
    _default_hdri_indoor_skies={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Indoor/{sky_name}.hdr" \
                                    for sky_name in ["adams_place_bridge_4k","autoshop_01_4k","bathroom_4k","carpentry_shop_01_4k",
                                                        "en_suite_4k","entrance_hall_4k","hospital_room_4k","hotel_room_4k","lebombo_4k",
                                                        "old_bus_depot_4k","small_empty_house_4k","studio_small_04_4k","surgery_4k",
                                                        "vulture_hide_4k","wooden_lounge_4k","ZetoCG_com_WarehouseInterior2b",
                                                        "ZetoCGcom_Exhibition_Hall_Interior1"]}
    _default_hdri_night_skies={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Night/{sky_name}.hdr" \
                                    for sky_name in ["kloppenheim_02_4k","moonlit_golf_4k"]}
    _default_hdri_storm_skies={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Storm/{sky_name}.hdr" \
                                    for sky_name in ["approaching_storm_4k"]}
    _default_hdri_studio_skies={sky_name : F"https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Skies/2022_1/Skies/Studio/{sky_name}.hdr" \
                                    for sky_name in ["photo_studio_01_4k","studio_small_05_4k","studio_small_07_4k"]}

    # Primary dictionary containing the above sub-dictionaries.
    default_hdri_skies = { "Clear" : _default_hdri_clear_skies,"Cloudy" : _default_hdri_cloudy_skies,"Evening" : _default_hdri_evening_skies,
                            "Indoor" : _default_hdri_indoor_skies,"Night" : _default_hdri_night_skies, "Storm" : _default_hdri_storm_skies,
                            "Studio" : _default_hdri_studio_skies,}
    
#---------------#
#   Materials   #
#---------------#

class Materials:
    omni.kit.commands.execute('CreateMdlMaterialPrimCommand',
	mtl_url='http://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/Base/Natural/Dirt.mdl',
	mtl_name='Dirt',
	mtl_path='/World/Looks/Dirt')

    omni.kit.commands.execute('BindMaterialCommand',
        prim_path='/World/Sphere',
        material_path='/World/Looks/Dirt')
