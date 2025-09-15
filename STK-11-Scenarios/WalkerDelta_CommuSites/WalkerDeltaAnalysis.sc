stk.v.11.0
WrittenBy    STK_v11.2.0
BEGIN Scenario
    Name            WalkerDeltaAnalysis

BEGIN Epoch

    Epoch        25 Aug 2025 00:00:00.000000000
    SmartEpoch
	BEGIN	EVENT
			Epoch	25 Aug 2025 00:00:00.000000000
			EventEpoch
				BEGIN	EVENT
					Type	EVENT_LINKTO
					Name	AnalysisStartTime
				END	EVENT
			EpochState	Implicit
	END	EVENT


END Epoch

BEGIN Interval

Start                   25 Aug 2025 00:00:00.000000000
Stop                    26 Aug 2025 00:00:00.000000000
    SmartInterval
	BEGIN	EVENTINTERVAL
			StartEvent
				BEGIN	EVENT
						Epoch	25 Aug 2025 00:00:00.000000000
						EpochState	Explicit
				END	EVENT
			Duration		+ 1 day
			IntervalState	StartDuration
	END	EVENTINTERVAL

EpochUsesAnalStart      No
AnimStartUsesAnalStart  Yes
AnimStopUsesAnalStop    Yes

END Interval

BEGIN EOPFile

    EOPFilename     EOP-v1.1.txt

END EOPFile

BEGIN GlobalPrefs

    SatelliteNoOrbWarning    No
    MissilePerigeeWarning    No
    MissileStopTimeWarning   No
    AircraftWGS84Warning     Always
END GlobalPrefs

BEGIN CentralBody

    PrimaryBody     Earth

END CentralBody

BEGIN CentralBodyTerrain

    BEGIN CentralBody
        Name            Earth
        UseTerrainCache Yes
        TotalCacheSize  402653184

        BEGIN StreamingTerrain
            UseCurrentStreamingTerrainServer     Yes
            CurrentStreamingTerrainServerName    http://twsusecovacc01.agi.com/stk-terrain
            StreamingTerrainTilesetName    world
            StreamingTerrainServerName           assets.agi.com/stk-terrain/
            StreamingTerrainAzimuthElevationMaskEnabled       No
            StreamingTerrainObscurationEnabled       No
            StreamingTerrainCoverageGridObscurationEnabled       No
        END StreamingTerrain
    END CentralBody

END CentralBodyTerrain

BEGIN StarCollection

    Name     Hipparcos 2 Mag 6

END StarCollection

BEGIN ScenarioLicenses
    Module    AMMv11.2
    Module    ASTGv11.2
    Module    CATv11.2
    Module    CHAINSv11.2
    Module    CONv11.2
    Module    COVv11.2
    Module    CRMv11.2
    Module    Commv11.2
    Module    DISv11.2
    Module    EOIRv11.2
    Module    HRMv11.2
    Module    MexServv11.2
    Module    RT3Clientv11.2
    Module    RdrAdvEnv11.2
    Module    SEETv11.2
    Module    SOLISv11.2
    Module    STKCAP
    Module    STKExpertv11.2
    Module    STKIntegrationv11.2
    Module    STKParallelComputingv11.2
    Module    STKProfessionalv11.2
    Module    STKv11.2
    Module    TERNv11.2
    Module    TIREMv11.2
    Module    UPropv11.2
    Module    Underseav11.2
END ScenarioLicenses

BEGIN QuickReports

    BEGIN Report
        Name    Grid Stats
        Type    Report
        BaseDir Install
        Style    Grid Stats
        AGIViewer    Yes
        Instance    CoverageDefinition/TH_Cov/FigureOfMerit/CovTimeTotal
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    2
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    3
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     No
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    339
        WindowRectTop     245
        WindowRectRight   2289
        WindowRectBottom  1020
    END Report

    BEGIN Report
        Name    Grid Stats Over Time
        Type    Report
        BaseDir Install
        Style    Grid Stats Over Time
        AGIViewer    Yes
        Instance    CoverageDefinition/TH_Cov/FigureOfMerit/SimpleCov_vs_Time
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    2
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    3
                SectionType      2
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     No
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    1157
        WindowRectTop     165
        WindowRectRight   1855
        WindowRectBottom  1215
    END Report

    BEGIN Report
        Name    Grid Stats1
        Type    Report
        BaseDir Install
        Style    Grid Stats
        AGIViewer    Yes
        Instance    CoverageDefinition/TH_Cov/FigureOfMerit/TimeAvgGap
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    2
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    3
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     No
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    676
        WindowRectTop     473
        WindowRectRight   2517
        WindowRectBottom  1185
    END Report

    BEGIN Report
        Name    Grid Stats2
        Type    Report
        BaseDir Install
        Style    Grid Stats
        AGIViewer    Yes
        Instance    CoverageDefinition/TH_Cov/FigureOfMerit/CovTimeTotal
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    2
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    3
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     No
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    410
        WindowRectTop     207
        WindowRectRight   2251
        WindowRectBottom  921
    END Report

    BEGIN Report
        Name    Grid Stats3
        Type    Report
        BaseDir Install
        Style    Grid Stats
        AGIViewer    Yes
        Instance    CoverageDefinition/TH_Cov/FigureOfMerit/CovTimeTotal
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    2
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    3
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     No
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    524
        WindowRectTop     321
        WindowRectRight   2365
        WindowRectBottom  1035
    END Report

    BEGIN Report
        Name    Grid Stats4
        Type    Report
        BaseDir Install
        Style    Grid Stats
        AGIViewer    Yes
        Instance    CoverageDefinition/TH_Cov/FigureOfMerit/TimeAvgGap
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    2
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    3
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     No
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    562
        WindowRectTop     359
        WindowRectRight   2403
        WindowRectBottom  1073
    END Report

    BEGIN Report
        Name    AER
        Type    Report
        BaseDir Install
        Style    AER
        AGIViewer    Yes
        Instance    Facility/Communication_Site/Receiver/Uplink_Ka_Rx
        BEGIN InstanceList
            Instance    Satellite/LOGSAT1101/Sensor/Sensor1/Transmitter/Transmitter_SensorFOV
        END InstanceList
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      2
                ShowIntervals    No
                TimeType    Availability
                SamplingType    Default
                Step        60.000000
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     No
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    524
        WindowRectTop     321
        WindowRectRight   2365
        WindowRectBottom  1035
    END Report

    BEGIN Report
        Name    AER1
        Type    Report
        BaseDir Install
        Style    AER
        AGIViewer    Yes
        Instance    Facility/Communication_Site/Receiver/Uplink_Ka_Rx
        BEGIN InstanceList
            Instance    Satellite/LOGSAT1101/Transmitter/Downlink_Ka_Tx
        END InstanceList
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      2
                ShowIntervals    No
                TimeType    Availability
                SamplingType    Default
                Step        60.000000
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     No
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    562
        WindowRectTop     359
        WindowRectRight   2403
        WindowRectBottom  1073
    END Report

    BEGIN Report
        Name    Coverage By Latitude
        Type    Report
        BaseDir Install
        Style    Coverage By Latitude
        AGIViewer    Yes
        Instance    CoverageDefinition/TH_Cov
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
            BEGIN Section
                SectionNumber    2
                SectionType      1
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Interval
                TimeInterval 	IntervalTimePeriod
                SamplingType    Default
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     No
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    524
        WindowRectTop     321
        WindowRectRight   2365
        WindowRectBottom  1035
    END Report

    BEGIN Report
        Name    Link Budget
        Type    Report
        BaseDir Install
        Style    Link Budget
        AGIViewer    Yes
        Instance    Facility/Communication_Site/Receiver/Uplink_Ka_Rx
        BEGIN InstanceList
            Instance    Satellite/LOGSAT1101/Sensor/Sensor1/Transmitter/Transmitter_SensorFOV
        END InstanceList
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      2
                ShowIntervals    No
                TimeType    Availability
                SamplingType    Default
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     No
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    562
        WindowRectTop     359
        WindowRectRight   2403
        WindowRectBottom  1073
    END Report

    BEGIN Report
        Name    Carrier_to_Noise_Ratio
        Type    Graph
        BaseDir Install
        Style    Carrier_to_Noise_Ratio
        AGIViewer    No
        Instance    Facility/Communication_Site/Receiver/Uplink_Ka_Rx
        BEGIN InstanceList
            Instance    Satellite/LOGSAT1101/Sensor/Sensor1/Transmitter/Transmitter_SensorFOV
        END InstanceList
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      2
                ShowIntervals    No
BEGIN IntervalList

DateUnitAbrv UTCG

BEGIN Intervals

    "25 Aug 2025 00:00:00.000000000" "26 Aug 2025 00:00:00.000000000"
END Intervals

END IntervalList

                TimeType    Availability
                SamplingType    Default
                Step        60.000000
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     Yes
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    1424
        WindowRectTop     736
        WindowRectRight   3265
        WindowRectBottom  1447
    END Report

    BEGIN Report
        Name    Link Budget1
        Type    Report
        BaseDir Install
        Style    Link Budget
        AGIViewer    Yes
        Instance    Facility/Communication_Site/Receiver/Uplink_Ka_Rx
        BEGIN InstanceList
            Instance    Satellite/LOGSAT1101/Sensor/Sensor1/Transmitter/Transmitter_SensorFOV
        END InstanceList
        BEGIN TimeData
            BEGIN Section
                SectionNumber    1
                SectionType      2
                ShowIntervals    No
                TimeType    Availability
                SamplingType    Default
                TimeBound    0
            END Section
        END TimeData
        DisplayOnLoad     Yes
        FrameType         0
        DockCircleID      0
        DockID            0
        WindowRectLeft    1071
        WindowRectTop     349
        WindowRectRight   2262
        WindowRectBottom  1045
    END Report
END QuickReports

BEGIN WebData
        EnableWebTerrainData    No
        SaveWebTerrainDataPasswords    No
        BEGIN ConfigServerDataList
            BEGIN ConfigServerData
                Name "globeserver.agi.com"
                Port 80
                DataURL "bin/getGlobeSvrConfig.pl"
            END ConfigServerData
        END ConfigServerDataList
END WebData

BEGIN Extensions
    
    BEGIN ClsApp
		RangeConstraint         5000.000
		ApoPeriPad              30000.000
		OrbitPathPad            100000.000
		TimeDistPad             30000.000
		OutOfDate               2592000.000
		MaxApoPeriStep          900.000
		ApoPeriAngle            0.785
		UseApogeePerigeeFilter  Yes
		UsePathFilter           No
		UseTimeFilter           No
		UseOutOfDate            Yes
		CreateSats              No
		MaxSatsToCreate         500
		UseModelScale           No
		ModelScale              0.000
		UseCrossRefDb           Yes
		CollisionDB                     stkAllTLE.tce
		CollisionCrossRefDB             stkAllTLE.sd
		ShowLine                Yes
		AnimHighlight           Yes
		StaticHighlight         Yes
		UseLaunchWindow                         No
		LaunchWindowUseEntireTraj               Yes
		LaunchWindowTrajMETStart                0.000
		LaunchWindowTrajMETStop                 900.000
		LaunchWindowStart                       10800.000
		LaunchWindowStop                        10800.000
		LaunchMETOffset                         0.000
		LaunchWindowUseSecEphem                 No 
		LaunchWindowUseScenFolderForSecEphem    Yes
		LaunchWindowUsePrimEphem                No 
		LaunchWindowUseScenFolderForPrimEphem   Yes
    LaunchWindowIntervalPtr
	BEGIN	EVENTINTERVAL
			BEGIN Interval
				Start	25 Aug 2025 03:00:00.000000000
				Stop	26 Aug 2025 03:00:00.000000000
			END Interval
			IntervalState	Explicit
	END	EVENTINTERVAL

		LaunchWindowUsePrimMTO                  No 
		GroupLaunches                           No 
		LWTimeConvergence                       1.000e-03
		LWRelValueConvergence                   1.000e-08
		LWTSRTimeConvergence                    1.000e-04
		LWTSRRelValueConvergence                1.000e-10
		LaunchWindowStep                        300.000
		MaxTSRStep                              180.000
		MaxTSRRelMotion                         20.000
		UseLaunchArea                           No 
		LaunchAreaOrientation                   North
		LaunchAreaAzimuth                       0.000
		LaunchAreaXLimits                       -10000.000   10000.000
		LaunchAreaYLimits                       -10000.000   10000.000
		LaunchAreaNumXIntrPnts                  1
		LaunchAreaNumYIntrPnts                  1
		LaunchAreaAltReference                  Ellipsoid
		TargetSameStop                          No 
		SkipSurfaceMetric                       No 
		LWAreaTSRRelValueConvergence            1.000e-10
		AreaLaunchWindowStep                    300.000
		AreaMaxTSRStep                          30.000
		AreaMaxTSRRelMotion                     1.000
		ShowLaunchArea                          No 
		ShowBlackoutTracks                      No 
		ShowClearedTracks                       No 
		UseObjectForClearedColor                No 
		BlackoutColor                           #ff0000
		ClearedColor                             #ffffff
		ShowTracksSegments                      Yes
		ShowMinRangeTracks                      Yes
		MinRangeTrackTimeStep                   0.500000
		UsePrimStepForTracks                    Yes
		GfxTracksTimeStep                       30.000
		GfxAreaNumXIntrPnts                     1
		GfxAreaNumYIntrPnts                     1
		CreateLaunchMTO                         No 
		CovarianceSigmaScale                    3.000
		CovarianceMode                          None 
    END ClsApp
    
    BEGIN Units
		DistanceUnit		Kilometers
		TimeUnit		Seconds
		DateFormat		GregorianUTC
		AngleUnit		Degrees
		MassUnit		Kilograms
		PowerUnit		dBW
		FrequencyUnit		Gigahertz
		SmallDistanceUnit		Meters
		LatitudeUnit		Degrees
		LongitudeUnit		Degrees
		DurationUnit		Hr:Min:Sec
		Temperature		Kelvin
		SmallTimeUnit		Seconds
		RatioUnit		Decibel
		RcsUnit		Decibel
		DopplerVelocityUnit		MetersperSecond
		SARTimeResProdUnit		Meter-Second
		ForceUnit		Newtons
		PressureUnit		Pascals
		SpecificImpulseUnit		Seconds
		PRFUnit		Kilohertz
		BandwidthUnit		Megahertz
		SmallVelocityUnit		CentimetersperSecond
		Percent		Percentage
		AviatorDistanceUnit		NauticalMiles
		AviatorTimeUnit		Hours
		AviatorAltitudeUnit		Feet
		AviatorFuelQuantityUnit		Pounds
		AviatorRunwayLengthUnit		Kilofeet
		AviatorBearingAngleUnit		Degrees
		AviatorAngleOfAttackUnit		Degrees
		AviatorAttitudeAngleUnit		Degrees
		AviatorGUnit		StandardSeaLevelG
		SolidAngle		Steradians
		AviatorTSFCUnit		TSFCLbmHrLbf
		AviatorPSFCUnit		PSFCLbmHrHp
		AviatorForceUnit		Pounds
		AviatorPowerUnit		Horsepower
		SpectralBandwidthUnit		Hertz
		AviatorAltTimeUnit		Minutes
		AviatorSmallTimeUnit		Seconds
		AviatorEnergyUnit		kilowatt-hours
		BitsUnit		MegaBits
		RadiationDose		Rads
		MagneticFieldUnit		nanoTesla
		RadiationShieldThickness		Mils
		ParticleEnergy		MeV
    END Units
    
    BEGIN ReportUnits
		DistanceUnit		Kilometers
		TimeUnit		Seconds
		DateFormat		GregorianUTC
		AngleUnit		Degrees
		MassUnit		Kilograms
		PowerUnit		dBW
		FrequencyUnit		Gigahertz
		SmallDistanceUnit		Meters
		LatitudeUnit		Degrees
		LongitudeUnit		Degrees
		DurationUnit		Hr:Min:Sec
		Temperature		Kelvin
		SmallTimeUnit		Seconds
		RatioUnit		Decibel
		RcsUnit		Decibel
		DopplerVelocityUnit		MetersperSecond
		SARTimeResProdUnit		Meter-Second
		ForceUnit		Newtons
		PressureUnit		Pascals
		SpecificImpulseUnit		Seconds
		PRFUnit		Kilohertz
		BandwidthUnit		Megahertz
		SmallVelocityUnit		CentimetersperSecond
		Percent		Percentage
		AviatorDistanceUnit		NauticalMiles
		AviatorTimeUnit		Hours
		AviatorAltitudeUnit		Feet
		AviatorFuelQuantityUnit		Pounds
		AviatorRunwayLengthUnit		Kilofeet
		AviatorBearingAngleUnit		Degrees
		AviatorAngleOfAttackUnit		Degrees
		AviatorAttitudeAngleUnit		Degrees
		AviatorGUnit		StandardSeaLevelG
		SolidAngle		Steradians
		AviatorTSFCUnit		TSFCLbmHrLbf
		AviatorPSFCUnit		PSFCLbmHrHp
		AviatorForceUnit		Pounds
		AviatorPowerUnit		Horsepower
		SpectralBandwidthUnit		Hertz
		AviatorAltTimeUnit		Minutes
		AviatorSmallTimeUnit		Seconds
		AviatorEnergyUnit		kilowatt-hours
		BitsUnit		MegaBits
		RadiationDose		Rads
		MagneticFieldUnit		nanoTesla
		RadiationShieldThickness		Mils
		ParticleEnergy		MeV
    END ReportUnits
    
    BEGIN ConnectReportUnits
		DistanceUnit		Kilometers
		TimeUnit		Seconds
		DateFormat		GregorianUTC
		AngleUnit		Degrees
		MassUnit		Kilograms
		PowerUnit		dBW
		FrequencyUnit		Gigahertz
		SmallDistanceUnit		Meters
		LatitudeUnit		Degrees
		LongitudeUnit		Degrees
		DurationUnit		Hr:Min:Sec
		Temperature		Kelvin
		SmallTimeUnit		Seconds
		RatioUnit		Decibel
		RcsUnit		Decibel
		DopplerVelocityUnit		MetersperSecond
		SARTimeResProdUnit		Meter-Second
		ForceUnit		Newtons
		PressureUnit		Pascals
		SpecificImpulseUnit		Seconds
		PRFUnit		Kilohertz
		BandwidthUnit		Megahertz
		SmallVelocityUnit		CentimetersperSecond
		Percent		Percentage
		AviatorDistanceUnit		NauticalMiles
		AviatorTimeUnit		Hours
		AviatorAltitudeUnit		Feet
		AviatorFuelQuantityUnit		Pounds
		AviatorRunwayLengthUnit		Kilofeet
		AviatorBearingAngleUnit		Degrees
		AviatorAngleOfAttackUnit		Degrees
		AviatorAttitudeAngleUnit		Degrees
		AviatorGUnit		StandardSeaLevelG
		SolidAngle		Steradians
		AviatorTSFCUnit		TSFCLbmHrLbf
		AviatorPSFCUnit		PSFCLbmHrHp
		AviatorForceUnit		Pounds
		AviatorPowerUnit		Horsepower
		SpectralBandwidthUnit		Hertz
		AviatorAltTimeUnit		Minutes
		AviatorSmallTimeUnit		Seconds
		AviatorEnergyUnit		kilowatt-hours
		BitsUnit		MegaBits
		RadiationDose		Rads
		MagneticFieldUnit		nanoTesla
		RadiationShieldThickness		Mils
		ParticleEnergy		MeV
    END ConnectReportUnits
    
    BEGIN ReportFavorites
        BEGIN Class
            Name  Access
            BEGIN Favorite
                Type    Graph
                BaseDir Install
                Style   Carrier_to_Noise_Ratio
            END Favorite
            BEGIN Favorite
                Type    Graph
                BaseDir Install
                Style   Bit_Error_Rate
            END Favorite
            BEGIN Favorite
                Type    Graph
                BaseDir Install
                Style   EbNo
            END Favorite
            BEGIN Favorite
                Type    Graph
                BaseDir User
                Style   Elevation_Time
            END Favorite
            BEGIN Favorite
                Type    Graph
                BaseDir Install
                Style   AER
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   AER
            END Favorite
        END Class
        BEGIN Class
            Name  FigureOfMerit
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Grid Stats
            END Favorite
            BEGIN Favorite
                Type    Graph
                BaseDir Install
                Style   Grid Stats Over Time
            END Favorite
            BEGIN Favorite
                Type    Graph
                BaseDir Install
                Style   GI Region FOM
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   GI Region FOM
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Satisfied By Time
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Grid Stats Over Time
            END Favorite
        END Class
        BEGIN Class
            Name  CoverageDefinition
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Coverage By Latitude
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   GI Region Pass Coverage
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Grid Point Information
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Percent Coverage
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Gaps in Global Coverage
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Time To Cover By Region
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Gap Duration
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Access Duration
            END Favorite
            BEGIN Favorite
                Type    Report
                BaseDir Install
                Style   Coverage By Asset
            END Favorite
            BEGIN Favorite
                Type    Graph
                BaseDir Install
                Style   Coverage By Latitude
            END Favorite
            BEGIN Favorite
                Type    Graph
                BaseDir Install
                Style   Access Duration
            END Favorite
            BEGIN Favorite
                Type    Graph
                BaseDir Install
                Style   Percent Coverage
            END Favorite
        END Class
    END ReportFavorites
    
    BEGIN ADFFileData
    END ADFFileData
    
    BEGIN GenDb

		BEGIN Database
		    DbType       Satellite
		    DefDb        stkAllTLE.sd
		    UseMyDb      Off
		    MaxMatches   2000
		    Use4SOC      On

		BEGIN FieldDefaults

			BEGIN Field
				Name "SSC Number"
				Default "*"
			END Field

			BEGIN Field
				Name "Common Name"
				Default "*"
			END Field

		END FieldDefaults

		END Database

		BEGIN Database
		    DbType       City
		    DefDb        stkCityDb.cd
		    UseMyDb      Off
		    MaxMatches   2000
		    Use4SOC      On

		BEGIN FieldDefaults

			BEGIN Field
				Name "City Name"
				Default "*"
			END Field

		END FieldDefaults

		END Database

		BEGIN Database
		    DbType       Facility
		    DefDb        stkFacility.fd
		    UseMyDb      Off
		    MaxMatches   2000
		    Use4SOC      On

		BEGIN FieldDefaults

		END FieldDefaults

		END Database
    END GenDb
    
    BEGIN SOCDb
        BEGIN Defaults
        END Defaults
    END SOCDb
    
    BEGIN Msgp4Ext
    END Msgp4Ext
    
    BEGIN FileLocations
    END FileLocations
    
    BEGIN Author
	Optimize	No
	UseBasicGlobe	No
	SaveEphemeris	Yes
	SaveScenFolder	No
	BEGIN ExternalFileTypes
	    BEGIN Type
		FileType  Calculation Scalar
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  Celestial Image
		Include    No
	    END Type
	    BEGIN Type
		FileType  Cloud
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  EOP
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  External Vector Data
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  Globe
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  Globe Data
		Include    No
	    END Type
	    BEGIN Type
		FileType  Map
		Include    No
	    END Type
	    BEGIN Type
		FileType  Map Image
		Include    No
	    END Type
	    BEGIN Type
		FileType  Marker/Label
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  Model
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  Object Break-up File
		Include    No
	    END Type
	    BEGIN Type
		FileType  Planetary Ephemeris
		Include    No
	    END Type
	    BEGIN Type
		FileType  Report Style Script
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  Report/Graph Style
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  Scalar Calculation File
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  Terrain
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  Volume Grid Intervals File
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  Volumetric File
		Include    Yes
	    END Type
	    BEGIN Type
		FileType  WTM
		Include    Yes
	    END Type
	END ExternalFileTypes
	ReadOnly	No
	ViewerPassword	No
	STKPassword	No
	ExcludeInstallFiles	No
	BEGIN ExternalFileList
	END ExternalFileList
    END Author
    
    BEGIN ExportDataFile
    FileType         Ephemeris
    IntervalType     Ephemeris
    TimePeriodStart  0.000000e+00
    TimePeriodStop   0.000000e+00
    StepType         Ephemeris
    StepSize         60.000000
    EphemType        STK
    UseVehicleCentralBody   Yes
    CentralBody      Earth
    SatelliteID      -200000
    CoordSys         ICRF
    NonSatCoordSys   Fixed
    InterpolateBoundaries  Yes
    EphemFormat      Current
    InterpType       9
    InterpOrder      5
    AttCoordSys      Fixed
    Quaternions      0
    ExportCovar      Position
    AttitudeFormat   Current
    TimePrecision      6
    CCSDSDateFormat    YMD
    CCSDSEphFormat     SciNotation
    CCSDSTimeSystem    UTC
    CCSDSRefFrame      ICRF
    UseSatCenterAndFrame   No
    IncludeCovariance      No
    IncludeAcceleration    No
    CCSDSFileFormat      KVN
    END ExportDataFile
    
    BEGIN Desc
    END Desc
    
    BEGIN RfEnv
<?xml version = "1.0" standalone = "yes"?>
<VAR name = "STK_RF_Environment">
    <SCOPE Class = "RFEnvironment">
        <VAR name = "Version">
            <STRING>&quot;1.0.0 a&quot;</STRING>
        </VAR>
        <VAR name = "ComponentName">
            <STRING>&quot;STK_RF_Environment&quot;</STRING>
        </VAR>
        <VAR name = "Description">
            <STRING>&quot;STK RF Environment&quot;</STRING>
        </VAR>
        <VAR name = "Type">
            <STRING>&quot;STK RF Environment&quot;</STRING>
        </VAR>
        <VAR name = "UserComment">
            <STRING>&quot;STK RF Environment&quot;</STRING>
        </VAR>
        <VAR name = "ReadOnly">
            <BOOL>false</BOOL>
        </VAR>
        <VAR name = "Clonable">
            <BOOL>true</BOOL>
        </VAR>
        <VAR name = "Category">
            <STRING>&quot;&quot;</STRING>
        </VAR>
        <VAR name = "PropagationChannel">
            <VAR name = "RF_Propagation_Channel">
                <SCOPE Class = "PropagationChannel">
                    <VAR name = "Version">
                        <STRING>&quot;1.0.0 a&quot;</STRING>
                    </VAR>
                    <VAR name = "ComponentName">
                        <STRING>&quot;RF_Propagation_Channel&quot;</STRING>
                    </VAR>
                    <VAR name = "Description">
                        <STRING>&quot;RF Propagation Channel&quot;</STRING>
                    </VAR>
                    <VAR name = "Type">
                        <STRING>&quot;RF Propagation Channel&quot;</STRING>
                    </VAR>
                    <VAR name = "UserComment">
                        <STRING>&quot;RF Propagation Channel&quot;</STRING>
                    </VAR>
                    <VAR name = "ReadOnly">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "Clonable">
                        <BOOL>true</BOOL>
                    </VAR>
                    <VAR name = "Category">
                        <STRING>&quot;&quot;</STRING>
                    </VAR>
                    <VAR name = "UseITU618Section2p5">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "UseCloudFogModel">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "CloudFogModel">
                        <VAR name = "ITU-R_P840-6">
                            <SCOPE Class = "CloudFogLossModel">
                                <VAR name = "Version">
                                    <STRING>&quot;1.0.0 a&quot;</STRING>
                                </VAR>
                                <VAR name = "ComponentName">
                                    <STRING>&quot;ITU-R_P840-6&quot;</STRING>
                                </VAR>
                                <VAR name = "Description">
                                    <STRING>&quot;ITU-R P840-6&quot;</STRING>
                                </VAR>
                                <VAR name = "Type">
                                    <STRING>&quot;ITU-R P840-6&quot;</STRING>
                                </VAR>
                                <VAR name = "UserComment">
                                    <STRING>&quot;ITU-R P840-6&quot;</STRING>
                                </VAR>
                                <VAR name = "ReadOnly">
                                    <BOOL>false</BOOL>
                                </VAR>
                                <VAR name = "Clonable">
                                    <BOOL>true</BOOL>
                                </VAR>
                                <VAR name = "Category">
                                    <STRING>&quot;&quot;</STRING>
                                </VAR>
                                <VAR name = "LiquidWaterDensityValueChoice">
                                    <STRING>&quot;Liquid Water Content Density Value&quot;</STRING>
                                </VAR>
                                <VAR name = "CloudCeiling">
                                    <QUANTITY Dimension = "DistanceUnit" Unit = "m">
                                        <REAL>3000</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "CloudLayerThickness">
                                    <QUANTITY Dimension = "DistanceUnit" Unit = "m">
                                        <REAL>500</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "CloudTemp">
                                    <QUANTITY Dimension = "Temperature" Unit = "K">
                                        <REAL>273.15</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "CloudLiqWaterDensity">
                                    <QUANTITY Dimension = "SmallDensity" Unit = "kg*m^-3">
                                        <REAL>0.0075</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "AnnualAveragePercentValue">
                                    <QUANTITY Dimension = "Percent" Unit = "unitValue">
                                        <REAL>0.01</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "MonthlyAveragePercentValue">
                                    <QUANTITY Dimension = "Percent" Unit = "unitValue">
                                        <REAL>0.01</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "LiqWaterAverageDataMonth">
                                    <INT>1</INT>
                                </VAR>
                            </SCOPE>
                        </VAR>
                    </VAR>
                    <VAR name = "UseTropoScintModel">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "TropoScintModel">
                        <VAR name = "ITU-R_P618-12">
                            <SCOPE Class = "TropoScintLossModel">
                                <VAR name = "Version">
                                    <STRING>&quot;1.0.0 a&quot;</STRING>
                                </VAR>
                                <VAR name = "ComponentName">
                                    <STRING>&quot;ITU-R_P618-12&quot;</STRING>
                                </VAR>
                                <VAR name = "Description">
                                    <STRING>&quot;ITU-R P618-12&quot;</STRING>
                                </VAR>
                                <VAR name = "Type">
                                    <STRING>&quot;ITU-R P618-12&quot;</STRING>
                                </VAR>
                                <VAR name = "UserComment">
                                    <STRING>&quot;ITU-R P618-12&quot;</STRING>
                                </VAR>
                                <VAR name = "ReadOnly">
                                    <BOOL>false</BOOL>
                                </VAR>
                                <VAR name = "Clonable">
                                    <BOOL>true</BOOL>
                                </VAR>
                                <VAR name = "Category">
                                    <STRING>&quot;&quot;</STRING>
                                </VAR>
                                <VAR name = "FadeDepthAverageTimeChoice">
                                    <STRING>&quot;Fade depth for the average year&quot;</STRING>
                                </VAR>
                                <VAR name = "ComputeDeepFade">
                                    <BOOL>false</BOOL>
                                </VAR>
                                <VAR name = "FadeOutage">
                                    <QUANTITY Dimension = "Percent" Unit = "unitValue">
                                        <REAL>0.001</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "PercentTimeRefracGrad">
                                    <QUANTITY Dimension = "Percent" Unit = "unitValue">
                                        <REAL>0.1</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "SurfaceTemperature">
                                    <QUANTITY Dimension = "Temperature" Unit = "K">
                                        <REAL>273.15</REAL>
                                    </QUANTITY>
                                </VAR>
                            </SCOPE>
                        </VAR>
                    </VAR>
                    <VAR name = "UseRainModel">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "RainModel">
                        <VAR name = "ITU-R_P618-12">
                            <SCOPE Class = "RainLossModel">
                                <VAR name = "Version">
                                    <STRING>&quot;1.0.0 a&quot;</STRING>
                                </VAR>
                                <VAR name = "ComponentName">
                                    <STRING>&quot;ITU-R_P618-12&quot;</STRING>
                                </VAR>
                                <VAR name = "Description">
                                    <STRING>&quot;ITU-R P618-12 rain model&quot;</STRING>
                                </VAR>
                                <VAR name = "Type">
                                    <STRING>&quot;ITU-R P618-12&quot;</STRING>
                                </VAR>
                                <VAR name = "UserComment">
                                    <STRING>&quot;ITU-R P618-12 rain model&quot;</STRING>
                                </VAR>
                                <VAR name = "ReadOnly">
                                    <BOOL>false</BOOL>
                                </VAR>
                                <VAR name = "Clonable">
                                    <BOOL>true</BOOL>
                                </VAR>
                                <VAR name = "Category">
                                    <STRING>&quot;&quot;</STRING>
                                </VAR>
                                <VAR name = "SurfaceTemperature">
                                    <QUANTITY Dimension = "Temperature" Unit = "K">
                                        <REAL>273.15</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "EnableDepolarizationLoss">
                                    <BOOL>false</BOOL>
                                </VAR>
                            </SCOPE>
                        </VAR>
                    </VAR>
                    <VAR name = "UseAtmosAbsorptionModel">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "AtmosAbsorptionModel">
                        <VAR name = "Simple_Satcom">
                            <SCOPE Class = "AtmosphericAbsorptionModel">
                                <VAR name = "Version">
                                    <STRING>&quot;1.0.1 a&quot;</STRING>
                                </VAR>
                                <VAR name = "ComponentName">
                                    <STRING>&quot;Simple_Satcom&quot;</STRING>
                                </VAR>
                                <VAR name = "Description">
                                    <STRING>&quot;Simple Satcom gaseous absorption model&quot;</STRING>
                                </VAR>
                                <VAR name = "Type">
                                    <STRING>&quot;Simple Satcom&quot;</STRING>
                                </VAR>
                                <VAR name = "UserComment">
                                    <STRING>&quot;Simple Satcom gaseous absorption model&quot;</STRING>
                                </VAR>
                                <VAR name = "ReadOnly">
                                    <BOOL>false</BOOL>
                                </VAR>
                                <VAR name = "Clonable">
                                    <BOOL>true</BOOL>
                                </VAR>
                                <VAR name = "Category">
                                    <STRING>&quot;&quot;</STRING>
                                </VAR>
                                <VAR name = "SurfaceTemperature">
                                    <QUANTITY Dimension = "Temperature" Unit = "K">
                                        <REAL>293.15</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "WaterVaporConcentration">
                                    <QUANTITY Dimension = "Density" Unit = "g*m^-3">
                                        <REAL>7.5</REAL>
                                    </QUANTITY>
                                </VAR>
                            </SCOPE>
                        </VAR>
                    </VAR>
                    <VAR name = "UseUrbanTerresPropLossModel">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "UrbanTerresPropLossModel">
                        <VAR name = "Two_Ray">
                            <SCOPE Class = "UrbanTerrestrialPropagationLossModel">
                                <VAR name = "Version">
                                    <STRING>&quot;1.0.0 a&quot;</STRING>
                                </VAR>
                                <VAR name = "ComponentName">
                                    <STRING>&quot;Two_Ray&quot;</STRING>
                                </VAR>
                                <VAR name = "Description">
                                    <STRING>&quot;Two Ray (Fourth Power Law) atmospheric absorption model&quot;</STRING>
                                </VAR>
                                <VAR name = "Type">
                                    <STRING>&quot;Two Ray&quot;</STRING>
                                </VAR>
                                <VAR name = "UserComment">
                                    <STRING>&quot;Two Ray (Fourth Power Law) atmospheric absorption model&quot;</STRING>
                                </VAR>
                                <VAR name = "ReadOnly">
                                    <BOOL>false</BOOL>
                                </VAR>
                                <VAR name = "Clonable">
                                    <BOOL>true</BOOL>
                                </VAR>
                                <VAR name = "Category">
                                    <STRING>&quot;&quot;</STRING>
                                </VAR>
                                <VAR name = "SurfaceTemperature">
                                    <QUANTITY Dimension = "Temperature" Unit = "K">
                                        <REAL>273.15</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "LossFactor">
                                    <REAL>1</REAL>
                                </VAR>
                            </SCOPE>
                        </VAR>
                    </VAR>
                    <VAR name = "UseCustomA">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "UseCustomB">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "UseCustomC">
                        <BOOL>false</BOOL>
                    </VAR>
                </SCOPE>
            </VAR>
        </VAR>
        <VAR name = "EarthTemperature">
            <QUANTITY Dimension = "Temperature" Unit = "K">
                <REAL>290</REAL>
            </QUANTITY>
        </VAR>
        <VAR name = "RainOutagePercent">
            <PROP name = "FormatString">
                <STRING>&quot;%#6.3f&quot;</STRING>
            </PROP>
            <REAL>0.1</REAL>
        </VAR>
        <VAR name = "ActiveCommSystem">
            <LINKTOOBJ>
                <STRING>&quot;None&quot;</STRING>
            </LINKTOOBJ>
        </VAR>
    </SCOPE>
</VAR>    END RfEnv
    
    BEGIN CommRad
    END CommRad
    
    BEGIN RadarCrossSection
<?xml version = "1.0" standalone = "yes"?>
<VAR name = "STK_Radar_RCS_Extension">
    <SCOPE Class = "RadarRCSExtension">
        <VAR name = "Version">
            <STRING>&quot;1.0.0 a&quot;</STRING>
        </VAR>
        <VAR name = "ComponentName">
            <STRING>&quot;STK_Radar_RCS_Extension&quot;</STRING>
        </VAR>
        <VAR name = "Description">
            <STRING>&quot;STK Radar RCS Extension&quot;</STRING>
        </VAR>
        <VAR name = "Type">
            <STRING>&quot;STK Radar RCS Extension&quot;</STRING>
        </VAR>
        <VAR name = "UserComment">
            <STRING>&quot;STK Radar RCS Extension&quot;</STRING>
        </VAR>
        <VAR name = "ReadOnly">
            <BOOL>false</BOOL>
        </VAR>
        <VAR name = "Clonable">
            <BOOL>true</BOOL>
        </VAR>
        <VAR name = "Category">
            <STRING>&quot;&quot;</STRING>
        </VAR>
        <VAR name = "Model">
            <VAR name = "Radar_Cross_Section">
                <SCOPE Class = "RCS">
                    <VAR name = "Version">
                        <STRING>&quot;1.0.0 a&quot;</STRING>
                    </VAR>
                    <VAR name = "ComponentName">
                        <STRING>&quot;Radar_Cross_Section&quot;</STRING>
                    </VAR>
                    <VAR name = "Description">
                        <STRING>&quot;Radar Cross Section&quot;</STRING>
                    </VAR>
                    <VAR name = "Type">
                        <STRING>&quot;Radar Cross Section&quot;</STRING>
                    </VAR>
                    <VAR name = "UserComment">
                        <STRING>&quot;Radar Cross Section&quot;</STRING>
                    </VAR>
                    <VAR name = "ReadOnly">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "Clonable">
                        <BOOL>true</BOOL>
                    </VAR>
                    <VAR name = "Category">
                        <STRING>&quot;&quot;</STRING>
                    </VAR>
                    <VAR name = "FrequencyBandList">
                        <LIST>
                            <SCOPE>
                                <VAR name = "MinFrequency">
                                    <QUANTITY Dimension = "BandwidthUnit" Unit = "Hz">
                                        <REAL>2997920</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "ComputeTypeStrategy">
                                    <VAR name = "Constant Value">
                                        <SCOPE Class = "RCS Compute Strategy">
                                            <VAR name = "ConstantValue">
                                                <QUANTITY Dimension = "RcsUnit" Unit = "sqm">
                                                    <REAL>1</REAL>
                                                </QUANTITY>
                                            </VAR>
                                            <VAR name = "Type">
                                                <STRING>&quot;Constant Value&quot;</STRING>
                                            </VAR>
                                            <VAR name = "ComponentName">
                                                <STRING>&quot;Constant Value&quot;</STRING>
                                            </VAR>
                                        </SCOPE>
                                    </VAR>
                                </VAR>
                                <VAR name = "SwerlingCase">
                                    <STRING>&quot;0&quot;</STRING>
                                </VAR>
                            </SCOPE>
                        </LIST>
                    </VAR>
                </SCOPE>
            </VAR>
        </VAR>
    </SCOPE>
</VAR>    END RadarCrossSection
    
    BEGIN RadarClutter
<?xml version = "1.0" standalone = "yes"?>
<VAR name = "STK_Radar_Clutter_Extension">
    <SCOPE Class = "RadarClutterExtension">
        <VAR name = "Version">
            <STRING>&quot;1.0.0 a&quot;</STRING>
        </VAR>
        <VAR name = "ComponentName">
            <STRING>&quot;STK_Radar_Clutter_Extension&quot;</STRING>
        </VAR>
        <VAR name = "Description">
            <STRING>&quot;STK Radar Clutter Extension&quot;</STRING>
        </VAR>
        <VAR name = "Type">
            <STRING>&quot;STK Radar Clutter Extension&quot;</STRING>
        </VAR>
        <VAR name = "UserComment">
            <STRING>&quot;STK Radar Clutter Extension&quot;</STRING>
        </VAR>
        <VAR name = "ReadOnly">
            <BOOL>false</BOOL>
        </VAR>
        <VAR name = "Clonable">
            <BOOL>true</BOOL>
        </VAR>
        <VAR name = "Category">
            <STRING>&quot;&quot;</STRING>
        </VAR>
        <VAR name = "ClutterMap">
            <VAR name = "Constant Coefficient">
                <SCOPE Class = "Clutter Map">
                    <VAR name = "ClutterCoefficient">
                        <QUANTITY Dimension = "RatioUnit" Unit = "units">
                            <REAL>1</REAL>
                        </QUANTITY>
                    </VAR>
                    <VAR name = "Type">
                        <STRING>&quot;Constant Coefficient&quot;</STRING>
                    </VAR>
                    <VAR name = "ComponentName">
                        <STRING>&quot;Constant Coefficient&quot;</STRING>
                    </VAR>
                </SCOPE>
            </VAR>
        </VAR>
    </SCOPE>
</VAR>    END RadarClutter
    
    BEGIN Gator
    END Gator
    
    BEGIN Crdn
    END Crdn
    
    BEGIN ScenSpaceEnvironment

        Begin RadiationEnvironment

           NasaModelsActivity      SolarMin
           CrresProActivity        Quiet
           CrresRadActivity        Average
           UseDefaultNasaEnergies  Yes

        End RadiationEnvironment

    END ScenSpaceEnvironment
    
    BEGIN SpiceExt
    END SpiceExt
    
    BEGIN FlightScenExt
    END FlightScenExt
    
    BEGIN Graphics

BEGIN Animation

    StartTime          25 Aug 2025 00:00:00.000000000
    EndTime            26 Aug 2025 00:00:00.000000000
    CurrentTime        25 Aug 2025 00:00:00.000000000
    Direction          Forward
    UpdateDelta        10.000000
    RefreshDelta       0.010000
    XRealTimeMult      1.000000
    RealTimeOffset     0.000000
    XRtStartFromPause  Yes

END Animation


        BEGIN DisplayFlags
            ShowLabels           On
            ShowPassLabel        Off
            ShowElsetNum         Off
            ShowGndTracks        On
            ShowGndMarkers       On
            ShowOrbitMarkers     On
            ShowPlanetOrbits     Off
            ShowPlanetCBIPos     On
            ShowPlanetCBILabel   On
            ShowPlanetGndPos     On
            ShowPlanetGndLabel   On
            ShowSensors          On
            ShowWayptMarkers     Off
            ShowWayptTurnMarkers Off
            ShowOrbits           On
            ShowDtedRegions      Off
            ShowAreaTgtCentroids On
            ShowToolBar          On
            ShowStatusBar        On
            ShowScrollBars       On
            AllowAnimUpdate      On
            AccShowLine          On
            AccAnimHigh          On
            AccStatHigh          On
            ShowPrintButton      On
            ShowAnimButtons      On
            ShowAnimModeButtons  On
            ShowZoomMsrButtons   On
            ShowMapCbButton      Off
        END DisplayFlags

BEGIN WinFonts

    System
    MS Sans Serif,22,0,0
    MS Sans Serif,28,0,0

END WinFonts

BEGIN MapData

    Begin TerrainConverterData
           NorthLat        0.00000000000000e+00
           EastLon         0.00000000000000e+00
           SouthLat        0.00000000000000e+00
           WestLon         0.00000000000000e+00
           ColorByRGB      No
           AltsFromMSL     No
           UseColorRamp    Yes
           UseRegionMinMax Yes
           SizeSameAsSrc   Yes
           MinAltHSV       0.00000000000000e+00 7.00000000000000e-01 8.00000000000000e-01 4.00000000000000e-01
           MaxAltHSV       1.00000000000000e+06 0.00000000000000e+00 2.00000000000000e-01 1.00000000000000e+00
           SmoothColors    Yes
           CreateChunkTrn  No
           OutputFormat    PDTTX
    End TerrainConverterData

    DisableDefKbdActions     Off
    TextShadowStyle          Dark
    TextShadowColor          #000000
    BingLevelOfDetailScale   2.000000
    BEGIN Map
        MapNum         1
        TrackingMode   LatLon
        PickEnabled    On
        PanEnabled     On

        BEGIN MapAttributes
            PrimaryBody          Earth
            SecondaryBody        Sun
            CenterLatitude       2.418903
            CenterLongitude      101.023160
            ProjectionAltitude   63621860.000000
            FieldOfView          35.000000
            OrthoDisplayDistance 20000000.000000
            TransformTrajectory  On
            EquatorialRadius     6378137.000000
            BackgroundColor      #000000
            LatLonLines          On
            LatSpacing           30.000000
            LonSpacing           30.000000
            LatLonLineColor      #999999
            LatLonLineStyle      2
            ShowOrthoDistGrid    Off
            OrthoGridXSpacing    5
            OrthoGridYSpacing    5
            OrthoGridColor       #ffffff
            ShowImageExtents     Off
            ImageExtentLineColor #ffffff
            ImageExtentLineStyle 0
            ImageExtentLineWidth 1.000000
            ShowImageNames       Off
            ImageNameFont        0
            Projection           EquidistantCylindrical
            Resolution           Low
            CoordinateSys        ECF
            UseBackgroundImage   On
            UseBingForBackground On
            BingType             Aerial
            BingLogoHorizAlign   Right
            BingLogoVertAlign    Bottom
            BackgroundImageFile  Basic.bmp
            UseNightLights       Off
            NightLightsFactor    3.500000
            UseCloudsFile        Off
            BEGIN ZoomLocations
                BEGIN ZoomLocation
                    CenterLat    12.353219
                    CenterLon    104.274177
                    ZoomWidth    43.767596
                    ZoomHeight   20.793434
                End ZoomLocation
                BEGIN ZoomLocation
                    CenterLat    11.690609
                    CenterLon    105.805162
                    ZoomWidth    55.788597
                    ZoomHeight   26.217312
                End ZoomLocation
                BEGIN ZoomLocation
                    CenterLat    2.418903
                    CenterLon    101.023160
                    ZoomWidth    101.674633
                    ZoomHeight   47.874334
                End ZoomLocation
            END ZoomLocations
            UseVarAspectRatio    No
            SwapMapResolution    Yes
            NoneToVLowSwapDist   2000000.000000
            VLowToLowSwapDist    20000.000000
            LowToMediumSwapDist  10000.000000
            MediumToHighSwapDist 5000.000000
            HighToVHighSwapDist  1000.000000
            VHighToSHighSwapDist 100.000000
            BEGIN Axes
                DisplayAxes no
                CoordSys    CBI
                2aryCB      Sun
                Display+x   yes
                Label+x     yes
                Color+x     #ffffff
                Scale+x     3.000000
                Display-x   yes
                Label-x     yes
                Color-x     #ffffff
                Scale-x     3.000000
                Display+y   yes
                Label+y     yes
                Color+y     #ffffff
                Scale+y     3.000000
                Display-y   yes
                Label-y     yes
                Color-y     #ffffff
                Scale-y     3.000000
                Display+z   yes
                Label+z     yes
                Color+z     #ffffff
                Scale+z     3.000000
                Display-z   yes
                Label-z     yes
                Color-z     #ffffff
                Scale-z     3.000000
            END Axes

        END MapAttributes

        BEGIN MapList
            BEGIN Detail
                Alias RWDB2_Coastlines
                Show Yes
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_International_Borders
                Show No
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_Islands
                Show No
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_Lakes
                Show No
                Color #87cefa
            END Detail
            BEGIN Detail
                Alias RWDB2_Provincial_Borders
                Show No
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_Rivers
                Show No
                Color #87cefa
            END Detail
        END MapList


        BEGIN MapAnnotations
        END MapAnnotations

        BEGIN DisplayFlags
            ShowLabels           On
            ShowPassLabel        Off
            ShowElsetNum         Off
            ShowGndTracks        On
            ShowGndMarkers       On
            ShowOrbitMarkers     On
            ShowPlanetOrbits     Off
            ShowPlanetCBIPos     On
            ShowPlanetCBILabel   On
            ShowPlanetGndPos     On
            ShowPlanetGndLabel   On
            ShowSensors          On
            ShowWayptMarkers     Off
            ShowWayptTurnMarkers Off
            ShowOrbits           On
            ShowDtedRegions      Off
            ShowAreaTgtCentroids On
            ShowToolBar          On
            ShowStatusBar        On
            ShowScrollBars       On
            AllowAnimUpdate      Off
            AccShowLine          On
            AccAnimHigh          On
            AccStatHigh          On
            ShowPrintButton      On
            ShowAnimButtons      On
            ShowAnimModeButtons  On
            ShowZoomMsrButtons   On
            ShowMapCbButton      Off
        END DisplayFlags

        BEGIN SoftVTR
            OutputFormat     WMV
            Directory        C:\Users\pakaw\OneDrive\Documents\STK 11 (x64)\WalkerDeltaAnalysis
            BaseName         Frame
            Digits           4
            Frame            1
            LastAnimTime     0.000000
            OutputMode       Normal
            HiResAssembly    Assemble
            HRWidth          6000
            HRHeight         4500
            HRDPI            600.000000
            UseSnapInterval  No
            SnapInterval     0.000000
            WmvCodec         "Windows Media Video 9"
            Framerate        30
            Bitrate          3000000
        END SoftVTR


        BEGIN TimeDisplay
            Show             0
            TextColor        #ffffff
            TextTranslucency 0.000000
            ShowBackground   0
            BackColor        #4d4d4d
            BackTranslucency 0.400000
            XPosition        20
            YPosition        -20
        END TimeDisplay

        BEGIN LightingData
            DisplayAltitude              0.000000
            SubsolarPoint                Off
            SubsolarPointColor           #ffff00
            SubsolarPointMarkerStyle     2

            ShowUmbraLine                Off
            UmbraLineColor               #000000
            UmbraLineStyle               0
            UmbraLineWidth               2
            FillUmbra                    On
            UmbraFillColor               #000000
            ShowSunlightLine             Off
            SunlightLineColor            #ffff00
            SunlightLineStyle            0
            SunlightLineWidth            2
            FillSunlight                 On
            SunlightFillColor            #ffffff
            SunlightMinOpacity           0.000000
            SunlightMaxOpacity           0.200000
            UmbraMaxOpacity              0.700000
            UmbraMinOpacity              0.400000
        END LightingData
    END Map

    BEGIN MapStyles

        UseStyleTime        No

        BEGIN Style
        Name                DefaultWithBing
        Time                10800.000000
        UpdateDelta         10.000000

        BEGIN MapAttributes
            PrimaryBody          Earth
            SecondaryBody        Sun
            CenterLatitude       0.000000
            CenterLongitude      0.000000
            ProjectionAltitude   63621860.000000
            FieldOfView          35.000000
            OrthoDisplayDistance 20000000.000000
            TransformTrajectory  On
            EquatorialRadius     6378137.000000
            BackgroundColor      #000000
            LatLonLines          On
            LatSpacing           30.000000
            LonSpacing           30.000000
            LatLonLineColor      #999999
            LatLonLineStyle      2
            ShowOrthoDistGrid    Off
            OrthoGridXSpacing    5
            OrthoGridYSpacing    5
            OrthoGridColor       #ffffff
            ShowImageExtents     Off
            ImageExtentLineColor #ffffff
            ImageExtentLineStyle 0
            ImageExtentLineWidth 1.000000
            ShowImageNames       Off
            ImageNameFont        0
            Projection           EquidistantCylindrical
            Resolution           VeryLow
            CoordinateSys        ECF
            UseBackgroundImage   On
            UseBingForBackground On
            BingType             Aerial
            BingLogoHorizAlign   Right
            BingLogoVertAlign    Bottom
            BackgroundImageFile  Basic.bmp
            UseNightLights       Off
            NightLightsFactor    3.500000
            UseCloudsFile        Off
            BEGIN ZoomLocations
                BEGIN ZoomLocation
                    CenterLat    0.000000
                    CenterLon    0.000000
                    ZoomWidth    359.999998
                    ZoomHeight   180.000000
                End ZoomLocation
            END ZoomLocations
            UseVarAspectRatio    No
            SwapMapResolution    Yes
            NoneToVLowSwapDist   2000000.000000
            VLowToLowSwapDist    20000.000000
            LowToMediumSwapDist  10000.000000
            MediumToHighSwapDist 5000.000000
            HighToVHighSwapDist  1000.000000
            VHighToSHighSwapDist 100.000000
            BEGIN Axes
                DisplayAxes no
                CoordSys    CBI
                2aryCB      Sun
                Display+x   yes
                Label+x     yes
                Color+x     #ffffff
                Scale+x     3.000000
                Display-x   yes
                Label-x     yes
                Color-x     #ffffff
                Scale-x     3.000000
                Display+y   yes
                Label+y     yes
                Color+y     #ffffff
                Scale+y     3.000000
                Display-y   yes
                Label-y     yes
                Color-y     #ffffff
                Scale-y     3.000000
                Display+z   yes
                Label+z     yes
                Color+z     #ffffff
                Scale+z     3.000000
                Display-z   yes
                Label-z     yes
                Color-z     #ffffff
                Scale-z     3.000000
            END Axes

        END MapAttributes

        BEGIN MapList
            BEGIN Detail
                Alias RWDB2_Coastlines
                Show Yes
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_International_Borders
                Show No
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_Islands
                Show No
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_Lakes
                Show No
                Color #87cefa
            END Detail
            BEGIN Detail
                Alias RWDB2_Provincial_Borders
                Show No
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_Rivers
                Show No
                Color #87cefa
            END Detail
        END MapList


        BEGIN MapAnnotations
        END MapAnnotations

        BEGIN SoftVTR
            OutputFormat     WMV
            Directory        C:\Users\pakaw\OneDrive\Documents\STK 11 (x64)\WalkerDeltaAnalysis
            BaseName         Frame
            Digits           4
            Frame            0
            LastAnimTime     0.000000
            OutputMode       Normal
            HiResAssembly    Assemble
            HRWidth          6000
            HRHeight         4500
            HRDPI            600.000000
            UseSnapInterval  No
            SnapInterval     0.000000
            WmvCodec         "Windows Media Video 9"
            Framerate        30
            Bitrate          3000000
        END SoftVTR


        BEGIN TimeDisplay
            Show             0
            TextColor        #ffffff
            TextTranslucency 0.000000
            ShowBackground   0
            BackColor        #4d4d4d
            BackTranslucency 0.400000
            XPosition        20
            YPosition        -20
        END TimeDisplay

        BEGIN LightingData
            DisplayAltitude              0.000000
            SubsolarPoint                Off
            SubsolarPointColor           #ffff00
            SubsolarPointMarkerStyle     2

            ShowUmbraLine                Off
            UmbraLineColor               #000000
            UmbraLineStyle               0
            UmbraLineWidth               2
            FillUmbra                    On
            UmbraFillColor               #000000
            ShowSunlightLine             Off
            SunlightLineColor            #ffff00
            SunlightLineStyle            0
            SunlightLineWidth            2
            FillSunlight                 On
            SunlightFillColor            #ffffff
            SunlightMinOpacity           0.000000
            SunlightMaxOpacity           0.200000
            UmbraMaxOpacity              0.700000
            UmbraMinOpacity              0.400000
        END LightingData

        ShowDtedRegions     Off

        End Style

        BEGIN Style
        Name                DefaultWithoutBing
        Time                10800.000000
        UpdateDelta         10.000000

        BEGIN MapAttributes
            PrimaryBody          Earth
            SecondaryBody        Sun
            CenterLatitude       0.000000
            CenterLongitude      0.000000
            ProjectionAltitude   63621860.000000
            FieldOfView          35.000000
            OrthoDisplayDistance 20000000.000000
            TransformTrajectory  On
            EquatorialRadius     6378137.000000
            BackgroundColor      #000000
            LatLonLines          On
            LatSpacing           30.000000
            LonSpacing           30.000000
            LatLonLineColor      #999999
            LatLonLineStyle      2
            ShowOrthoDistGrid    Off
            OrthoGridXSpacing    5
            OrthoGridYSpacing    5
            OrthoGridColor       #ffffff
            ShowImageExtents     Off
            ImageExtentLineColor #ffffff
            ImageExtentLineStyle 0
            ImageExtentLineWidth 1.000000
            ShowImageNames       Off
            ImageNameFont        0
            Projection           EquidistantCylindrical
            Resolution           VeryLow
            CoordinateSys        ECF
            UseBackgroundImage   On
            UseBingForBackground Off
            BingType             Aerial
            BingLogoHorizAlign   Right
            BingLogoVertAlign    Bottom
            BackgroundImageFile  Basic.bmp
            UseNightLights       Off
            NightLightsFactor    3.500000
            UseCloudsFile        Off
            BEGIN ZoomLocations
                BEGIN ZoomLocation
                    CenterLat    0.000000
                    CenterLon    0.000000
                    ZoomWidth    359.999998
                    ZoomHeight   180.000000
                End ZoomLocation
            END ZoomLocations
            UseVarAspectRatio    No
            SwapMapResolution    Yes
            NoneToVLowSwapDist   2000000.000000
            VLowToLowSwapDist    20000.000000
            LowToMediumSwapDist  10000.000000
            MediumToHighSwapDist 5000.000000
            HighToVHighSwapDist  1000.000000
            VHighToSHighSwapDist 100.000000
            BEGIN Axes
                DisplayAxes no
                CoordSys    CBI
                2aryCB      Sun
                Display+x   yes
                Label+x     yes
                Color+x     #ffffff
                Scale+x     3.000000
                Display-x   yes
                Label-x     yes
                Color-x     #ffffff
                Scale-x     3.000000
                Display+y   yes
                Label+y     yes
                Color+y     #ffffff
                Scale+y     3.000000
                Display-y   yes
                Label-y     yes
                Color-y     #ffffff
                Scale-y     3.000000
                Display+z   yes
                Label+z     yes
                Color+z     #ffffff
                Scale+z     3.000000
                Display-z   yes
                Label-z     yes
                Color-z     #ffffff
                Scale-z     3.000000
            END Axes

        END MapAttributes

        BEGIN MapList
            BEGIN Detail
                Alias RWDB2_Coastlines
                Show Yes
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_International_Borders
                Show No
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_Islands
                Show No
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_Lakes
                Show No
                Color #87cefa
            END Detail
            BEGIN Detail
                Alias RWDB2_Provincial_Borders
                Show No
                Color #8fbc8f
            END Detail
            BEGIN Detail
                Alias RWDB2_Rivers
                Show No
                Color #87cefa
            END Detail
        END MapList


        BEGIN MapAnnotations
        END MapAnnotations

        BEGIN SoftVTR
            OutputFormat     WMV
            Directory        C:\Users\pakaw\OneDrive\Documents\STK 11 (x64)\WalkerDeltaAnalysis
            BaseName         Frame
            Digits           4
            Frame            0
            LastAnimTime     0.000000
            OutputMode       Normal
            HiResAssembly    Assemble
            HRWidth          6000
            HRHeight         4500
            HRDPI            600.000000
            UseSnapInterval  No
            SnapInterval     0.000000
            WmvCodec         "Windows Media Video 9"
            Framerate        30
            Bitrate          3000000
        END SoftVTR


        BEGIN TimeDisplay
            Show             0
            TextColor        #ffffff
            TextTranslucency 0.000000
            ShowBackground   0
            BackColor        #4d4d4d
            BackTranslucency 0.400000
            XPosition        20
            YPosition        -20
        END TimeDisplay

        BEGIN LightingData
            DisplayAltitude              0.000000
            SubsolarPoint                Off
            SubsolarPointColor           #ffff00
            SubsolarPointMarkerStyle     2

            ShowUmbraLine                Off
            UmbraLineColor               #000000
            UmbraLineStyle               0
            UmbraLineWidth               2
            FillUmbra                    On
            UmbraFillColor               #000000
            ShowSunlightLine             Off
            SunlightLineColor            #ffff00
            SunlightLineStyle            0
            SunlightLineWidth            2
            FillSunlight                 On
            SunlightFillColor            #ffffff
            SunlightMinOpacity           0.000000
            SunlightMaxOpacity           0.200000
            UmbraMaxOpacity              0.700000
            UmbraMinOpacity              0.400000
        END LightingData

        ShowDtedRegions     Off

        End Style

    END MapStyles

END MapData

        BEGIN GfxClassPref

        END GfxClassPref


        BEGIN ConnectGraphicsOptions

            AsyncPickReturnUnique          OFF

        END ConnectGraphicsOptions

    END Graphics
    
    BEGIN Overlays
    END Overlays
    
    BEGIN VO
    END VO
    
    BEGIN ScenSpaceEnvironmentGfx

        Begin Gfx

           Begin MagFieldGfx
               Show               No
               ColorBy            Magnitude
               ColorScale         Log
               ColorRampStart     #0000ff
               ColorRampStart     #0000ff
               ColorRampStop      #ff0000
               MaxTranslucency    0.700000
               LineStyle          0
               LineWidth          2.000000
               FieldLineRefresh   300.000000
               NumLats            8
               NumLongs           6
               StartLat           15.000000
               StopLat            85.000000
               RefLongitude       3.141593
               MainField          IGRF
               ExternalField      None
               IGRF_UpdateRate    86400.000000
           End MagFieldGfx

        End Gfx

    END ScenSpaceEnvironmentGfx
    
    BEGIN DIS

		Begin General

			Verbose                    Off
			Processing                 Off
			Statistics                 Off
			ExerciseID                 -1
			ForceID                    -1

		End General


		Begin Output

			Version                    5
			ExerciseID                 1
			forceID                    1
			HeartbeatTimer             5.000000
			DistanceThresh             1.000000
			OrientThresh               3.000000

		End Output


		Begin Time

			Mode                       rtPDUTimestamp

		End Time


		Begin PDUInfo


		End PDUInfo


		Begin Parameters

			ParmData  COLORFRIENDLY        blue
			ParmData  COLORNEUTRAL         white
			ParmData  COLOROPFORCE         red
			ParmData  MAXDRELSETS          1000

		End Parameters


		Begin Network

			NetIF                      Default
			Mode                       Broadcast
			McastIP                    224.0.0.1
			Port                       3000
			rChannelBufferSize         65000
			ReadBufferSize             1500
			QueuePollPeriod            20
			MaxRcvQueueEntries         1000
			MaxRcvIOThreads            4
			sChannelBufferSize         65000

		End Network


		Begin EntityTypeDef


#			order: kind:domain:country:catagory:subCatagory:specific:xtra ( -1 = * )


		End EntityTypeDef


		Begin EntityFilter
			Include                    *:*:*
		End EntityFilter

    END DIS

END Extensions

BEGIN SubObjects

Class AreaTarget

	Thailand

END Class

Class Constellation

	Constellation_Plane1

END Class

Class CoverageDefinition

	TH_Cov

END Class

Class Facility

	Chiangmai_Facility
	Communication_Site

END Class

Class Place

	LOGSAT_GCS

END Class

Class Satellite

	LOGSAT1
	LOGSAT1101
	LOGSAT1102
	LOGSAT1103
	LOGSAT1104
	LOGSAT1105
	LOGSAT1106
	LOGSAT1107
	LOGSAT1108
	LOGSAT1109
	LOGSAT1110
	LOGSAT1111
	LOGSAT1112
	LOGSAT1113
	LOGSAT1114
	LOGSAT1115
	LOGSAT1116
	LOGSAT1117
	LOGSAT1118
	LOGSAT1201
	LOGSAT1202
	LOGSAT1203
	LOGSAT1204
	LOGSAT1205
	LOGSAT1206
	LOGSAT1207
	LOGSAT1208
	LOGSAT1209
	LOGSAT1210
	LOGSAT1211
	LOGSAT1212
	LOGSAT1213
	LOGSAT1214
	LOGSAT1215
	LOGSAT1216
	LOGSAT1217
	LOGSAT1218
	LOGSAT1301
	LOGSAT1302
	LOGSAT1303
	LOGSAT1304
	LOGSAT1305
	LOGSAT1306
	LOGSAT1307
	LOGSAT1308
	LOGSAT1309
	LOGSAT1310
	LOGSAT1311
	LOGSAT1312
	LOGSAT1313
	LOGSAT1314
	LOGSAT1315
	LOGSAT1316
	LOGSAT1317
	LOGSAT1318
	TELEOS2_56310

END Class

END SubObjects

BEGIN References
    Instance *
        *
        CoverageDefinition/TH_Cov
    END Instance
    Instance AreaTarget/Thailand
        AreaTarget/Thailand
        CoverageDefinition/TH_Cov
    END Instance
    Instance Constellation/Constellation_Plane1
    END Instance
    Instance CoverageDefinition/TH_Cov
        CoverageDefinition/TH_Cov/FigureOfMerit/CovTimeTotal
        CoverageDefinition/TH_Cov/FigureOfMerit/SimpleCov_vs_Time
        CoverageDefinition/TH_Cov/FigureOfMerit/TimeAvgGap
    END Instance
    Instance CoverageDefinition/TH_Cov/FigureOfMerit/CovTimeTotal
    END Instance
    Instance CoverageDefinition/TH_Cov/FigureOfMerit/SimpleCov_vs_Time
    END Instance
    Instance CoverageDefinition/TH_Cov/FigureOfMerit/TimeAvgGap
    END Instance
    Instance Facility/Chiangmai_Facility
        Facility/Chiangmai_Facility
    END Instance
    Instance Facility/Communication_Site
        Facility/Communication_Site
        Facility/Communication_Site/Receiver/Uplink_Ka_Rx
    END Instance
    Instance Facility/Communication_Site/Receiver/Uplink_Ka_Rx
        Facility/Communication_Site/Receiver/Uplink_Ka_Rx
    END Instance
    Instance Place/LOGSAT_GCS
        Place/LOGSAT_GCS
    END Instance
    Instance Satellite/LOGSAT1
        Satellite/LOGSAT1
    END Instance
    Instance Satellite/LOGSAT1101
        Satellite/LOGSAT1101
        Satellite/LOGSAT1101/Sensor/Sensor1
        Satellite/LOGSAT1101/Transmitter/Downlink_Ka_Tx
        Satellite/LOGSAT1101/Transmitter/Downlink_S_Tx
    END Instance
    Instance Satellite/LOGSAT1101/Sensor/Sensor1
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1101/Sensor/Sensor1
        Satellite/LOGSAT1101/Sensor/Sensor1/Transmitter/Transmitter_SensorFOV
    END Instance
    Instance Satellite/LOGSAT1101/Sensor/Sensor1/Transmitter/Transmitter_SensorFOV
        Satellite/LOGSAT1101/Sensor/Sensor1/Transmitter/Transmitter_SensorFOV
    END Instance
    Instance Satellite/LOGSAT1101/Transmitter/Downlink_Ka_Tx
        Satellite/LOGSAT1101/Transmitter/Downlink_Ka_Tx
    END Instance
    Instance Satellite/LOGSAT1101/Transmitter/Downlink_S_Tx
        Satellite/LOGSAT1101/Transmitter/Downlink_S_Tx
    END Instance
    Instance Satellite/LOGSAT1102
        Satellite/LOGSAT1102
        Satellite/LOGSAT1102/Sensor/Sensor2
    END Instance
    Instance Satellite/LOGSAT1102/Sensor/Sensor2
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1102/Sensor/Sensor2
        Satellite/LOGSAT1102/Sensor/Sensor2/Transmitter/Downlink_Ka_Tx_SensorFOV
    END Instance
    Instance Satellite/LOGSAT1102/Sensor/Sensor2/Transmitter/Downlink_Ka_Tx_SensorFOV
        Satellite/LOGSAT1102/Sensor/Sensor2/Transmitter/Downlink_Ka_Tx_SensorFOV
    END Instance
    Instance Satellite/LOGSAT1103
        Satellite/LOGSAT1103
        Satellite/LOGSAT1103/Sensor/Sensor3
    END Instance
    Instance Satellite/LOGSAT1103/Sensor/Sensor3
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1103/Sensor/Sensor3
    END Instance
    Instance Satellite/LOGSAT1104
        Satellite/LOGSAT1104
        Satellite/LOGSAT1104/Sensor/Sensor4
    END Instance
    Instance Satellite/LOGSAT1104/Sensor/Sensor4
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1104/Sensor/Sensor4
    END Instance
    Instance Satellite/LOGSAT1105
        Satellite/LOGSAT1105
        Satellite/LOGSAT1105/Sensor/Sensor5
    END Instance
    Instance Satellite/LOGSAT1105/Sensor/Sensor5
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1105/Sensor/Sensor5
    END Instance
    Instance Satellite/LOGSAT1106
        Satellite/LOGSAT1106
        Satellite/LOGSAT1106/Sensor/Sensor6
    END Instance
    Instance Satellite/LOGSAT1106/Sensor/Sensor6
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1106/Sensor/Sensor6
    END Instance
    Instance Satellite/LOGSAT1107
        Satellite/LOGSAT1107
        Satellite/LOGSAT1107/Sensor/Sensor7
    END Instance
    Instance Satellite/LOGSAT1107/Sensor/Sensor7
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1107/Sensor/Sensor7
    END Instance
    Instance Satellite/LOGSAT1108
        Satellite/LOGSAT1108
        Satellite/LOGSAT1108/Sensor/Sensor8
    END Instance
    Instance Satellite/LOGSAT1108/Sensor/Sensor8
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1108/Sensor/Sensor8
    END Instance
    Instance Satellite/LOGSAT1109
        Satellite/LOGSAT1109
        Satellite/LOGSAT1109/Sensor/Sensor9
    END Instance
    Instance Satellite/LOGSAT1109/Sensor/Sensor9
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1109/Sensor/Sensor9
    END Instance
    Instance Satellite/LOGSAT1110
        Satellite/LOGSAT1110
        Satellite/LOGSAT1110/Sensor/Sensor10
    END Instance
    Instance Satellite/LOGSAT1110/Sensor/Sensor10
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1110/Sensor/Sensor10
    END Instance
    Instance Satellite/LOGSAT1111
        Satellite/LOGSAT1111
        Satellite/LOGSAT1111/Sensor/Sensor11
    END Instance
    Instance Satellite/LOGSAT1111/Sensor/Sensor11
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1111/Sensor/Sensor11
    END Instance
    Instance Satellite/LOGSAT1112
        Satellite/LOGSAT1112
        Satellite/LOGSAT1112/Sensor/Sensor12
    END Instance
    Instance Satellite/LOGSAT1112/Sensor/Sensor12
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1112/Sensor/Sensor12
    END Instance
    Instance Satellite/LOGSAT1113
        Satellite/LOGSAT1113
        Satellite/LOGSAT1113/Sensor/Sensor13
    END Instance
    Instance Satellite/LOGSAT1113/Sensor/Sensor13
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1113/Sensor/Sensor13
    END Instance
    Instance Satellite/LOGSAT1114
        Satellite/LOGSAT1114
        Satellite/LOGSAT1114/Sensor/Sensor14
    END Instance
    Instance Satellite/LOGSAT1114/Sensor/Sensor14
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1114/Sensor/Sensor14
    END Instance
    Instance Satellite/LOGSAT1115
        Satellite/LOGSAT1115
        Satellite/LOGSAT1115/Sensor/Sensor15
    END Instance
    Instance Satellite/LOGSAT1115/Sensor/Sensor15
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1115/Sensor/Sensor15
    END Instance
    Instance Satellite/LOGSAT1116
        Satellite/LOGSAT1116
        Satellite/LOGSAT1116/Sensor/Sensor16
    END Instance
    Instance Satellite/LOGSAT1116/Sensor/Sensor16
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1116/Sensor/Sensor16
    END Instance
    Instance Satellite/LOGSAT1117
        Satellite/LOGSAT1117
        Satellite/LOGSAT1117/Sensor/Sensor17
    END Instance
    Instance Satellite/LOGSAT1117/Sensor/Sensor17
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1117/Sensor/Sensor17
    END Instance
    Instance Satellite/LOGSAT1118
        Satellite/LOGSAT1118
        Satellite/LOGSAT1118/Sensor/Sensor18
    END Instance
    Instance Satellite/LOGSAT1118/Sensor/Sensor18
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1118/Sensor/Sensor18
    END Instance
    Instance Satellite/LOGSAT1201
        Satellite/LOGSAT1201
        Satellite/LOGSAT1201/Sensor/Sensor19
    END Instance
    Instance Satellite/LOGSAT1201/Sensor/Sensor19
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1201/Sensor/Sensor19
    END Instance
    Instance Satellite/LOGSAT1202
        Satellite/LOGSAT1202
        Satellite/LOGSAT1202/Sensor/Sensor20
    END Instance
    Instance Satellite/LOGSAT1202/Sensor/Sensor20
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1202/Sensor/Sensor20
    END Instance
    Instance Satellite/LOGSAT1203
        Satellite/LOGSAT1203
        Satellite/LOGSAT1203/Sensor/Sensor21
    END Instance
    Instance Satellite/LOGSAT1203/Sensor/Sensor21
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1203/Sensor/Sensor21
    END Instance
    Instance Satellite/LOGSAT1204
        Satellite/LOGSAT1204
        Satellite/LOGSAT1204/Sensor/Sensor22
    END Instance
    Instance Satellite/LOGSAT1204/Sensor/Sensor22
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1204/Sensor/Sensor22
    END Instance
    Instance Satellite/LOGSAT1205
        Satellite/LOGSAT1205
        Satellite/LOGSAT1205/Sensor/Sensor23
    END Instance
    Instance Satellite/LOGSAT1205/Sensor/Sensor23
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1205/Sensor/Sensor23
    END Instance
    Instance Satellite/LOGSAT1206
        Satellite/LOGSAT1206
        Satellite/LOGSAT1206/Sensor/Sensor24
    END Instance
    Instance Satellite/LOGSAT1206/Sensor/Sensor24
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1206/Sensor/Sensor24
    END Instance
    Instance Satellite/LOGSAT1207
        Satellite/LOGSAT1207
        Satellite/LOGSAT1207/Sensor/Sensor25
    END Instance
    Instance Satellite/LOGSAT1207/Sensor/Sensor25
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1207/Sensor/Sensor25
    END Instance
    Instance Satellite/LOGSAT1208
        Satellite/LOGSAT1208
        Satellite/LOGSAT1208/Sensor/Sensor26
    END Instance
    Instance Satellite/LOGSAT1208/Sensor/Sensor26
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1208/Sensor/Sensor26
    END Instance
    Instance Satellite/LOGSAT1209
        Satellite/LOGSAT1209
        Satellite/LOGSAT1209/Sensor/Sensor27
    END Instance
    Instance Satellite/LOGSAT1209/Sensor/Sensor27
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1209/Sensor/Sensor27
    END Instance
    Instance Satellite/LOGSAT1210
        Satellite/LOGSAT1210
        Satellite/LOGSAT1210/Sensor/Sensor28
    END Instance
    Instance Satellite/LOGSAT1210/Sensor/Sensor28
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1210/Sensor/Sensor28
    END Instance
    Instance Satellite/LOGSAT1211
        Satellite/LOGSAT1211
        Satellite/LOGSAT1211/Sensor/Sensor29
    END Instance
    Instance Satellite/LOGSAT1211/Sensor/Sensor29
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1211/Sensor/Sensor29
    END Instance
    Instance Satellite/LOGSAT1212
        Satellite/LOGSAT1212
        Satellite/LOGSAT1212/Sensor/Sensor30
    END Instance
    Instance Satellite/LOGSAT1212/Sensor/Sensor30
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1212/Sensor/Sensor30
    END Instance
    Instance Satellite/LOGSAT1213
        Satellite/LOGSAT1213
        Satellite/LOGSAT1213/Sensor/Sensor31
    END Instance
    Instance Satellite/LOGSAT1213/Sensor/Sensor31
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1213/Sensor/Sensor31
    END Instance
    Instance Satellite/LOGSAT1214
        Satellite/LOGSAT1214
        Satellite/LOGSAT1214/Sensor/Sensor32
    END Instance
    Instance Satellite/LOGSAT1214/Sensor/Sensor32
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1214/Sensor/Sensor32
    END Instance
    Instance Satellite/LOGSAT1215
        Satellite/LOGSAT1215
        Satellite/LOGSAT1215/Sensor/Sensor33
    END Instance
    Instance Satellite/LOGSAT1215/Sensor/Sensor33
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1215/Sensor/Sensor33
    END Instance
    Instance Satellite/LOGSAT1216
        Satellite/LOGSAT1216
        Satellite/LOGSAT1216/Sensor/Sensor34
    END Instance
    Instance Satellite/LOGSAT1216/Sensor/Sensor34
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1216/Sensor/Sensor34
    END Instance
    Instance Satellite/LOGSAT1217
        Satellite/LOGSAT1217
        Satellite/LOGSAT1217/Sensor/Sensor35
    END Instance
    Instance Satellite/LOGSAT1217/Sensor/Sensor35
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1217/Sensor/Sensor35
    END Instance
    Instance Satellite/LOGSAT1218
        Satellite/LOGSAT1218
        Satellite/LOGSAT1218/Sensor/Sensor36
    END Instance
    Instance Satellite/LOGSAT1218/Sensor/Sensor36
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1218/Sensor/Sensor36
    END Instance
    Instance Satellite/LOGSAT1301
        Satellite/LOGSAT1301
        Satellite/LOGSAT1301/Sensor/Sensor37
    END Instance
    Instance Satellite/LOGSAT1301/Sensor/Sensor37
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1301/Sensor/Sensor37
    END Instance
    Instance Satellite/LOGSAT1302
        Satellite/LOGSAT1302
        Satellite/LOGSAT1302/Sensor/Sensor38
    END Instance
    Instance Satellite/LOGSAT1302/Sensor/Sensor38
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1302/Sensor/Sensor38
    END Instance
    Instance Satellite/LOGSAT1303
        Satellite/LOGSAT1303
        Satellite/LOGSAT1303/Sensor/Sensor39
    END Instance
    Instance Satellite/LOGSAT1303/Sensor/Sensor39
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1303/Sensor/Sensor39
    END Instance
    Instance Satellite/LOGSAT1304
        Satellite/LOGSAT1304
        Satellite/LOGSAT1304/Sensor/Sensor40
    END Instance
    Instance Satellite/LOGSAT1304/Sensor/Sensor40
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1304/Sensor/Sensor40
    END Instance
    Instance Satellite/LOGSAT1305
        Satellite/LOGSAT1305
        Satellite/LOGSAT1305/Sensor/Sensor41
    END Instance
    Instance Satellite/LOGSAT1305/Sensor/Sensor41
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1305/Sensor/Sensor41
    END Instance
    Instance Satellite/LOGSAT1306
        Satellite/LOGSAT1306
        Satellite/LOGSAT1306/Sensor/Sensor42
    END Instance
    Instance Satellite/LOGSAT1306/Sensor/Sensor42
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1306/Sensor/Sensor42
    END Instance
    Instance Satellite/LOGSAT1307
        Satellite/LOGSAT1307
        Satellite/LOGSAT1307/Sensor/Sensor43
    END Instance
    Instance Satellite/LOGSAT1307/Sensor/Sensor43
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1307/Sensor/Sensor43
    END Instance
    Instance Satellite/LOGSAT1308
        Satellite/LOGSAT1308
        Satellite/LOGSAT1308/Sensor/Sensor44
    END Instance
    Instance Satellite/LOGSAT1308/Sensor/Sensor44
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1308/Sensor/Sensor44
    END Instance
    Instance Satellite/LOGSAT1309
        Satellite/LOGSAT1309
        Satellite/LOGSAT1309/Sensor/Sensor45
    END Instance
    Instance Satellite/LOGSAT1309/Sensor/Sensor45
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1309/Sensor/Sensor45
    END Instance
    Instance Satellite/LOGSAT1310
        Satellite/LOGSAT1310
        Satellite/LOGSAT1310/Sensor/Sensor46
    END Instance
    Instance Satellite/LOGSAT1310/Sensor/Sensor46
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1310/Sensor/Sensor46
    END Instance
    Instance Satellite/LOGSAT1311
        Satellite/LOGSAT1311
        Satellite/LOGSAT1311/Sensor/Sensor47
    END Instance
    Instance Satellite/LOGSAT1311/Sensor/Sensor47
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1311/Sensor/Sensor47
    END Instance
    Instance Satellite/LOGSAT1312
        Satellite/LOGSAT1312
        Satellite/LOGSAT1312/Sensor/Sensor48
    END Instance
    Instance Satellite/LOGSAT1312/Sensor/Sensor48
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1312/Sensor/Sensor48
    END Instance
    Instance Satellite/LOGSAT1313
        Satellite/LOGSAT1313
        Satellite/LOGSAT1313/Sensor/Sensor49
    END Instance
    Instance Satellite/LOGSAT1313/Sensor/Sensor49
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1313/Sensor/Sensor49
    END Instance
    Instance Satellite/LOGSAT1314
        Satellite/LOGSAT1314
        Satellite/LOGSAT1314/Sensor/Sensor50
    END Instance
    Instance Satellite/LOGSAT1314/Sensor/Sensor50
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1314/Sensor/Sensor50
    END Instance
    Instance Satellite/LOGSAT1315
        Satellite/LOGSAT1315
        Satellite/LOGSAT1315/Sensor/Sensor51
    END Instance
    Instance Satellite/LOGSAT1315/Sensor/Sensor51
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1315/Sensor/Sensor51
    END Instance
    Instance Satellite/LOGSAT1316
        Satellite/LOGSAT1316
        Satellite/LOGSAT1316/Sensor/Sensor52
    END Instance
    Instance Satellite/LOGSAT1316/Sensor/Sensor52
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1316/Sensor/Sensor52
    END Instance
    Instance Satellite/LOGSAT1317
        Satellite/LOGSAT1317
        Satellite/LOGSAT1317/Sensor/Sensor53
    END Instance
    Instance Satellite/LOGSAT1317/Sensor/Sensor53
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1317/Sensor/Sensor53
    END Instance
    Instance Satellite/LOGSAT1318
        Satellite/LOGSAT1318
        Satellite/LOGSAT1318/Sensor/Sensor54
    END Instance
    Instance Satellite/LOGSAT1318/Sensor/Sensor54
        CoverageDefinition/TH_Cov
        Satellite/LOGSAT1318/Sensor/Sensor54
    END Instance
    Instance Satellite/TELEOS2_56310
        Satellite/TELEOS2_56310
    END Instance
END References

END Scenario
