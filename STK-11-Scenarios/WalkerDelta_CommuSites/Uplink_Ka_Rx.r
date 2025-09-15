stk.v.11.0
WrittenBy    STK_v11.2.0

BEGIN	Receiver

Name	Uplink_Ka_Rx
<?xml version = "1.0" standalone = "yes"?>
<VAR name = "STK_Receiver_Object">
    <SCOPE Class = "CommRadarObject">
        <VAR name = "Version">
            <STRING>&quot;1.0.0 a&quot;</STRING>
        </VAR>
        <VAR name = "ComponentName">
            <STRING>&quot;STK_Receiver_Object&quot;</STRING>
        </VAR>
        <VAR name = "Description">
            <STRING>&quot;STK Receiver Object&quot;</STRING>
        </VAR>
        <VAR name = "Type">
            <STRING>&quot;STK Receiver Object&quot;</STRING>
        </VAR>
        <VAR name = "UserComment">
            <STRING>&quot;STK Receiver Object&quot;</STRING>
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
            <VAR name = "Simple_Receiver_Model">
                <SCOPE Class = "Receiver">
                    <VAR name = "Version">
                        <STRING>&quot;1.0.0 a&quot;</STRING>
                    </VAR>
                    <VAR name = "ComponentName">
                        <STRING>&quot;Simple_Receiver_Model&quot;</STRING>
                    </VAR>
                    <VAR name = "Description">
                        <STRING>&quot;Simple model of a receiver&quot;</STRING>
                    </VAR>
                    <VAR name = "Type">
                        <STRING>&quot;Simple Receiver Model&quot;</STRING>
                    </VAR>
                    <VAR name = "UserComment">
                        <STRING>&quot;Simple model of a receiver&quot;</STRING>
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
                    <VAR name = "AutoSelectDemodulator">
                        <BOOL>true</BOOL>
                    </VAR>
                    <VAR name = "Demodulator">
                        <VAR name = "BPSK">
                            <SCOPE Class = "Demodulator">
                                <VAR name = "Version">
                                    <STRING>&quot;1.0.0 a&quot;</STRING>
                                </VAR>
                                <VAR name = "ComponentName">
                                    <STRING>&quot;BPSK&quot;</STRING>
                                </VAR>
                                <VAR name = "Description">
                                    <STRING>&quot;Demodulator capable of demodulating a BPSK modulated signal.&quot;</STRING>
                                </VAR>
                                <VAR name = "Type">
                                    <STRING>&quot;BPSK&quot;</STRING>
                                </VAR>
                                <VAR name = "UserComment">
                                    <STRING>&quot;Demodulator capable of demodulating a BPSK modulated signal.&quot;</STRING>
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
                            </SCOPE>
                        </VAR>
                    </VAR>
                    <VAR name = "UseFilter">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "Filter">
                        <VAR name = "Butterworth">
                            <SCOPE Class = "Filter">
                                <VAR name = "Version">
                                    <STRING>&quot;1.0.0 a&quot;</STRING>
                                </VAR>
                                <VAR name = "ComponentName">
                                    <STRING>&quot;Butterworth&quot;</STRING>
                                </VAR>
                                <VAR name = "Description">
                                    <STRING>&quot;General form of nth order Butterworth filter with flat passband and stopband regions&quot;</STRING>
                                </VAR>
                                <VAR name = "Type">
                                    <STRING>&quot;Butterworth&quot;</STRING>
                                </VAR>
                                <VAR name = "UserComment">
                                    <STRING>&quot;General form of nth order Butterworth filter with flat passband and stopband regions&quot;</STRING>
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
                                <VAR name = "LowerBandwidthLimit">
                                    <QUANTITY Dimension = "BandwidthUnit" Unit = "Hz">
                                        <REAL>-20000000</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "UpperBandwidthLimit">
                                    <QUANTITY Dimension = "BandwidthUnit" Unit = "Hz">
                                        <REAL>20000000</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "InsertionLoss">
                                    <QUANTITY Dimension = "RatioUnit" Unit = "units">
                                        <REAL>1</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "Order">
                                    <INT>4</INT>
                                </VAR>
                                <VAR name = "CutoffFrequency">
                                    <QUANTITY Dimension = "BandwidthUnit" Unit = "Hz">
                                        <REAL>10000000</REAL>
                                    </QUANTITY>
                                </VAR>
                            </SCOPE>
                        </VAR>
                    </VAR>
                    <VAR name = "Bandwidth">
                        <QUANTITY Dimension = "BandwidthUnit" Unit = "Hz">
                            <REAL>2000</REAL>
                        </QUANTITY>
                    </VAR>
                    <VAR name = "AutoScaleBandwidth">
                        <BOOL>true</BOOL>
                    </VAR>
                    <VAR name = "PreReceiveGainsLosses">
                        <SCOPE>
                            <VAR name = "GainLossList">
                                <LIST />
                            </VAR>
                        </SCOPE>
                    </VAR>
                    <VAR name = "PreDemodGainsLosses">
                        <SCOPE>
                            <VAR name = "GainLossList">
                                <LIST />
                            </VAR>
                        </SCOPE>
                    </VAR>
                    <VAR name = "EnableLinkMargin">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "LinkMarginType">
                        <STRING>&quot;Eb/No&quot;</STRING>
                    </VAR>
                    <VAR name = "LinkMarginThreshold">
                        <QUANTITY Dimension = "RatioUnit" Unit = "units">
                            <REAL>1</REAL>
                        </QUANTITY>
                    </VAR>
                    <VAR name = "RainOutagePercent">
                        <PROP name = "FormatString">
                            <STRING>&quot;%#6.3f&quot;</STRING>
                        </PROP>
                        <REAL>0.1</REAL>
                    </VAR>
                    <VAR name = "UseRain">
                        <BOOL>true</BOOL>
                    </VAR>
                    <VAR name = "GOverT">
                        <QUANTITY Dimension = "GainTempRatio" Unit = "units*K^-1">
                            <REAL>3.981071705534972</REAL>
                        </QUANTITY>
                    </VAR>
                    <VAR name = "UsePolarization">
                        <BOOL>false</BOOL>
                    </VAR>
                    <VAR name = "Polarization">
                        <VAR name = "Linear">
                            <SCOPE Class = "Polarization">
                                <VAR name = "ReferenceAxis">
                                    <STRING>&quot;X Axis&quot;</STRING>
                                </VAR>
                                <VAR name = "TiltAngle">
                                    <QUANTITY Dimension = "AngleUnit" Unit = "rad">
                                        <REAL>0</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "CrossPolLeakage">
                                    <QUANTITY Dimension = "RatioUnit" Unit = "units">
                                        <REAL>1e-06</REAL>
                                    </QUANTITY>
                                </VAR>
                                <VAR name = "Type">
                                    <STRING>&quot;Linear&quot;</STRING>
                                </VAR>
                            </SCOPE>
                        </VAR>
                    </VAR>
                    <VAR name = "Frequency">
                        <QUANTITY Dimension = "FrequencyUnit" Unit = "Hz">
                            <REAL>14500000000</REAL>
                        </QUANTITY>
                    </VAR>
                    <VAR name = "FrequencyAutoTracking">
                        <BOOL>true</BOOL>
                    </VAR>
                </SCOPE>
            </VAR>
        </VAR>
    </SCOPE>
</VAR>
END	Receiver

BEGIN Extensions
    
    BEGIN ExternData
    END ExternData
    
    BEGIN ADFFileData
    END ADFFileData
    
    BEGIN AccessConstraints
		LineOfSight   IncludeIntervals 
    END AccessConstraints
    
    BEGIN ObjectCoverage
    END ObjectCoverage
    
    BEGIN Desc
    END Desc
    
    BEGIN Refraction
		RefractionModel	Effective Radius Method

		UseRefractionInAccess		No

		BEGIN ModelData
			RefractionCeiling	5.00000000000000e+03
			MaxTargetAltitude	1.00000000000000e+04
			EffectiveRadius		1.33333333333333e+00

			UseExtrapolation	 Yes


		END ModelData
    END Refraction
    
    BEGIN Crdn
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	Chiangmai_Facility
			Description	Displacement vector to Chiangmai_Facility
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Facility/Chiangmai_Facility
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1
			Description	Displacement vector to LOGSAT1
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1101
			Description	Displacement vector to LOGSAT1101
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1101
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1101-Downlink_Ka_Tx
			Description	Displacement vector to LOGSAT1101-Downlink_Ka_Tx
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1101/Transmitter/Downlink_Ka_Tx
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1101-Sensor1
			Description	Displacement vector to LOGSAT1101-Sensor1
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1101/Sensor/Sensor1
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1101-Sensor1-Transmitter_SensorFOV
			Description	Displacement vector to LOGSAT1101-Sensor1-Transmitter_SensorFOV
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1101/Sensor/Sensor1/Transmitter/Transmitter_SensorFOV
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1102
			Description	Displacement vector to LOGSAT1102
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1102
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1102-Sensor2
			Description	Displacement vector to LOGSAT1102-Sensor2
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1102/Sensor/Sensor2
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1102-Sensor2-Downlink_Ka_Tx_SensorFOV
			Description	Displacement vector to LOGSAT1102-Sensor2-Downlink_Ka_Tx_SensorFOV
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1102/Sensor/Sensor2/Transmitter/Downlink_Ka_Tx_SensorFOV
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1103
			Description	Displacement vector to LOGSAT1103
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1103
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1103-Sensor3
			Description	Displacement vector to LOGSAT1103-Sensor3
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1103/Sensor/Sensor3
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1104
			Description	Displacement vector to LOGSAT1104
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1104
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1104-Sensor4
			Description	Displacement vector to LOGSAT1104-Sensor4
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1104/Sensor/Sensor4
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1105
			Description	Displacement vector to LOGSAT1105
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1105
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1105-Sensor5
			Description	Displacement vector to LOGSAT1105-Sensor5
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1105/Sensor/Sensor5
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1106
			Description	Displacement vector to LOGSAT1106
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1106
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1106-Sensor6
			Description	Displacement vector to LOGSAT1106-Sensor6
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1106/Sensor/Sensor6
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1107
			Description	Displacement vector to LOGSAT1107
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1107
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1107-Sensor7
			Description	Displacement vector to LOGSAT1107-Sensor7
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1107/Sensor/Sensor7
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1108
			Description	Displacement vector to LOGSAT1108
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1108
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1108-Sensor8
			Description	Displacement vector to LOGSAT1108-Sensor8
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1108/Sensor/Sensor8
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1109
			Description	Displacement vector to LOGSAT1109
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1109
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1109-Sensor9
			Description	Displacement vector to LOGSAT1109-Sensor9
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1109/Sensor/Sensor9
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1110
			Description	Displacement vector to LOGSAT1110
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1110
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1110-Sensor10
			Description	Displacement vector to LOGSAT1110-Sensor10
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1110/Sensor/Sensor10
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1111
			Description	Displacement vector to LOGSAT1111
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1111
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1111-Sensor11
			Description	Displacement vector to LOGSAT1111-Sensor11
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1111/Sensor/Sensor11
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1112
			Description	Displacement vector to LOGSAT1112
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1112
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1112-Sensor12
			Description	Displacement vector to LOGSAT1112-Sensor12
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1112/Sensor/Sensor12
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1113
			Description	Displacement vector to LOGSAT1113
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1113
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1113-Sensor13
			Description	Displacement vector to LOGSAT1113-Sensor13
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1113/Sensor/Sensor13
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1114
			Description	Displacement vector to LOGSAT1114
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1114
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1114-Sensor14
			Description	Displacement vector to LOGSAT1114-Sensor14
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1114/Sensor/Sensor14
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1115
			Description	Displacement vector to LOGSAT1115
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1115
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1115-Sensor15
			Description	Displacement vector to LOGSAT1115-Sensor15
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1115/Sensor/Sensor15
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1116
			Description	Displacement vector to LOGSAT1116
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1116
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1116-Sensor16
			Description	Displacement vector to LOGSAT1116-Sensor16
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1116/Sensor/Sensor16
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1117
			Description	Displacement vector to LOGSAT1117
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1117
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1117-Sensor17
			Description	Displacement vector to LOGSAT1117-Sensor17
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1117/Sensor/Sensor17
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1118
			Description	Displacement vector to LOGSAT1118
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1118
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1118-Sensor18
			Description	Displacement vector to LOGSAT1118-Sensor18
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1118/Sensor/Sensor18
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1201
			Description	Displacement vector to LOGSAT1201
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1201
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1201-Sensor19
			Description	Displacement vector to LOGSAT1201-Sensor19
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1201/Sensor/Sensor19
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1202
			Description	Displacement vector to LOGSAT1202
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1202
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1202-Sensor20
			Description	Displacement vector to LOGSAT1202-Sensor20
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1202/Sensor/Sensor20
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1203
			Description	Displacement vector to LOGSAT1203
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1203
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1203-Sensor21
			Description	Displacement vector to LOGSAT1203-Sensor21
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1203/Sensor/Sensor21
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1204
			Description	Displacement vector to LOGSAT1204
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1204
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1204-Sensor22
			Description	Displacement vector to LOGSAT1204-Sensor22
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1204/Sensor/Sensor22
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1205
			Description	Displacement vector to LOGSAT1205
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1205
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1205-Sensor23
			Description	Displacement vector to LOGSAT1205-Sensor23
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1205/Sensor/Sensor23
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1206
			Description	Displacement vector to LOGSAT1206
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1206
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1206-Sensor24
			Description	Displacement vector to LOGSAT1206-Sensor24
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1206/Sensor/Sensor24
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1207
			Description	Displacement vector to LOGSAT1207
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1207
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1207-Sensor25
			Description	Displacement vector to LOGSAT1207-Sensor25
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1207/Sensor/Sensor25
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1208
			Description	Displacement vector to LOGSAT1208
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1208
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1208-Sensor26
			Description	Displacement vector to LOGSAT1208-Sensor26
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1208/Sensor/Sensor26
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1209
			Description	Displacement vector to LOGSAT1209
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1209
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1209-Sensor27
			Description	Displacement vector to LOGSAT1209-Sensor27
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1209/Sensor/Sensor27
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1210
			Description	Displacement vector to LOGSAT1210
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1210
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1210-Sensor28
			Description	Displacement vector to LOGSAT1210-Sensor28
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1210/Sensor/Sensor28
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1211
			Description	Displacement vector to LOGSAT1211
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1211
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1211-Sensor29
			Description	Displacement vector to LOGSAT1211-Sensor29
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1211/Sensor/Sensor29
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1212
			Description	Displacement vector to LOGSAT1212
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1212
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1212-Sensor30
			Description	Displacement vector to LOGSAT1212-Sensor30
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1212/Sensor/Sensor30
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1213
			Description	Displacement vector to LOGSAT1213
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1213
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1213-Sensor31
			Description	Displacement vector to LOGSAT1213-Sensor31
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1213/Sensor/Sensor31
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1214
			Description	Displacement vector to LOGSAT1214
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1214
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1214-Sensor32
			Description	Displacement vector to LOGSAT1214-Sensor32
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1214/Sensor/Sensor32
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1215
			Description	Displacement vector to LOGSAT1215
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1215
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1215-Sensor33
			Description	Displacement vector to LOGSAT1215-Sensor33
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1215/Sensor/Sensor33
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1216
			Description	Displacement vector to LOGSAT1216
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1216
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1216-Sensor34
			Description	Displacement vector to LOGSAT1216-Sensor34
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1216/Sensor/Sensor34
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1217
			Description	Displacement vector to LOGSAT1217
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1217
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1217-Sensor35
			Description	Displacement vector to LOGSAT1217-Sensor35
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1217/Sensor/Sensor35
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1218
			Description	Displacement vector to LOGSAT1218
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1218
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1218-Sensor36
			Description	Displacement vector to LOGSAT1218-Sensor36
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1218/Sensor/Sensor36
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1301
			Description	Displacement vector to LOGSAT1301
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1301
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1301-Sensor37
			Description	Displacement vector to LOGSAT1301-Sensor37
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1301/Sensor/Sensor37
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1302
			Description	Displacement vector to LOGSAT1302
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1302
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1302-Sensor38
			Description	Displacement vector to LOGSAT1302-Sensor38
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1302/Sensor/Sensor38
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1303
			Description	Displacement vector to LOGSAT1303
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1303
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1303-Sensor39
			Description	Displacement vector to LOGSAT1303-Sensor39
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1303/Sensor/Sensor39
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1304
			Description	Displacement vector to LOGSAT1304
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1304
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1304-Sensor40
			Description	Displacement vector to LOGSAT1304-Sensor40
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1304/Sensor/Sensor40
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1305
			Description	Displacement vector to LOGSAT1305
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1305
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1305-Sensor41
			Description	Displacement vector to LOGSAT1305-Sensor41
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1305/Sensor/Sensor41
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1306
			Description	Displacement vector to LOGSAT1306
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1306
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1306-Sensor42
			Description	Displacement vector to LOGSAT1306-Sensor42
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1306/Sensor/Sensor42
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1307
			Description	Displacement vector to LOGSAT1307
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1307
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1307-Sensor43
			Description	Displacement vector to LOGSAT1307-Sensor43
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1307/Sensor/Sensor43
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1308
			Description	Displacement vector to LOGSAT1308
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1308
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1308-Sensor44
			Description	Displacement vector to LOGSAT1308-Sensor44
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1308/Sensor/Sensor44
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1309
			Description	Displacement vector to LOGSAT1309
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1309
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1309-Sensor45
			Description	Displacement vector to LOGSAT1309-Sensor45
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1309/Sensor/Sensor45
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1310
			Description	Displacement vector to LOGSAT1310
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1310
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1310-Sensor46
			Description	Displacement vector to LOGSAT1310-Sensor46
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1310/Sensor/Sensor46
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1311
			Description	Displacement vector to LOGSAT1311
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1311
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1311-Sensor47
			Description	Displacement vector to LOGSAT1311-Sensor47
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1311/Sensor/Sensor47
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1312
			Description	Displacement vector to LOGSAT1312
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1312
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1312-Sensor48
			Description	Displacement vector to LOGSAT1312-Sensor48
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1312/Sensor/Sensor48
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1313
			Description	Displacement vector to LOGSAT1313
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1313
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1313-Sensor49
			Description	Displacement vector to LOGSAT1313-Sensor49
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1313/Sensor/Sensor49
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1314
			Description	Displacement vector to LOGSAT1314
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1314
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1314-Sensor50
			Description	Displacement vector to LOGSAT1314-Sensor50
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1314/Sensor/Sensor50
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1315
			Description	Displacement vector to LOGSAT1315
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1315
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1315-Sensor51
			Description	Displacement vector to LOGSAT1315-Sensor51
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1315/Sensor/Sensor51
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1316
			Description	Displacement vector to LOGSAT1316
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1316
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1316-Sensor52
			Description	Displacement vector to LOGSAT1316-Sensor52
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1316/Sensor/Sensor52
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1317
			Description	Displacement vector to LOGSAT1317
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1317
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1317-Sensor53
			Description	Displacement vector to LOGSAT1317-Sensor53
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1317/Sensor/Sensor53
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1318
			Description	Displacement vector to LOGSAT1318
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1318
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT1318-Sensor54
			Description	Displacement vector to LOGSAT1318-Sensor54
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/LOGSAT1318/Sensor/Sensor54
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	LOGSAT_GCS
			Description	Displacement vector to LOGSAT_GCS
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Place/LOGSAT_GCS
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	TELEOS2_56310
			Description	Displacement vector to TELEOS2_56310
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	Satellite/TELEOS2_56310
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
		BEGIN	VECTOR
			Type	VECTOR_TOVECTOR
			Name	Thailand
			Description	Displacement vector to Thailand
				Origin
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
					END	POINT
				Destination
					BEGIN	POINT
						Type	POINT_LINKTO
						Name	Center
						RelativePath	AreaTarget/Thailand
					END	POINT
				LTDRefSystem
					BEGIN	SYSTEM
						Type	SYSTEM_LINKTO
						Name	BarycenterICRF
						AbsolutePath	CentralBody/Sun
					END	SYSTEM
				Apparent	No
				TimeConvergence	 1.0000000000000000e-03
				TimeSense	Receive
				IgnoreAberration	No
		END	VECTOR
    END Crdn
    
    BEGIN Graphics

BEGIN Graphics

	ShowGfx           On
	Relative          Off
	ShowBoresight     On
	BoresightMarker   4
	BoresightColor    #ffffff

END Graphics
    END Graphics
    
    BEGIN ContourGfx
	ShowContours      Off
    END ContourGfx
    
    BEGIN Contours
	ActiveContourType Antenna Gain

	BEGIN ContourSet Antenna Gain
		Altitude          0.000000e+00
		ShowAtAltitude    Off
		Projected         On
		Relative          On
		ShowLabels        Off
		LineWidth         1.000000
		DecimalDigits     1
		ColorRamp         On
		ColorRampStartColor   #0000ff
		ColorRampEndColor     #ff0000
		BEGIN ContourDefinition
		BEGIN CntrAntAzEl
			BEGIN AzElPattern
				BEGIN AzElPatternDef
					SetResolutionTogether 0
					CoordinateSystem 0
					NumAzPoints      181
					AzimuthRes       2.000000
					MinAzimuth       -180.000000
					MaxAzimuth       180.000000
					NumElPoints      91
					ElevationRes     1.000000
					MinElevation     0.000000
					MaxElevation     90.000000
				END AzElPatternDef
			END AzElPattern
		END CntrAntAzEl
		END ContourDefinition
	END ContourSet
    END Contours
    
    BEGIN VO
    END VO
    
    BEGIN 3dVolume
        ActiveVolumeType  Antenna Beam

        BEGIN VolumeSet Antenna Beam
            Scale 4.000000
            NumericGainOffset 1.000000
            Frequency 14500000000.000000
            ShowAsWireframe 0
				BEGIN AzElPatternDef
					SetResolutionTogether 0
					CoordinateSystem 0
					NumAzPoints      181
					AzimuthRes       2.000000
					MinAzimuth       -180.000000
					MaxAzimuth       180.000000
					NumElPoints      91
					ElevationRes     1.000000
					MinElevation     0.000000
					MaxElevation     90.000000
				END AzElPatternDef
        END VolumeSet
        BEGIN VolumeGraphics
            ShowContours    No
            ShowVolume No
        END VolumeGraphics
    END 3dVolume

END Extensions
