from io import StringIO
import os
from eppy.modeleditor import IDF
from openstudio import energyplus, osversion
from doe_xstock.utilities import get_data_from_path, write_data

class OpenStudioModelEditor:
    def __init__(self,osm):
        self.osm = osm

    @property
    def osm(self):
        return self.__osm

    @osm.setter
    def osm(self,osm):
        self.__osm = get_data_from_path(osm)

    def forward_translate(self):
        osm = self.__get_model()
        forward_translator = energyplus.ForwardTranslator()
        idf = forward_translator.translateModel(osm)
        idf = str(idf)
        return idf

    def use_ideal_loads_air_system(self):
        # Reference: https://www.rubydoc.info/gems/openstudio-standards/Standard#remove_hvac-instance_method
        osm = self.__get_model()

        # remove air loop hvac
        for air_loop in osm.getAirLoopHVACs():
            air_loop.remove()

        # remove plant loops
        for plant_loop in osm.getPlantLoops():
            shw_use = False

            for component in plant_loop.demandComponents():
                if component.to_WaterUseConnections().is_initialized() or component.to_CoilWaterHeatingDesuperheater().is_initialized():
                    shw_use = True
                    break
                else:
                    pass

            if  not shw_use:
                plant_loop.remove()
            else:
                continue

        # remove vrf
        for ac_refrigerant_flow in osm.getAirConditionerVariableRefrigerantFlows():
            ac_refrigerant_flow.remove()

        for terminal_unit_refrigerant_flow in osm.getZoneHVACTerminalUnitVariableRefrigerantFlows():
            terminal_unit_refrigerant_flow.remove()

        # remove zone equipment
        for zone in osm.getThermalZones():
            for equipment in zone.equipment():
                if not equipment.to_FanZoneExhaust().is_initialized():
                    equipment.remove()
                else:
                    pass

        # remove unused curves
        for curve in osm.getCurves():
            if curve.directUseCount() == 0:
                curve.remove()
            else:
                pass

        # add ideal load system
        for zone in osm.getThermalZones():
            zone.setUseIdealAirLoads(True)
            
        osm = str(osm)
        self.osm = osm

    def __get_model(self):
        version_translator = osversion.VersionTranslator()
        osm = version_translator.loadModelFromString(self.osm).get()
        return osm

class Simulator:
    def __init__(self,idd_filepath,idf,epw,simulation_id=None,output_directory=None):
        self.idd_filepath = idd_filepath
        self.epw = epw
        self.idf = idf
        self.simulation_id = simulation_id
        self.output_directory = output_directory
    
    @property
    def idd_filepath(self):
        return self.__idd_filepath

    @property
    def idf(self):
        return self.__idf

    @property
    def epw(self):
        return self.__epw

    @property
    def simulation_id(self):
        return self.__simulation_id

    @property
    def output_directory(self):
        return self.__output_directory

    @idd_filepath.setter
    def idd_filepath(self,idd_filepath):
        self.__idd_filepath = idd_filepath
        IDF.setiddname(self.idd_filepath)

    @idf.setter
    def idf(self,idf):
        self.__idf = get_data_from_path(idf)

    @epw.setter
    def epw(self,epw):
        epw = get_data_from_path(epw)
        self.__epw = epw

    @simulation_id.setter
    def simulation_id(self,simulation_id):
        self.__simulation_id = simulation_id if simulation_id is not None else 'simulation'

    @output_directory.setter
    def output_directory(self,output_directory):
        self.__output_directory = output_directory if output_directory is not None else 'simulation'

    def simulate(self,**run_kwargs):
        os.makedirs(self.output_directory,exist_ok=True)
        self.__write_epw()
        self.__write_idf()
        run_kwargs = self.__get_run_kwargs(**run_kwargs if run_kwargs is not None else {})
        idf = self.get_idf_object(weather=self.__epw_filepath) 
        idf.run(**run_kwargs)

    def __get_run_kwargs(self,**kwargs):
        idf = self.get_idf_object()
        idf_version = idf.idfobjects['version'][0].Version_Identifier.split('.')
        idf_version.extend([0] * (3 - len(idf_version)))
        idf_version_str = '-'.join([str(item) for item in idf_version])
        options = {
            'ep_version':idf_version_str,
            'output_prefix':self.simulation_id,
            'output_suffix':'C',
            'output_directory':self.output_directory,
            'readvars':True,
            'expandobjects':True,
            'verbose':'q',
        }
        options = {**options,**kwargs}
        return options

    def __write_epw(self):
        filepath = os.path.join(self.output_directory,'weather.epw')
        write_data(self.epw,filepath)
        self.__epw_filepath = filepath

    def __write_idf(self):
        filepath = os.path.join(self.output_directory,f'{self.simulation_id}.idf')
        write_data(self.idf,filepath)

    def get_idf_object(self,weather=None):
        return IDF(StringIO(self.idf),weather)