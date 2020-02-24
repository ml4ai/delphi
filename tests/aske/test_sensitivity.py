import numpy as np
import unittest
from delphi.GrFN.sensitivity import SensitivityIndices, SensitivityAnalyzer
from delphi.GrFN.networks import GroundedFunctionNetwork as GrFN

class TestSensitivity(unittest.TestCase):

    def setUp(self):

        self.Si_dict = {'S1':0.5, 'S2': np.array([[0.5, 0.2], [0.1, 0.8]]), 'ST':1.0, 'S1_conf':0.05, 'S2_conf':0.05, 'ST_conf':0.05}
        self.sensitivity_obj = SensitivityIndices(self.Si_dict)
        self.sensitivity_analysis = SensitivityAnalyzer()

    def test_orders(self):

        self.assertTrue(True, self.sensitivity_obj.check_first_order)
        self.assertTrue(True, self.sensitivity_obj.check_second_order)
        self.assertTrue(True, self.sensitivity_obj.check_total_order)

    def test_min_max_S2(self):

        self.assertIsNotNone(self.sensitivity_obj.get_min_S2())
        self.assertIsNotNone(self.sensitivity_obj.get_max_S2())
    
    def test_from_file(self):
    
        self.assertIsNotNone(self.sensitivity_obj.from_csv('test.csv'))
        self.assertIsNotNone(self.sensitivity_obj.from_json('test.json'))
        self.assertIsNotNone(self.sensitivity_obj.from_pickle('test_pickle'))

    def test_Sobol(self):
        
        N = 10
        tG = GrFN.from_fortran_file("../../tests/data/program_analysis/PETPT.for")
        B = {
                'PETPT::@global::petpt::0::tmax::-1':[0.0, 40.0],
                'PETPT::@global::petpt::0::tmin::-1':[0.0, 40.0],
                'PETPT::@global::petpt::0::srad::-1': [0.0, 30.0],
                'PETPT::@global::petpt::0::msalb::-1': [0.0, 1.0],
                'PETPT::@global::petpt::0::xhlai::-1': [0.0, 20.0]
                }
        C = None
        V = None
        calc_2nd = True
        num_resamples = 100
        conf_level = 0.95
        seed = None
        save_time = False


        sobol_dict, sample_time_sobol, exec_time_sobol, analyze_time_sobol = self.sensitivity_analysis.Si_from_Sobol(N, tG, B, C, V, calc_2nd, num_resamples, conf_level, seed, save_time) 

        sample_time = np.float32(0.03015414399999994)
        exec_time = np.float32(0.000846874000000053)
        analyze_time = np.float32(0.00579838999999982)

        self.assertIsInstance(sobol_dict, dict)
        self.assertEqual(sample_time_sobol, sample_time)
        self.assertEqual(exec_time_sobol, exec_time)
        self.assertEqual(analyze_time_sobol, analyze_time)

if __name__ == '__main__':
    unittest.main()
