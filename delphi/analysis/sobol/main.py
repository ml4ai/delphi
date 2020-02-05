from delphi.GrFN.networks import GroundedFunctionNetwork as GrFN
from sensitivity import SobolIndex
from visualization import SobolVisualizer

def main(model, file_bounds, sample_size):

    sobol = SobolIndex(model, file_bounds, sample_size)
    sobol_dict = sobol.sobol_index_from_GrFN(GrFN)
    
    sobol_plots = SobolVisualizer(model)
    sobol_plots.S1_Sobol_plot(sobol_dict)
    sobol_plots.S2_Sobol_plot(sobol_dict)
    sobol_plots.clocktime_sobol_plot(sobol_dict)
    
if __name__ == '__main__':

    model = 'PETPT'
    file_bounds = 'petpt_var_bounds.csv'
    sample_size = [10, 100, 1000, 10000]
    main(model, file_bounds, sample_size)

    model = 'PETASCE'
    file_bounds = 'petasce_var_bounds.csv'
    main(model, file_bounds, sample_size)
