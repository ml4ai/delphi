The files in this directory are taken from the source distribution of SWAT (Soil and Water Assessment Tool) executable.

-- To build the executable:

        make

    This should create an executable file 'main.out'

-- To execute the file: run the command

        ./main.out

    This will run the program on the input files corresponding to those in

        <SWAT root dir>/data/TxtInOut_CoonCreek_aqu/


    The program's execution will write a number of lines to stdout reporting on the progress of the simulation, and dump the results of the simulation to a number of output files.  The output files created are listed in a file 'files_out.out'.  The reference versions of the output files are:

    REF-basin_aqu_aa.txt
    REF-basin_crop_yld_aa.txt
    REF-basin_crop_yld_yr.txt
    REF-basin_pw_yr.txt
    REF-basin_wb_aa.txt
    REF-basin_wb_yr.txt
    REF-channel_sdmorph_yr.txt
    REF-channel_sd_yr.txt
    REF-crop_yld_aa.txt
    REF-hru_pw_aa.txt
    REF-hru_wb_aa.txt

    The files listed above do not include one of the output files, hru_wb_day.txt, which is over 400M in size and so exceeds github's file size limit.  For testing purposes, it may suffice to focus on the other files.
