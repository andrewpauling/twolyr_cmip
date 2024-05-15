#!/bin/bash

if [ ! -f 'processed.zip' ]; then
    echo "Downloading processed CMIP6 data from Zenodo"
    wget 'https://zenodo.org/record/7686736/files/processed_v2.zip'
else
    echo "Already downloaded processed CMIP6 data from Zenodo"
fi

if [ ! -d processed ]; then
    echo "Unzipping CMIP6 data"
    unzip processed_v2.zip
else
    echo "Processed CMIP6 data already unzipped"
fi

if [ ! -d ecsdata ]; then
    echo "Getting ECS data from Mark Zelinka GitHub repo"
    wget https://zenodo.org/record/5206851/files/mzelinka/cmip56_forcing_feedback_ecs-v2.0.zip
    unzip cmip56_forcing_feedback_ecs-v2.0.zip
    mv mzelinka-cmip56_forcing_feedback_ecs-2fc752f ecsdata
else
    echo "Already downloaded ECS data from Mark Zelinka GitHub repo"
fi

if [ ! -d forcingdata ]; then
    echo "Getting Schmidt et al. forcing data"
    mkdir forcingdata
    cd forcingdata
    wget https://gdex.ucar.edu/dataset/2018_Schmidt_JGR_Volcanic_RF/file.zip
    unzip file.zip
    cd ../
else
    echo "Already downloaded Schmidt et al. forcing data"
fi

if [ ! -d 'obsdata' ]; then
    echo "Getting observational data"
    mkdir obsdata
    cd obsdata

    wget 'https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/analysis/diagnostics/HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.nc'

    for yr in {1985..1999}; do
	outfile="ERBE_S10N_WFOV_SF_ERBS_Regional_Edition4.1.36Day_${yr}.nc"
	wget "https://opendap.larc.nasa.gov/opendap/ERBE/S10N/WFOV_SF_ERBS_Regional_Edition4.1/${yr}/${outfile}"
	ncks -O --mk_rec_dmn time ${outfile} ${outfile}
	ncpdq -O -a time,lat,lon ${outfile} ${outfile}
    done

    ncrcat -d time,0,149 ERBE_S10N_WFOV_SF_ERBS_Regional_Edition4.1.36Day_19* ERBE_S10N_WFOV_SF_ERBS_Regional_Edition4.1.36Day_1985-1999.nc

    cd ../
fi

echo "All data downloaded"
    
