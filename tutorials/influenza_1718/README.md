# Contents

The files in this demo illustrate how a simple ODE or stochastic jump process model can be setup and calibrated to data.

# What happens in this tutorial?

First, by running the script `data_conversion.py`, the user converts the *raw* weekly incidence of Influenza cases in Belgium (per 100K inhabitats) during the 2017-2018 Influenza season into a better suited format (daily incidence data, not per 100K inhabitants). Second, `calibration.py` can be executed to calibrate an epidemiological model to the Influenza incidence data. In this script, first, a SEIR-like model is loaded from `models.py` and initialized. Then, an *objective function* for the model calibration is initialized. This *objective function* tells our optimization algorithms what deviations between model prediction and data are statistically sound and which ones are not. In this example, as Influenza data are incidence data resulting from a counting experiment, a Poisson objective function is used. The calibration procedure will first run the Particle Swarm Optimiser (PSO) to scan the parameter space for a global optimum. Then, it will run the Nelder-Mead simplex optimization starting from the *optimal parameters* found by the PSO. Then, the *optimal parameters* are perturbated and a Markov-Chain Monte-Carlo sampling algorithm is started to sample our Poisson objective function. The Markov-Chain Monte-Carlo sampling will generate a folder `sampler_output/`. In this folder, every 5 iterations, diagnostic figures (named autocorrelation and trace plots) and a backup of the sampler will be saved so you can monitor progress while the sampler is running (usefull for models that take much longer to run). Finally, the model is simulated *N* times with parameters drawn from the distributions obtained from the sampler. Poisson observational noise is added to every model trajectory.

# Files

+ `data_conversion.py`: Converts the raw dataset into the interim dataset.
+ `models.py`: Contains the definition of the SEIR-like ODE and stochastic jump process model for Influenza.
+ `calibration.py`: Calibrates the SEIR-like ODE model for Influenza to incidence data.

## Data

> Remember: A naive user should be able to reproduce all your results using the raw data only. Making a `readme.md` with good descriptions on the origins and contents of every dataset is always a good idea!

### Raw

+ `social_contacts_holiday_week.xlsx`: Copy of the social contact data used in the Influenza model. Extracted using SOCRATES (http://www.socialcontactdata.org/socrates/). Contacts Mon-Fri (holidays). Includes physical contacts longer than 15 minutes. An integration of the contacts with the contact duration is performed in the spreadsheet.

+ `social_contacts_noholiday_week.xlsx`: Copy of the social contact data used in the Influenza model. Extracted using SOCRATES (http://www.socialcontactdata.org/socrates/). Contacts Mon-Fri (excl. holidays). Includes physical contacts longer than 15 minutes. An integration of the contacts with the contact duration is performed in the spreadsheet.

+ `social_contacts_weekend.xlsx`: Copy of the social contact data used in the Influenza model. Extracted using SOCRATES (http://www.socialcontactdata.org/socrates/). Contacts Sat-Sun. Includes physical contacts longer than 15 minutes. An integration of the contacts with the contact duration is performed in the spreadsheet.

+ `Influenza 2017-2018 End of Season_NL.pdf`: End of Influenza season report of the Belgian Scientific Institute of Public Health (Sciensano). Retrieved from [Sciensano](https://www.sciensano.be/sites/default/files/influenza_2017-2018_end_of_season_nl.pdf) (accessed Nov. 9 2022).

+ `dataset_influenza_1718.csv`: Weekly incidence of Influenza cases in Belgium (per 100K inhabitats) during the 2017-2018 Influenza season. Data available for four age groups: [0,5(, [5,15(, [15,65(, [65,120(. Extracted from Fig. 2 in `Influenza 2017-2018 End of Season_NL.pdf` using [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/).

### Interim

+ `ILI_weekly_100K.csv`:  Weekly incidence of GP visits for Influenza-like illness per 100K inhabitants in Belgium during the 2017-2018 season. Data available for four age groups: [0,5(, [5,15(, [15,65(, [65,120(. Dates are the reported week number's midpoint. Generated from `dataset_influenza_1718.csv` by executing the data conversion script `data_conversion.py`.

+ `ILI_weekly_ABS.csv`: Weekly incidence of GP visits for Influenza-like illness in Belgium during the 2017-2018 season. Data available for four age groups: [0,5(, [5,15(, [15,65(, [65,120(. Dates are the reported week number's midpoint. Generated from `dataset_influenza_1718.csv` by executing the data conversion script `data_conversion.py`.
