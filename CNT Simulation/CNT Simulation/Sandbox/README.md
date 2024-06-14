# CNT Parameter Calculation
Contributors: Waqas Khalid for calculation and framework, Tyler Wang for plotting functionality_

Takes in a set of input material/design parameters and outputs the expected values for the capacitance and other parameters.
Contains univariate plotting functionality.

To run simulations, use `New Main.vi`, add in input parameters (make sure none are zero to avoid division by zero errors), and run the program.

Bulk of calculation is performed in `Single Calculation.vi`, but note that the block diagram is difficult to read due to issues with bookkeeping for large multivariate systems in LabVIEW.