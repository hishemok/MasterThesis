### Just a file to sort my plans and thoughts

*Task 1*:
Since we are almost in goal, we need to figure out one more thing.
We have in braiding/ folder and especially braiding_model.py braiding_model_ad.py and braiding_model_ad2.py figured out that braiding our optimized majoranas works very well in the reduced ground state space $P_gs$ which is an 8x8 dimentional ground state basis.

We need to figure out if braiding works either with or without interactions for P > P_gs.
We expect non-interacting to work, and interacting system to not work, but we hope the interacting system still can work.

To do this we need a couple of things. 
Non-interacting majorana configurations should be easy to make since they are saved in configurations.json
We always include all 4 energylevels of the majoranas, for realism. 
We need to figure out the dimensions of H_full's degenerate spaces. We know P_gs is 8x8, but there is no guarantee that the first excited basis is 8x8 as well, so we need to group them up by some tolerance of degeneracy to figure out the dimensions to include.

Then we need to run the braiding protocol (most likely with the correct majoranas we found in braiding_model_ad.py - meaning i\gamma_B2 and i\gamma_C2). For both U=0 and U=0.1, for the three dot system. 

We will do this gradually beginning with P_gs, and increasing excited level by excited level, to explore where the braiding protocol falls apart for both U =0 and U=0.1. 

The results needs to be represented in a table where the main thing to show is dimensions and braiding error, so that we can see when the errors became zero. 


-- First --
Energy level Dimensions of H_full

-- Second --
Prepare Majoranas with U=0 and U=0.1

-- Third --
Run P_gs for γ_U=0 and γ_U=0.1

-- Four --
Run P_gs + P_excited for the same γ's

