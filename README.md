# Emod3d_SGT

The emod3d code is set up to perform reciprocal calculations. Based on your notes, I think you have the basic understanding of the process.

The main steps are:

1) determine the target source locations and create an “sgtcoord” file (gen_sgtgrid)
2) run emod3d for three orthogonal body forces (fx, fy, fz) inserted at recording site location and save output SGTs at the target source locations
3) generate velocity seismograms for recording site for either a point double-couple or SRF finite-fault rupture (ptsim3d, jbsim3d)

The codes listed in the ()’s above are pre- and post-processing codes. Also, I don’t have a general CMT code, but the ptsim3d code could be easily modified to accept a generalized moment-tensor input.

#####################################

I’ve now installed an example for the SGT computation along with documentation on the NeSI system. To get started, please take a look at the “README” file in /home/rgraves/SGT-Example. All of the pre- and post-processing codes (source and executable) are installed in my home directory. I don’t provide information in the documentation on file formats. To get this, you’ll have to look at the source codes (usually in structure.h).

I’ve tested the example and it works as expected. The attached plot compares forward and reciprocal computations for three point sources. The waveforms show extremely close agreement (note: differences in later arrivals are due to boundary reflections which do not obey reciprocity).

#################################

My comment means that the x,y,z coordinate system used for the moment tensor has to be the same orientation as the x,y,z coordinate system used for the SGTs. In general, this means one of them needs to be rotated and since it is a lot easier to rotate the moment tensor than the SGTs, that is my recommendation.

 

SGT Orientation:

The x,y,z orientation of the SGTs is determined by the orientation of the finite-difference grid. The x-axis points in the azimuth of 90.0 - MODEL_ROT, and the y-axis is x-azimuth + 90.0. The z-axis is always positive down.

 

What is the value of MODEL_ROT for you SGT calculation?

 

Moment-tensor Orientation:

I used Aki & Richards for the moment tensor components. My recollection is that they define x=000, y=090, z=down. So I rotate their x,y,z by an amount of (90.0 – MODEL_ROT) so that their “x” aligns with the x-axis of the SGTs. This basically boils down to adjusting the strike of the fault. After computing the 3-component motions, I rotate the horizontals back to 000,090. This is all done in point_mech_sgt() in the file sgt3d_subs.c.

 

Producing a version of ptsim3d that can be compiled by itself requires a fair bit of work. This is because of various libraries that it needs. But since you have access to the source codes, I think you have all you need to sort this out.

 
