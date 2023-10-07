default:
	nvcc mySgemm_final.cu -o mySgemm_final.exe -arch=compute_61 -code=sm_61 -lcublas --ptxas-options=-v -maxrregcount=128

run:
	mySgemm_final.exe
