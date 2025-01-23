EXECUTABLE := HITS
CUFILES := HITS.cu

all:
	nvcc -O3 ${CUFILES} -o ${EXECUTABLE} 
keep:
	nvcc -arch=sm_60 -keep -o ${EXECUTABLE} ${CUFILES}

dry:
	nvcc -dryrun -arch=sm_60 -o ${EXECUTABLE} ${CUFILES} --keep 2>dryrun.out

clean:
	rm -f *.ptx *.ii *.i *.c *.cpp *.gpu *.fatbin *.hash *.o *.cubin *.module_id dryrun.out

clobber:
	rm -f *~ *.ptx *.ii *.i *.c *.cpp *.gpu *.fatbin *.hash *.o *.cubin *.module_id *.out
	rm -f  *.log
	rm -f outcome.txt

fi: 
	rm -f outcome.txt
	rm -r result
	rm -r golden

