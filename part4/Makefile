prepare:
	objcopy ../part3/one.o ./one.o --redefine-sym myfunc=func1 --redefine-sym _mlir_ciface_myfunc=func_l
	objcopy ../part3/two.o ./two.o --redefine-sym myfunc=func2 --redefine-sym _mlir_ciface_myfunc=func_o

compile:
	g++ main.cpp one.o two.o -o main

clean:
	rm main one.o two.o
