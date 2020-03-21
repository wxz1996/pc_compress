TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

/usr/local/cuda/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc tf_nndistance.cu -o tf_nndistance.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc tf_interpolate.cu -o tf_interpolate.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc tf_recerse_knn.cu -o tf_recerse_knn.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc tf_auctionmatch_g.cu -o tf_auctionmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
/usr/local/cuda/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_nndistance.cpp tf_nndistance.cu.o -o tf_nndistance.so -shared -fPIC -I $TF_LIB/include -I /usr/local/cuda/include -I $TF_LIB/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ ${TF_LFLAGS[@]} -O2  -D_GLIBCXX_USE_CXX11_ABI=0

# g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I $TF_LIB/include -I /usr/local/cuda/include -I $TF_LIB/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I $TF_LIB/include -I /usr/local/cuda/include -I $TF_LIB/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

g++ -std=c++11 tf_interpolate.cpp tf_interpolate.cu.o -o tf_interpolate.so -shared -fPIC -I $TF_LIB/include -I /usr/local/cuda/include -I $TF_LIB/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1

g++ -std=c++11 tf_recerse_knn.cpp tf_recerse_knn.cu.o -o tf_recerse_knn.so -shared -fPIC -I $TF_LIB/include -I /usr/local/cuda/include -I $TF_LIB/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1

g++ -std=c++11 tf_auctionmatch.cpp tf_auctionmatch_g.cu.o -o tf_auctionmatch_so.so -shared -fPIC -I $TF_LIB/include -I /usr/local/cuda/include -I $TF_LIB/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $TF_LIB/include -I /usr/local/cuda/include -I $TF_LIB/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0