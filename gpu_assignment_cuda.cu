#include "cuda.cuh"
#include <cstring>
#include "helper.h"
#include <stdlib.h>
#include <string.h>


///
/// Algorithm storage
///




////////////////////////////////////////////
/* Define the variables I am going to use */
////////////////////////////////////////////


// Host copy of input image
Image cuda_input_image;
// Host copy of output image
Image cuda_output_image;
// Host copy of image tiles numbers in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;


// Pointer to host buffer for calculating the sum of each tile mosaic
unsigned long long* h_mosaic_sum;
// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned long long* d_mosaic_sum;

// Pointer to host buffer for storing the output pixels of each tile
unsigned char* h_mosaic_value;
// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned char* d_mosaic_value;


// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned long long* d_global_pixel_sum;





// function to initialize necessary variables
void cuda_begin(const Image* input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    // total tiles' number in x and y direction:
    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;

    // define a variable to represent the whole tile numbers for all channels
    const int tile_num = cuda_TILES_X * cuda_TILES_Y * input_image->channels;

    // Allocate buffer for calculating the sum of each tile mosaic on gpu memory
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, tile_num * sizeof(unsigned long long)));
    // Allocate buffer for calculating the sum of each tile mosaic on cpu memory
    h_mosaic_sum = (unsigned long long*)malloc(tile_num * sizeof(unsigned long long));
    // Initialize the values of "h_mosaic_sum" to 0
    memset(h_mosaic_sum, 0, tile_num * sizeof(unsigned long long));
    // initialize the d_mosaic_sum to zero
    CUDA_CALL(cudaMemcpy(d_mosaic_sum, h_mosaic_sum, tile_num * sizeof(unsigned long long), cudaMemcpyHostToDevice));



    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, tile_num * sizeof(unsigned char)));
    // Allocate buffer for storing the output pixel value of each tile
    h_mosaic_value = (unsigned char*)malloc(tile_num * sizeof(unsigned char));


    const size_t image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);
    // Allocate copy of input image on CPU
    cuda_input_image = *input_image;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_input_image.data, input_image->data, image_data_size);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));

    // Allocate copy of output image
    cuda_output_image = *input_image;
    cuda_output_image.data = (unsigned char*)malloc(image_data_size);

    // Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));
}








///////////////////////////////////
// define the kernel for stage 1 //
///////////////////////////////////

__global__ void kernel_stage1(unsigned char* d_input_image_data, unsigned long long* d_mosaic_sum) {

    /*
        Note:
            gridDim.x = cuda_TILES_X; - tiles' numbers in x dimension
            gridDim.y = cuda_TILES_Y; - tiles' numbers in y dimension

            blockDim.x = TILE_SIZE; - 32
            blockDim.y = cuda_input_image->channels; - 3

        Each block will process one tile, then each warp in one tile would process one color channel.
    */

    // Declare an array on shared memory to store the 3 channels - RGB - values of each tile:
    __shared__ unsigned long long tile_rgb[3];


    // For each tile (block), the "tile_rgb" will be shared, therefore, we only need one thread to initialize "tile_rgb":
    // Let (1, 1) thread in each block to initialize the shared variable "tile_rgb".
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tile_rgb[0] = 0;
        tile_rgb[1] = 0;
        tile_rgb[2] = 0;
    }

    __syncthreads();



    // Compute the tile offset:
    // Here, x coordinate of tile equals to blockIdx.x, y coordinate of tile equals to blockIdx.y;
    // image_width = cuda_TILES_X * TILE_SIZE, where our gridDim.x = cuda_TILE_X;
    // then, our blockDim.y = cuda_input_image.channels = 3
    // Therefore, we could get following tile_offset:
    const unsigned int tile_offset = (blockIdx.y * gridDim.x * TILE_SIZE * TILE_SIZE + blockIdx.x * TILE_SIZE) * blockDim.y;

    // Compute the tile index:
    const unsigned int tile_index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y;


    // Declare a variable to represent the y coordinate of pixel in a tile ( imagine our tile is a 2d space, 32 rows (y) and 32*3 columns (x) )
    int p_y;

    // Declare a variable to represent channel index:
    int ch;

    // Note, in each block, we have 3 warps. Each warp will process one channel of one row, each thread in one warp will process one pixel of one channel of one row.
    // Here, we define a variable to represent in one row, which pixel is being processed by current thread.
    // Note: this column index is different with the one in kernel 3.
    // Because in kernel 1, threads in one warp only access the pixel value from one single color channel.
    // It means they could not access input image data sequentially.
    // But in kernel 3, threads in one warp would sequentially access 32 pixel values including all three color channels.
    int column_index = blockDim.y * threadIdx.x + threadIdx.y;


    // Declare a local variable to store the sum of one column
    unsigned long long thread_pixel_value = 0;

    // Declare the pixel offset
    unsigned int pixel_offset;

    // sum one column's values
    for (p_y = 0; p_y < TILE_SIZE; p_y++) {
        // Calculate the pixel offset
        pixel_offset = p_y * gridDim.x * TILE_SIZE * blockDim.y;

        thread_pixel_value += d_input_image_data[tile_offset + pixel_offset + column_index];

        __syncwarp();
    }


    // warp_size = 32
    // Then, sum all 32 columns for one colour channel by warp addition operation
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_pixel_value += __shfl_down(thread_pixel_value, offset);
    }


    // Here, for each block, we will have three values, one for red channel's sum, one for green's and one for blue's
    // We need to write these three values to the block shared memory "tile_rgb"
    if (threadIdx.x % 32 == 0) {
        tile_rgb[threadIdx.y] = thread_pixel_value;
    }

    __syncthreads();


    // Let one thread in each block to write the rgb values of this tile to our global variable "d_mosaic_sum"
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (ch = 0; ch < blockDim.y; ch++) {
            d_mosaic_sum[tile_index + ch] = tile_rgb[ch];
        }
    }

}



// call kernel 1 to calculate the sum for each tile
void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);


    // define the dimension of grid, here I decide the blocks num = tiles num, thus 1 block process 1 tile
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y);

    // define the dimension of block, here I decide a size of TILE_SIZE * channel.
    // each warp could process one color channel of this tile
    dim3 threadsPerBlock(TILE_SIZE, cuda_input_image.channels);


    // call the kernel
    kernel_stage1 << <blocksPerGrid, threadsPerBlock >> > (d_input_image_data, d_mosaic_sum);



#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)

    CUDA_CALL(cudaMemcpy(h_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    validate_tile_sum(&cuda_input_image, h_mosaic_sum);
#endif
}




///////////////////////////////////
// define the kernel for stage 2 //
///////////////////////////////////

__global__ void kernel_stage2(unsigned char* d_mosaic_value, unsigned long long* d_mosaic_sum, unsigned long long* d_global_pixel_sum, unsigned int cuda_TILES_X, unsigned int cuda_TILES_Y) {

    /*
        Note:
            blockDim.x = warp_size = 32;
            blockDim.y = cuda_input_image.channels = 3;
            gridDim.x = ceil( (float) cuda_TILES_X * cuda_TILES_Y / warp_size);


        Each block processes 32 tiles's sum values;
        Because each tile has three color channels, therefore, we need 32*3 threads in one block to process one tile.
        The threads in the same warp will process the same color channel.
    */


    // Declare a local variable to store the
    unsigned long long thread_tile_value = 0;


    int i = threadIdx.x * blockDim.y + threadIdx.y + blockIdx.x * blockDim.x * blockDim.y;


    // Kill those threads whose indices are out of the length of "d_mosaic_value".
    if (i < cuda_TILES_X* cuda_TILES_Y * 3) {


        thread_tile_value = d_mosaic_sum[i] / TILE_PIXELS;

        d_mosaic_value[i] = thread_tile_value;

    }

    // Through a warp addition, we could get the sum of one color channel of 32 tiles
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_tile_value += __shfl_down(thread_tile_value, offset);
    }

    // Here, we could get the red sum, blue sum, and green sum of each 32 tiles, and the values are stored on the first threads of each warp respectively.
    // What we need to do is to sum these red, blue, and green sum of each 32 tiles together.
    if (threadIdx.x % 32 == 0) {

        atomicAdd(&d_global_pixel_sum[threadIdx.y], thread_tile_value);
    }

}



// call the kernel 2 to calculate the averaget of each tile, and calculate the global pixel average
void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);

    unsigned long long h_global_pixel_sum[3] = { 0, 0, 0 };

    float grid_dim = ceil((float)cuda_TILES_X * cuda_TILES_Y / 32);

    dim3 blocksPerGrid((int)grid_dim);

    // Our block dimension is (32, 3). It means each block could process 32 tiles and each warp processes one color channel.
    dim3 threadsPerBlock(32, cuda_input_image.channels);

    kernel_stage2 << <blocksPerGrid, threadsPerBlock >> > (d_mosaic_value, d_mosaic_sum, d_global_pixel_sum, cuda_TILES_X, cuda_TILES_Y);

    // copy the global pixel sum value from device to host to calculate the global pixel average
    CUDA_CALL(cudaMemcpy(h_global_pixel_sum, d_global_pixel_sum, cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));


    // calculate the global_average by using h_global_pixel_sum
    for (int ch = 0; ch < cuda_input_image.channels; ++ch) {
        output_global_average[ch] = (unsigned char)(h_global_pixel_sum[ch] / (cuda_TILES_X * cuda_TILES_Y));
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)

    CUDA_CALL(cudaMemcpy(h_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, h_mosaic_sum, h_mosaic_value, output_global_average);
#endif    
}







///////////////////////////////////
// define the kernel for stage 3 //
///////////////////////////////////

__global__ void kernel_stage3(unsigned char* d_mosaic_value, unsigned char* d_output_image_data) {

    /*
        NOTE:
            blockIdx.x = tile_x_coordinate
            blockIdx.y = tile_y_coordinate

        Here, we has the same thread configuration as stage 1.
        Each block will process one tile, and each warp will process one color channel in this tile.
    */

    // declare a shared memory to store the pixel value for one row ( all rows in one tile are same, so it could be regarded as the rgb values for one tile)
    __shared__ unsigned char tile_rgb[32 * 3];

    // declare and calculate the tile index:
    const unsigned int tile_index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y;

    // Note, in each block, we have 3 warps. Each warp will process one channel in one row, each thread in one warp will process one pixel of one color pixel in one row.
    // Here, we define a variable to represent: in one row, which pixel is being processed by current thread.
    unsigned int column_index = blockDim.x * threadIdx.y + threadIdx.x;

    // it will be stored like "rgbrgbrgb..." for one row 
    tile_rgb[column_index] = d_mosaic_value[tile_index + (column_index % 3)];

    __syncwarp(); // Because the threads of a warp will only access the part of shared memory which they have already initialized,
                  // and this threads of a warp will not access the part which is initialized by other warps.
                  // Here we don't need __syncthreads() but a __syncwarp()


    // Now, each tile's rgb values are prepared, we need to broadcast these rgb value to our output image
    // Define the tile_offset
    const unsigned int tile_offset = (blockIdx.y * gridDim.x * TILE_SIZE * TILE_SIZE + blockIdx.x * TILE_SIZE) * blockDim.y;

    // Declare a variable to represent the y coordinate in a tile:
    int p_y;

    for (p_y = 0; p_y < TILE_SIZE; p_y++) {
        d_output_image_data[tile_offset + p_y * gridDim.x * TILE_SIZE * blockDim.y + column_index] = tile_rgb[column_index];
        __syncwarp();
    }


    //// old version without using shared memory
    // 
    //for (p_y = 0; p_y < TILE_SIZE; p_y++) {
    //    d_output_image_data[tile_offset + p_y * gridDim.x * TILE_SIZE * blockDim.y + column_index] = d_mosaic_value[tile_index + (column_index % 3)];
    //    __syncwarp();
    //}
}




// Call kernel 3 to broadcast pixel value to output image
void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, d_output_image);


    // the thread configuration of kernel 3 is the same as kernel 1's

    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y);

    dim3 threadsPerBlock(TILE_SIZE, cuda_input_image.channels);

    kernel_stage3 << <blocksPerGrid, threadsPerBlock >> > (d_mosaic_value, d_output_image_data);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)

    CUDA_CALL(cudaMemcpy(cuda_output_image.data, d_output_image_data, cuda_input_image.height * cuda_input_image.width * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));


    validate_broadcast(&cuda_input_image, h_mosaic_value, &cuda_output_image);
#endif    
}




void cuda_end(Image* output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    output_image->width = cuda_input_image.width;
    output_image->height = cuda_input_image.height;
    output_image->channels = cuda_input_image.channels;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(cuda_input_image.data);
    free(h_mosaic_sum);
    free(h_mosaic_value);
    CUDA_CALL(cudaFree(d_mosaic_value));
    CUDA_CALL(cudaFree(d_mosaic_sum));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_global_pixel_sum));
}


