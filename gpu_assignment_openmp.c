#include "openmp.h"
#include "helper.h"
#include <stdlib.h>
#include <string.h>

//#define OMP_OPTIMIZE

///
/// Algorithm storage
///



// Declare the input image and output image
Image openmp_input_image;
Image openmp_output_image;

// Declare the coordinates in x and y direction for tiles
unsigned int openmp_TILES_X, openmp_TILES_Y;
unsigned long long* openmp_mosaic_sum;

// Declare the variable for the summing of each tile
unsigned char* openmp_mosaic_value;








//////////////////////////////////////////////
// Define a function for memory allocation ///
//////////////////////////////////////////////

void openmp_begin(const Image* input_image) {

    // compute the number of pixels in image
    int LEN = input_image->width * input_image->height * input_image->channels;

    // compute the number of tiles in x and y directions
    openmp_TILES_X = input_image->width / TILE_SIZE;
    openmp_TILES_Y = input_image->height / TILE_SIZE;


    // Allocate buffer for calculating the sum of each tile mosaic
    openmp_mosaic_sum = (unsigned long long*)malloc(openmp_TILES_X * openmp_TILES_Y * input_image->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    openmp_mosaic_value = (unsigned char*)malloc(openmp_TILES_X * openmp_TILES_Y * input_image->channels * sizeof(unsigned char));

    // Allocate copy of input image
    openmp_input_image = *input_image;
    openmp_input_image.data = (unsigned char*)malloc(LEN * sizeof(unsigned char));
    memcpy(openmp_input_image.data, input_image->data, LEN * sizeof(unsigned char));

    // Allocate output image
    openmp_output_image = *input_image;
    openmp_output_image.data = (unsigned char*)malloc(LEN * sizeof(unsigned char));
}









////////////////////////////////////
// Define a function for stage 1 ///
////////////////////////////////////

void openmp_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);



    // Reset all memory for mosaic sum to 0
    memset(openmp_mosaic_sum, 0, openmp_TILES_X * openmp_TILES_Y * openmp_input_image.channels * sizeof(unsigned long long));



    // Parallel the for loops by openmp, here I choose 4 threads by trying a range of different threads numbers
#pragma omp parallel num_threads(4)
    {

        // Declare four counters for following for loops:
        // t_y, t_x represent the y and x coordinates for a tile in the whole image
        // p_y, p_x represent the y and x coordinates for a pixel in one tile
        int t_y = 0;
        int t_x = 0;
        int p_y = 0;
        int p_x = 0;

#pragma omp for
        // Here, I reverse the iteration order from t_x - t_y - p_x - p_y to t_y - t_x - p_x - p_y.
        // Because in this order, each thread could access cache for reading data from "open_input_image.data" more effciently

        // Then each thread will calculate the sum for one tile. After completing the calculation for current tile, this thread will
        // move to next tile.
        for (t_y = 0; t_y < openmp_TILES_Y; ++t_y) {
            for (t_x = 0; t_x < openmp_TILES_X; ++t_x) {
                // Define the tile_index and tile_offset
                const unsigned int tile_index = (t_y * openmp_TILES_X + t_x) * openmp_input_image.channels;
                const unsigned int tile_offset = (t_y * openmp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * openmp_input_image.channels;

                for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
                        const unsigned int pixel_offset = (p_y * openmp_input_image.width + p_x) * openmp_input_image.channels;

                        // calculate the the sum of each channel (we have rgb 3 channels) in a tile and store the value to "openmp_mosaic_sum"
                        openmp_mosaic_sum[tile_index] += openmp_input_image.data[tile_offset + pixel_offset];
                        openmp_mosaic_sum[tile_index + 1] += openmp_input_image.data[tile_offset + pixel_offset + 1];
                        openmp_mosaic_sum[tile_index + 2] += openmp_input_image.data[tile_offset + pixel_offset + 2];

                    }
                }
            }
        }



        // Here is an old version of stage 1, which using original iteration order - t_x, t_y, p_x, p_y

        //for (t_x = 0; t_x < openmp_TILES_X; ++t_x) {
        //    for (t_y = 0; t_y < openmp_TILES_Y; ++t_y) {
        //        // Define the tile_index and tile_offset
        //        const unsigned int tile_index = (t_y * openmp_TILES_X + t_x) * openmp_input_image.channels;
        //        const unsigned int tile_offset = (t_y * openmp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * openmp_input_image.channels;

        //        for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
        //            for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
        //                const unsigned int pixel_offset = (p_y * openmp_input_image.width + p_x) * openmp_input_image.channels;

        //                // calculate the the sum of each channel (we have rgb 3 channels) in a tile and store the value to "openmp_mosaic_sum"
        //                openmp_mosaic_sum[tile_index] += openmp_input_image.data[tile_offset + pixel_offset];
        //                openmp_mosaic_sum[tile_index + 1] += openmp_input_image.data[tile_offset + pixel_offset + 1];
        //                openmp_mosaic_sum[tile_index + 2] += openmp_input_image.data[tile_offset + pixel_offset + 2];

        //            }
        //        }
        //    }
        //}


    }



#ifdef VALIDATION
    validate_tile_sum(&openmp_input_image, openmp_mosaic_sum);
#endif
}







////////////////////////////////////
// Define a function for stage 2 ///
////////////////////////////////////

void openmp_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);


    // Declare and initialize a variable to store the three sum values (RGB) for the whole image.
    unsigned long long whole_image_sum[3] = { 0, 0, 0 };


#pragma omp parallel num_threads(4)
    {
        // Here, declare a "t" to represent the tile index
        int t;

        // calculate the average pixel value for each tile
#pragma omp for
        for (t = 0; t < openmp_TILES_X * openmp_TILES_Y; ++t) {
            // For each tile, it has three channels - rgb channel.
            // Calculate the average values 
            openmp_mosaic_value[t * openmp_input_image.channels] = (unsigned char)(openmp_mosaic_sum[t * openmp_input_image.channels] / TILE_PIXELS);
            openmp_mosaic_value[t * openmp_input_image.channels + 1] = (unsigned char)(openmp_mosaic_sum[t * openmp_input_image.channels + 1] / TILE_PIXELS);
            openmp_mosaic_value[t * openmp_input_image.channels + 2] = (unsigned char)(openmp_mosaic_sum[t * openmp_input_image.channels + 2] / TILE_PIXELS);
        }
        // "omp for" has a implicit barrier


        int ch;
        // calculate the pixel sum for the whole image
#pragma omp for
        for (ch = 0; ch < openmp_input_image.channels; ++ch) {
            for (unsigned int t = 0; t < openmp_TILES_X * openmp_TILES_Y; ++t) {
                whole_image_sum[ch] += openmp_mosaic_value[t * openmp_input_image.channels + ch];
            }
        }
    }


    // Here is a for loop which only has three iterations, therefore, I think it would be better to process it without OPENMP.
    // This for loop calculate the average values for rgb channels of the whole image.
    for (int ch = 0; ch < openmp_input_image.channels; ++ch) {
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (openmp_TILES_X * openmp_TILES_Y));
    }


#ifdef VALIDATION
    validate_compact_mosaic(openmp_TILES_X, openmp_TILES_Y, openmp_mosaic_sum, openmp_mosaic_value, output_global_average);
#endif    
}








////////////////////////////////////
// Define a function for stage 3 ///
////////////////////////////////////

void openmp_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(input_image, compact_mosaic, output_image);


    int t_y;

#pragma omp parallel for num_threads(4)
    for (t_y = 0; t_y < openmp_TILES_Y; ++t_y) {
        for (unsigned int t_x = 0; t_x < openmp_TILES_X; ++t_x) {
            // Calculate the tile_index and tile_offset
            const unsigned int tile_index = (t_y * openmp_TILES_X + t_x) * openmp_input_image.channels;
            const unsigned int tile_offset = (t_y * openmp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * openmp_input_image.channels;

            // declare a 3 byte length array to store the rgb values for current tile:
            unsigned char current_tile_rgb[3];
            // write the rgb values for current tile from "openmp_mosaic_value"
            current_tile_rgb[0] = openmp_mosaic_value[tile_index];
            current_tile_rgb[1] = openmp_mosaic_value[tile_index + 1];
            current_tile_rgb[2] = openmp_mosaic_value[tile_index + 2];

            for (unsigned int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                for (unsigned int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                    const unsigned int pixel_offset = (p_y * openmp_input_image.width + p_x) * openmp_input_image.channels;
                    
                    // broadcast the rgb values of current tile to all the pixels in this tile
                    // I hard-code the channel index because we only have three channels - rgb channels
                    openmp_output_image.data[tile_offset + pixel_offset] = current_tile_rgb[0];
                    openmp_output_image.data[tile_offset + pixel_offset + 1] = current_tile_rgb[1];
                    openmp_output_image.data[tile_offset + pixel_offset + 2] = current_tile_rgb[2];
                }
            }



            //// Here is an old version which using "memcpy" to broadcast 3-byte length average pixel values to each tile.

            //for (unsigned int p_y = 0; p_y < TILE_SIZE; ++p_y) {
            //    for (unsigned int p_x = 0; p_x < TILE_SIZE; ++p_x) {
            //        const unsigned int pixel_offset = (p_y * openmp_input_image.width + p_x) * openmp_input_image.channels;

            //        // Broadcast the RGB values of current tile to all the pixels in this tile
            //        // I hard-code the channel index because we only have three channels - RGB channels
            //        memcpy(openmp_output_image.data + tile_offset + pixel_offset, openmp_mosaic_value + tile_index, openmp_input_image.channels);
            //    }
            //}


        }
    }

#ifdef VALIDATION
    validate_broadcast(&openmp_input_image, openmp_mosaic_value, &openmp_output_image);
#endif    
}








void openmp_end(Image* output_image) {
    // Store return value/
    output_image->width = openmp_output_image.width;
    output_image->height = openmp_output_image.height;
    output_image->channels = openmp_output_image.channels;
    memcpy(output_image->data, openmp_output_image.data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char));
    // Release allocations
    free(openmp_output_image.data);
    free(openmp_input_image.data);
    free(openmp_mosaic_value);
    free(openmp_mosaic_sum);
}
