#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <cmath>
// -------------
// pure C++ code
// -------------
using namespace std;

float interpolate_x(double thickness, double x, double y, double z, double nx, double ny, double nz, int t, vector<double>* data, int data_0, int data_1, int data_2){ 
    int xt = (int)(x+t);
    double yt = y + (xt - x) * ny / nx;
    double zt = z + (xt - x) * nz / nx;

    if (( pow((xt - x), 2) + pow((yt - y),2) + pow((zt - z), 2) > pow(thickness, 2)) || (
        xt < 0 || xt >= data_2 || yt < 0 || yt >= data_1 || zt < 0 || zt >= data_0))
        return -1;

    int y0 = (int)yt;
    int y1 = y0 + 1;
    int z0 = (int)zt;
    int z1 = z0 + 1;

    double res = (*data)[z0 * data_1 * data_2 + y0 * data_2 + xt] * (z1 - zt) * (y1 - yt);
    if (z1 < data_0)
        res += (*data)[z1 * data_1 * data_2 + y0 * data_2 + xt] * (zt - z0) * (y1 - yt);
    if (y1 < data_1)
        res += (*data)[z0 * data_1 * data_2 + y1 * data_2 + xt] * (z1 - zt) * (yt - y0);
    if (z1 < data_0 && y1 < data_1)
        res += (*data)[z1 * data_1 * data_2 + y1 * data_2 + xt] * (zt - z0) * (yt - y0);
    return res;               
}


float interpolate_y(double thickness, double x, double y, double z, double nx, double ny, double nz, int t, vector<double>* data, int data_0, int data_1, int data_2){ 
    int yt = (int)(y+t);
    double xt = x + (yt - y) * nx / ny;
    double zt = z + (yt - y) * nz / ny;

    if (( pow((xt - x), 2) + pow((yt - y),2) + pow((zt - z), 2) > pow(thickness, 2)) || (
        xt < 0 || xt >= data_2 || yt < 0 || yt >= data_1 || zt < 0 || zt >= data_0))
        return -1;

    int x0 = (int)xt;
    int x1 = x0 + 1;
    int z0 = (int)zt;
    int z1 = z0 + 1;

    double res = (*data)[z0 * data_1 * data_2 + yt * data_2 + x0] * (z1 - zt) * (x1 - xt);
    if (z1 < data_0)
        res += (*data)[z1 * data_1 * data_2 + yt * data_2 + x0] * (zt - z0) * (x1 - xt);
    if (x1 < data_2)
        res += (*data)[z0 * data_1 * data_2 + yt * data_2 + x1] * (z1 - zt) * (xt - x0);
    if (z1 < data_0 && x1 < data_2)
        res += (*data)[z1 * data_1 * data_2 + yt * data_2 + x1] * (zt - z0) * (xt - x0);
    return res;               
}

float interpolate_z(double thickness, double x, double y, double z, double nx, double ny, double nz, int t, vector<double>* data, int data_0, int data_1, int data_2){ 
    int zt = (int)(z+t);
    double xt = x + (zt - z) * nx / nz;
    double yt = y + (zt - z) * ny / nz;

    if (( pow((xt - x), 2) + pow((yt - y),2) + pow((zt - z), 2) > pow(thickness, 2)) || (
        xt < 0 || xt >= data_2 || yt < 0 || yt >= data_1 || zt < 0 || zt >= data_0))
        return -1;

    int x0 = (int)xt;
    int x1 = x0 + 1;
    int y0 = (int)yt;
    int y1 = y0 + 1;

    double res = (*data)[zt * data_1 * data_2 + y0 * data_2 + x0] * (y1 - yt) * (x1 - xt);
    if (y1 < data_1)
        res += (*data)[zt * data_1 * data_2 + y1 * data_2 + x0] * (yt - y0) * (x1 - xt);
    if (x1 < data_2)
        res += (*data)[zt * data_1 * data_2 + y0 * data_2 + x1] * (y1 - yt) * (xt - x0);
    if (y1 < data_1 && x1 < data_2)
        res += (*data)[zt * data_1 * data_2 + y1 * data_2 + x1] * (yt - y0) * (xt - x0);
    return res;               
}
// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::array c_ray_sum(int hp, int wp, py::array_t<double, py::array::c_style | py::array::forcecast> xs, py::array_t<double, py::array::c_style | py::array::forcecast> ys,
py::array_t<double, py::array::c_style | py::array::forcecast> dxdus, py::array_t<double, py::array::c_style | py::array::forcecast> dxdvs,
py::array_t<double, py::array::c_style | py::array::forcecast> dydus, py::array_t<double, py::array::c_style | py::array::forcecast> dydvs,
float pixel_size, float thickness, py::array_t<double, py::array::c_style | py::array::forcecast> data)
{
    vector<double> panoramic_image;
    for(int i = 0; i < hp*wp; i++){
        panoramic_image.push_back(0);
    }
    int shape_1 = xs.shape()[1];
    int data_0 = data.shape()[0];
    int data_1 = data.shape()[1];
    int data_2 = data.shape()[2];

    vector<double> new_xs(xs.size());
    memcpy(new_xs.data(),xs.data(),xs.size()*sizeof(double));
    
    vector<double> new_ys(ys.size());
    memcpy(new_ys.data(),ys.data(),ys.size()*sizeof(double));
    
    vector<double> new_dxdus(dxdus.size());
    memcpy(new_dxdus.data(),dxdus.data(),dxdus.size()*sizeof(double));
    
    vector<double> new_dxdvs(dxdvs.size());
    memcpy(new_dxdvs.data(),dxdvs.data(),dxdvs.size()*sizeof(double));
    
    vector<double> new_dydus(dydus.size());
    memcpy(new_dydus.data(),dydus.data(),dydus.size()*sizeof(double));
    
    vector<double> new_dydvs(dydvs.size());
    memcpy(new_dydvs.data(),dydvs.data(),dydvs.size()*sizeof(double));

    vector<double> new_data(data.size());
    memcpy(new_data.data(),data.data(),data.size()*sizeof(double));
    for(int v = 0; v < hp; v ++){
        cout << v << " / " << hp << '\r' << flush;
        for(int u = 0; u < wp; u ++){
            double x = new_xs[u*shape_1 + v];
            double y = new_ys[u*shape_1 + v];
            double z = v * pixel_size;
            double nx = new_dydus[u * shape_1 + v] * pixel_size;
            double ny = -new_dxdus[u * shape_1 + v] * pixel_size;
            double nz = new_dxdus[u * shape_1 + v] * new_dydvs[u * shape_1 + v] - new_dydus[u * shape_1 + v] * new_dxdvs[u * shape_1 + v];
            double size = sqrt(nx * nx + ny * ny + nz * nz);
            nx /= size;
            ny /= size;
            nz /= size;
            
            if (abs(nx) >= abs(ny) && abs(nx) >= abs(nz)){
                int t = 1;
                while(1){
                    double res = interpolate_x(thickness, x, y, z, nx, ny, nz, t, &new_data, data_0, data_1, data_2);
                    if(res >= 0){
                        panoramic_image[v * wp + u] += res;
                        t += 1;
                    }
                    else{
                        break;
                    }
                }

                t = 0;
                while(1){
                    double res = interpolate_x(thickness, x, y, z, nx, ny, nz, t, &new_data, data_0, data_1, data_2);
                    if(res >= 0){
                        panoramic_image[v * wp + u] += res;
                        t -= 1;
                    }
                    else{
                        break;
                    }
                }
            }
            else if((abs(ny) >= abs(nx))&&(abs(ny) >= abs(nz))){
                int t = 1;
                while(1){
                    double res = interpolate_y(thickness, x, y, z, nx, ny, nz, t, &new_data, data_0, data_1, data_2);
                    if(res >= 0){
                        panoramic_image[v * wp + u] += res;
                        t += 1;
                    }
                    else
                        break;
                }

                t = 0;
                while(1){
                    double res = interpolate_y(thickness, x, y, z, nx, ny, nz, t, &new_data, data_0, data_1, data_2);
                    if(res >= 0){
                        panoramic_image[v * wp + u] += res;
                        t -= 1;
                    }
                    else{
                        break;
                    }
                }
            }
            else{
                int t = 1;
                while(1){
                    double res = interpolate_z(thickness, x, y, z, nx, ny, nz, t, &new_data, data_0, data_1, data_2);
                    if(res >= 0){
                        panoramic_image[v * wp + u] += res;
                        t += 1;
                    }
                    else
                        break;
                }

                t = 0;
                while(1){
                    double res = interpolate_z(thickness, x, y, z, nx, ny, nz, t, &new_data, data_0, data_1, data_2);
                    if(res >= 0){
                        panoramic_image[v * wp + u] += res;
                        t -= 1;
                    }
                    else{
                        break;
                    }
                }
            }
        }
    }
    ssize_t              ndim    = 2;
    std::vector<ssize_t> shape   = { hp, wp };
    std::vector<ssize_t> strides = { (ssize_t)(sizeof(double)*wp) , sizeof(double) };
//   return 2-D NumPy array
    return py::array(py::buffer_info(
        panoramic_image.data(),                           /* data as contiguous array  */
        sizeof(double),                          /* size of one scalar        */
        py::format_descriptor<double>::format(), /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
    ));
}

// wrap as Python module
PYBIND11_MODULE(ray_sum,m)
{
  m.doc() = "pybind11 example plugin";

  m.def("ray_sum", &c_ray_sum, "Calculate the length of an array of vectors");
}
