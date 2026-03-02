/***********************************************************************************

    MIT License

    Copyright (c) 2020 masterkeech

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

***********************************************************************************/

/** \mainpage PRefToMotion C++ Plugin documentation
 *  PRefToMotion is a C++ plugin for Foundry's Nuke software that calculates the motion vectors
 *  from a position reference map or similar. It uses the nanoflann kd-tree under the hood to calculate the
 *  nearest neighbours from the target to the source frame.
 *
 *  PRefToMotion requires compiling using CMake, gcc with c++11 and a copy of Nuke's libraries to link against
 *  which can be found at https://www.foundry.com/product-downloads
 *
 *  See:
 *   - <a href="https://github.com/masterkeech/PRefToMotion">PRefToMotion repository</a>
 *   - <a href="https://github.com/jlblancoc/nanoflann">Nanoflann repository</a>
 */

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/DeepOp.h"
#include "DDImage/Knobs.h"
#include "DDImage/Tile.h"

#include "nanoflann.hpp"
#include <chrono>
#include <iostream>
#include <cmath>

#define APPROX_ZERO 0.00000001

static const char *CLASS = "PRefToMotion";

using namespace nanoflann;
using namespace DD::Image;

template<typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z;
        T pos_x, pos_y;
    };

    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template<class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
};

const char *modes[] = {"stmap", "uvmap", "pixel", nullptr};


typedef KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<float, PointCloud<float> >,
        PointCloud<float>,
        3 /* dim */
> my_kd_tree_t;

typedef std::shared_ptr<my_kd_tree_t> my_kd_tree_t_ptr;
typedef PointCloud<float> point_cloud_float;
typedef std::shared_ptr<point_cloud_float> point_cloud_float_ptr;

class PRefToMotion : public Iop {
    ChannelSet _channels;
    Channel _mask_channel;
    Channel _out_channels[2];

    int _mode;
    int _samples;
    U64 _source_hash;
    int _source_frame;
    bool _rebuild;
    
    // User-adjustable parameter
    float _max_distance;        // Maximum search distance (0 = unlimited)

    Lock _lock;
    my_kd_tree_t_ptr _kd_tree_ptr;
    point_cloud_float_ptr _point_cloud_ptr;

public:
    explicit PRefToMotion(Node *node) : Iop(node), 
        _channels(Mask_RGB), 
        _mask_channel(Chan_Black), 
        _mode(0), 
        _samples(3), 
        _source_hash(0x0), 
        _source_frame(1001), 
        _rebuild(true),
        _max_distance(0.0f),
        _kd_tree_ptr(nullptr),
        _point_cloud_ptr(nullptr)
    {
        _out_channels[0] = Chan_U;
        _out_channels[1] = Chan_V;
    }

    int maximum_inputs() const override { return 2; }

    int minimum_inputs() const override { return 2; }

    int optional_input() const override { return 1; }

    const char* input_label(int input, char*) const override { return input == 0 ? "" : "mask"; }

    /// split the inputs so that we have a target frame and a source frame
    int split_input(int n) const override { return 2; }

    /// The time for image input n :-
    const OutputContext &inputContext(int i, int n, OutputContext &context) const override
    {
        context = outputContext();
        if (n == 1)
        {
            // in order for the source frame to be correct you must have the knob early store flag set
            context.setFrame(_source_frame);
        }
        return context;
    }

    const char *node_help() const override
    {
        return "Converts a pref or similar pass to a backwards mapping that can be used with an STMap or IDistort.\n\n"
               "Uses a 3-dimensional KD-tree to find the nearest neighbours from a source frame and outputs "
               "the resulting motion as either ST, UV channels, or pixel coordinates.\n\n"
               "Options:\n"
               "- samples: Number of neighbors to blend (default: 3)\n"
               "- max distance: Ignore neighbors beyond this distance (0=unlimited)";
    }

    const char *Class() const override
    {
        return CLASS;
    }

    void knobs(Knob_Callback f) override
    {
        Input_ChannelSet_knob(f, &_channels, 0, "channels", "pref channels");
        Tooltip(f, "Channels to calculate the motion vectors from, the optional 4th channel is used as a mask.");

        Channel_knob(f, _out_channels, 2, "uv_channels", "uv channels");
        Tooltip(f, "Channels to store the uv data that is being generated.");

        Channel_knob(f, &_mask_channel, 1, "mask");
        Tooltip(f, "Channel to use as a mask");

        Int_knob(f, &_source_frame, "source_frame", "source frame");
        Tooltip(f, "Source frame to calculate the motion vectors from.");
        // this is needed so that we can query the source frame from the input context
        SetFlags(f, Knob::EARLY_STORE);

        Int_knob(f, &_samples, "samples");
        Tooltip(f, "Number of neighbouring samples used to calculate motion using a weighted average.");
        SetRange(f, 1, 16);
        ClearFlags(f, Knob::STARTLINE);

        Enumeration_knob(f, &_mode, modes, "mode");
        Tooltip(f, "Generate the motion as either:\n  - st map (normalised)\n"
                   "  - uv map (vectors)\n  - pixels (source)");
        ClearFlags(f, Knob::STARTLINE);

        // Performance section - only max_distance exposed
        Divider(f, "Performance");
        
        Float_knob(f, &_max_distance, "max_distance", "max radius");
        Tooltip(f, "Maximum search distance (squared). Neighbors beyond this distance are ignored. "
                   "0 = unlimited. Useful for rejecting distant mismatches.");
        SetRange(f, 0.0, 1.0);
    }

    int knob_changed(Knob *k) override
    {
        // reset the kd tree if the channels or source frame used to calculate the motion vectors from change
        if (k && (k->is("channels") || k->is("source_frame") || k->is("mask")))
        {
            _kd_tree_ptr.reset();
            _rebuild = true;
            return 1;
        }
        return Iop::knob_changed(k);
    }

    void _validate(bool for_real) override
    {
#ifndef NDEBUG
        std::cout << "_validate(" << for_real << ")" << std::endl;
#endif
        if (input(0))
        {
            copy_info();

            // validate the input at time _source_frame
            input(0, 1)->force_validate(true);

            // check the input at the source frame to see if it's changed and requires an update
            U64 hash = input(0, 1)->hash().value();
            if (hash != _source_hash)
            {
                _kd_tree_ptr.reset();
                _source_hash = hash;
                _rebuild = true;
            }

            // create the uv channel set
            ChannelSet uv_channels(_out_channels[0]);
            uv_channels += _out_channels[1];

            ChannelSet input_mask_channels = info_.channels();

            ChannelSet out_channels = info_.channels();
            out_channels += uv_channels;

            // set the out channels to include uv and make sure to turn them on
            set_out_channels(out_channels);
            info_.turn_on(uv_channels);

            // if we have an input1 than let's validate
            if (input(1))
            {
                input(1)->validate(for_real);
            }

            // check to see if input1 truly is our first input, sometimes it's actually input0?
            if (input(1) && Op::input(1) != default_input(1) && input(0)->firstOp() != input(1)->firstOp())
            {
                input_mask_channels = input(1)->channels();

                // validate the input at time _source_frame
                input(1, 1)->force_validate(true);
            }

            // check to make sure we're using the correct channels
            if (!(_channels & info_.channels()) || (_mask_channel != Chan_Black && !(ChannelSet(_mask_channel) & input_mask_channels)) || _channels.empty())
            {
                set_out_channels(Mask_None);
            }
        } else {
            set_out_channels(Mask_None);
        }
    }

    void _request(int x, int y, int r, int t, ChannelMask channels, int count) override
    {
#ifndef NDEBUG
        std::cout << "_request(" << x << ", " << y << ", " << r << ", " << t << ", " << channels << ", " << count << ")" << std::endl;
#endif
        if (Iop *iop = input(0))
        {
            ChannelSet in_channels = channels;
            // we don't want to request the out channels, they are generated only
            in_channels -= _out_channels[0];
            in_channels -= _out_channels[1];
            in_channels += _channels;

            // check to see if we're using a mask from input 0
            bool use_input0 = !input(1) || Op::input(1) == default_input(1) || input(0)->firstOp() == input(1)->firstOp();
            if (_mask_channel != Chan_Black && use_input0)
            {
                in_channels += _mask_channel;
            }

            iop->request(x, y, r, t, in_channels, count);

            // request the whole source frame and only the _channels
            Info source_info = input(0, 1)->info();
            ChannelSet source_channels = _mask_channel != Chan_Black && use_input0 ? _channels + _mask_channel : _channels;
            input(0, 1)->request(source_info.x(), source_info.y(), source_info.r(), source_info.t(),
                                 source_channels, count);

            if (_mask_channel != Chan_Black && !use_input0)
            {
                input(1)->request(x, y, r, t, ChannelSet(_mask_channel), count);
                input(1, 1)->request(source_info.x(), source_info.y(), source_info.r(), source_info.t(),
                                     ChannelSet(_mask_channel), count);
            }
        }
    }

    // Helper function to calculate weight using inverse distance squared (default)
    inline float calculateWeight(float dist_sqr) const
    {
        // Clamp to avoid division by zero
        float d = std::max<float>(dist_sqr, (float)APPROX_ZERO);
        // Use inverse distance squared weighting (original default behavior)
        return 1.0f / d;
    }

    void engine(int y, int x, int r, ChannelMask channels, Row &outrow) override
    {
        if (!input(0) || aborted())
        {
            return;
        }

        // check to see if the input 1 has the mask channel, if so we're using that
        ChannelSet mask_Channel(_mask_channel);
        int mask_input = input(1, 0) && input(1, 0)->channels() & mask_Channel ? 1 : 0;

        // check to see if the kd tree has been reset and needs the index to be rebuilt
        if (_rebuild)
        {
            Guard guard(_lock);
            if (_rebuild)
            {
#ifndef NDEBUG
                auto start_time = std::chrono::high_resolution_clock::now();
#endif
                _point_cloud_ptr = std::make_shared<point_cloud_float>();
                // grab the data from input1 which is at _source_frame
                Info source_info = input(1)->info();

                Tile tile(*input(0, 1), source_info.x(), source_info.y(), source_info.r(), source_info.t(), _channels, true);

                if (aborted())
                {
                    return;
                }

                Tile mask_tile(*input(mask_input, 1), source_info.x(), source_info.y(), source_info.r(), source_info.t(), mask_Channel, true);

                if (aborted())
                {
                    return;
                }

                // clear the point cloud and reserve the memory to a quarter of the size to avoid vector resizing
                _point_cloud_ptr->pts.reserve(((source_info.r() - source_info.x()) * (source_info.t() - source_info.y())) / 4);

                // loop through the whole tile to get the x,y,z values in order to generate the kd-tree
                for (int ty = source_info.y(); ty < source_info.t(); ++ty)
                {
                    for (int tx = source_info.x(); tx < source_info.r(); ++tx)
                    {
                        // check to see if we're using the mask channel
                        if (_mask_channel == Chan_Black || *(mask_tile[_mask_channel][ty] + tx) != 0.0f)
                        {
                            float values[3] = {0.0f, 0.0f, 0.0f};
                            Channel z = _channels.first();
                            for (unsigned int i = 0; i < std::min<unsigned int>(_channels.size(), 3); ++i, z = _channels.next(z))
                            {
                                values[i] = tile[z][ty][tx];
                            }
                            // only add non-zero values into the point cloud for speed
                            if (values[0] != 0.0f || values[1] != 0.0f || values[2] != 0.0f)
                            {
                                PointCloud<float>::Point pt = {values[0], values[1], values[2], (float) tx + 0.5f, (float) ty + 0.5f};
                                _point_cloud_ptr->pts.push_back(pt);
                            }
                        }
                    }
                }
#ifndef NDEBUG
                auto build_time = std::chrono::high_resolution_clock::now();
#endif
                // Use default max_leaf of 10
                _kd_tree_ptr = std::make_shared<my_kd_tree_t>(3 /*dim*/, *_point_cloud_ptr, KDTreeSingleIndexAdaptorParams(10));
                _kd_tree_ptr->buildIndex();
#ifndef NDEBUG
                auto finish_time = std::chrono::high_resolution_clock::now();
                std::cout << "--------------- kd tree ---------------" << std::endl;
                std::cout << "           bbox: " << source_info.x() << ", " << source_info.y() << ", " << source_info.r() << ", " << source_info.t() << std::endl;
                std::cout << "       channels: " << channels << std::endl;
                std::cout << "  PRef channels: " << _channels << std::endl;
                std::cout << "   mask channel: " << mask_Channel << std::endl;
                std::cout << "     num points: " << _point_cloud_ptr->pts.size() << std::endl;
                std::cout << "        samples: " << _samples << std::endl;
                std::cout << "   source frame: " << _source_frame << std::endl;
                std::cout << "  current frame: " << (int) outputContext().frame() << std::endl;
                std::cout << "   max distance: " << _max_distance << std::endl;
                std::cout << "--------------- timing ---------------" << std::endl;
                std::cout << "     query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(build_time - start_time).count() << " ms" << std::endl;
                std::cout << "     build time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - build_time).count() << " ms" << std::endl;
                std::cout << "     total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count() << " ms" << std::endl;
                std::cout.flush();
#endif
                _rebuild = false;
            }
        }

        // grab our input row that we are going to read the channels and target channels from
        ChannelSet actual_channels = channels;
        actual_channels -= _out_channels[0];
        actual_channels -= _out_channels[1];

        // grab our input row that we are going to read the channels and target channels from
        Row row(x, r);
        row.get(input0(), y, x, r, actual_channels + _channels);

        Row mask_row(x, r);
        mask_row.get(*input(mask_input, 0), y, x, r, mask_Channel);

        // pass through all the channels except for uv which we will be calculating
        foreach(z, actual_channels)
        {
            outrow.copy(row, z, x, r);
        }

        if (aborted())
        {
            return;
        }

        // these are used by the kd tree to return the indices of the neighbours and the squared distance
        std::unique_ptr<size_t[]> ret_index(new size_t[_samples]);
        std::unique_ptr<float[]> out_dist_sqr(new float[_samples]);

        // loop through the row's pixels
        for (int xx = x; xx < r; ++xx)
        {
            float query[3] = {0.0f, 0.0f, 0.0f};
            Channel z = _channels.first();
            for (unsigned int i = 0; i < std::min<unsigned int>(_channels.size(), 3); ++i, z = _channels.next(z))
            {
                query[i] = row[z][xx];
            }

            // our uv values that we're calculating
            float u = 0.0f, v = 0.0f;

            // check to see if the channel has an alpha / w to use as a mask
            if (_mask_channel == Chan_Black || mask_row[_mask_channel][xx] != 0.0f)
            {
                // create a new result set that will return the distance and index to the returned samples
                nanoflann::KNNResultSet<float> resultSet(_samples);
                resultSet.init(&ret_index[0], &out_dist_sqr[0]);

                _kd_tree_ptr->findNeighbors(resultSet, &query[0], nanoflann::SearchParameters());

                // calculate the weighted average using inverse distance squared
                float total_weights = 0.0f;
                unsigned int valid_samples = 0;
                
                // First pass: calculate total weight (filtering by max_distance if set)
                for (unsigned int i = 0; i < resultSet.size(); ++i)
                {
                    // Skip samples beyond max_distance if max_distance is set
                    if (_max_distance > 0.0f && out_dist_sqr[i] > _max_distance)
                    {
                        continue;
                    }
                    total_weights += calculateWeight(out_dist_sqr[i]);
                    valid_samples++;
                }
                
                // Second pass: accumulate weighted positions
                if (total_weights > 0.0f)
                {
                    for (unsigned int i = 0; i < resultSet.size(); ++i)
                    {
                        // Skip samples beyond max_distance if max_distance is set
                        if (_max_distance > 0.0f && out_dist_sqr[i] > _max_distance)
                        {
                            continue;
                        }
                        float weight = calculateWeight(out_dist_sqr[i]) / total_weights;
                        u += weight * _point_cloud_ptr->pts[ret_index[i]].pos_x;
                        v += weight * _point_cloud_ptr->pts[ret_index[i]].pos_y;
                    }
                }
            }

            // calculate the correct u,v based on the mode
            if (_mode == 0)
            {
                // stmap
                u = u / (float) format().width();
                v = v / (float) format().height();
            }
            else if (_mode == 1)
            {
                // uvmap
                u = u - (float) xx;
                v =  v - (float) y;
            }

            // set the final uv values of the outrow
            *(outrow.writable(_out_channels[0]) + xx) = u;
            *(outrow.writable(_out_channels[1]) + xx) = v;
        }
    }

    static const Description d;
};

static Op *build(Node *node) { return new PRefToMotion(node); }

const Op::Description PRefToMotion::d(::CLASS, "Transform/PRefToMotion", build);