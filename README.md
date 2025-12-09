# PRefToMotion

C++ plugin for Foundry's Nuke software that calculates the motion vectors from a source frame to the current frame using a position reference pass or similar. It uses the [nanoflann](https://github.com/jlblancoc/nanoflann) **v1.8** kd-tree library under the hood to do a nearest neighbour lookup using the xyz value of the supplied pref channel.

| ![alt text](examples/PRefToMotion_example.png) |
| --- |
| **example image** a smiley face drawn at frame 1 and warped to 5, 10 and 15 using a position reference pass |

# usage

PRefToMotion(.dylib|.so|.dll) should now be built and installed into your user's .nuke folder, so now it's just a matter of launching nuke and opening the test script `pref_motion_example.nk`

# parameters

Very simple list of parameters to interact with:

* **pref channels**: channels that hold the xyz pref or similar data. If a 4th channel is supplied, it will be used as a mask.
* **uv channels**: channels to store the uv data that is being generated
* **mask**: optional mask channel
* **source frame**: source frame to calculate the motion vectors from
* **samples**: number of nearest neighbours to look up, used to calculate a weighted average from the squared distance to the lookup point
* **mode**: generates the motion as either an st map (normalised), uv map (vectors) or as the source pixels

## Performance

* **build threads**: number of threads to use for building the kd-tree (0 = auto)
* **use rknn**: enable radius-based k-nearest neighbor search for improved performance
* **max radius**: maximum search radius for rknn queries (only used when "use rknn" is enabled). Smaller values can significantly improve query speed by limiting the search space.

# speed

Building the indices and querying a kd-tree with more than a million points can be slow. The new **rknn (radius k-nearest neighbor)** feature with **max radius** parameter provides significant performance improvements by limiting the search space during queries.

```
example timing of building of a kd-tree with 1,063,379 points

--------------- kd tree ---------------
           bbox: 91, 48, 1469, 1023
output channels: rgba,forward.u,forward.v
  pref channels: rgb
     num points: 1063379
        samples: 3
   source frame: 1
  current frame: 15
--------------- timing ---------------
     query time: 208 ms
     build time: 1096 ms
     total time: 1304 ms
```

Performance can be further improved by:
* Enabling **use rknn** checkbox
* Tuning the **max radius** parameter (smaller values = faster queries, but may miss distant neighbors)
* Adjusting **build threads** for faster kd-tree construction

# dependencies

* [nanoflann v1.8](https://github.com/jlblancoc/nanoflann) - A C++11 header-only library for building KD-Trees
