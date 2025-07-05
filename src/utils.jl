# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENSE in the project root.
# ------------------------------------------------------------------

"""
    isthreaded(cond=true)

Return true if `cond`ition is true in the presence of multiple threads.
"""
isthreaded(cond=true) = cond && Threads.nthreads() > 1
